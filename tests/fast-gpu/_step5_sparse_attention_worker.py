"""Compare sparse and dense layers, gradients, and recompute selections on one distributed rank."""

import argparse
import itertools
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from megatron.core import parallel_state, tensor_parallel
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

from miles_plugins.models.step5.indexer import CHECKPOINT_NAMES
from miles_plugins.models.step5.layers import FULL_ATTENTION_ROPE, Step5Attention, Step5RMSNorm
from miles_plugins.models.step5.qkv_layout import fuse_qkv_and_gate, split_qkv_and_gate
from miles_plugins.models.step5.sparse_attention import Step5SparseAttention, Step5SparseCoreAttention

HIDDEN, HEADS, GROUPS, HEAD_DIM = 4096, 64, 4, 192
# Layer 4 of this skip pattern is a full-attention layer, like Step5's layer 3.
SKIP_PATTERN, LAYER_NUMBER = [1, 1, 1, 0], 4
PACKS = {
    # every region selected, so the sparse layer is dense causal attention
    "short": [4104, 1, 333, 1024, 7, 11],
    # the first sequence selects past position 4103; a region grid tail and a lone token
    "long": [5000, 1, 2603, 396],
}


class DenseFullAttention(Step5Attention):
    """Step5's full-attention layer before region selection: dense causal attention, full-attention RoPE."""

    rope = FULL_ATTENTION_ROPE


def _config(tp_size: int, sequence_parallel: bool) -> TransformerConfig:
    return TransformerConfig(
        num_layers=LAYER_NUMBER,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_query_groups=GROUPS,
        kv_channels=HEAD_DIM,
        layernorm_epsilon=1e-5,
        qk_layernorm=True,
        head_wise_attn_gate=True,
        add_bias_linear=False,
        window_size=(511, 0),
        window_attn_skip_freq=SKIP_PATTERN,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=sequence_parallel,
        apply_rope_fusion=True,
        gradient_accumulation_fusion=False,
    )


def _build(module, core_attention, config: TransformerConfig):
    spec = ModuleSpec(
        module=module,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=SelfAttentionSubmodules(
            linear_qkv=TEColumnParallelLinear,
            core_attention=core_attention,
            linear_proj=TERowParallelLinear,
            q_layernorm=Step5RMSNorm,
            k_layernorm=Step5RMSNorm,
        ),
    )
    return build_module(spec, config=config, layer_number=LAYER_NUMBER).cuda()


def _global_weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)

    def randn(*shape: int, std: float) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * std

    return {
        "q_proj": randn(HEADS * HEAD_DIM, HIDDEN, std=0.02),
        "k_proj": randn(GROUPS * HEAD_DIM, HIDDEN, std=0.02),
        "v_proj": randn(GROUPS * HEAD_DIM, HIDDEN, std=0.02),
        "g_proj": randn(HEADS, HIDDEN, std=0.02),
        "o_proj": randn(HIDDEN, HEADS * HEAD_DIM, std=0.02),
        "q_norm": randn(HEAD_DIM, std=0.1),
        "k_norm": randn(HEAD_DIM, std=0.1),
    }


def _indexer_checkpoint() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)

    def randn(*shape: int, std: float) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * std

    return {
        CHECKPOINT_NAMES["q_weight"]: randn(16 * 256, HIDDEN, std=0.02).bfloat16(),
        CHECKPOINT_NAMES["k_weight"]: randn(256, HIDDEN, std=0.02).bfloat16(),
        CHECKPOINT_NAMES["z_weight"]: randn(256, HIDDEN, std=0.02).bfloat16(),
        CHECKPOINT_NAMES["w_weight"]: randn(16, HIDDEN, std=0.02),
        CHECKPOINT_NAMES["q_norm_weight"]: randn(256, std=0.1),
        CHECKPOINT_NAMES["k_norm_weight"]: 1 + randn(256, std=0.1),
        CHECKPOINT_NAMES["k_norm_bias"]: randn(256, std=0.1),
        CHECKPOINT_NAMES["ssmax_s"]: torch.full((HEADS,), 0.085),
    }


def _load_attention(module, weights: dict[str, torch.Tensor], tp_size: int, tp_rank: int) -> None:
    fused = fuse_qkv_and_gate(
        query=weights["q_proj"],
        key=weights["k_proj"],
        value=weights["v_proj"],
        gate=weights["g_proj"],
        num_query_groups=GROUPS,
        tp_size=tp_size,
    )
    with torch.no_grad():
        module.linear_qkv.weight.copy_(fused.chunk(tp_size)[tp_rank])
        module.linear_proj.weight.copy_(weights["o_proj"].chunk(tp_size, dim=1)[tp_rank])
        module.q_layernorm.weight.copy_(weights["q_norm"])
        module.k_layernorm.weight.copy_(weights["k_norm"])


class _Ranks:
    """This rank's slice of the tensors every rank builds whole, and the gathers that undo it."""

    def __init__(self, tp_size: int, sequence_parallel: bool) -> None:
        self.tp_size, self.sequence_parallel = tp_size, sequence_parallel
        self.group = parallel_state.get_tensor_model_parallel_group()
        self.rank = parallel_state.get_tensor_model_parallel_rank()

    def gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        parts = [torch.empty_like(tensor) for _ in range(self.tp_size)]
        dist.all_gather(parts, tensor.contiguous(), group=self.group)
        return torch.cat(parts, dim=dim)

    def tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sequence-parallel layers see their shard of the tokens."""
        return tensor.chunk(self.tp_size)[self.rank] if self.sequence_parallel else tensor

    def all_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.gather(tensor, dim=0) if self.sequence_parallel else tensor

    def summed(self, tensor: torch.Tensor) -> torch.Tensor:
        # q/k norms see only this rank's heads; Megatron sums their gradients across TP
        tensor = tensor.clone()
        dist.all_reduce(tensor, group=self.group)
        return tensor


def _packed(lengths: list[int]) -> PackedSeqParams:
    cu_seqlens = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device="cuda")
    longest = max(lengths)
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=longest,
        max_seqlen_kv=longest,
        qkv_format="thd",
    )


def _call(module, hidden: torch.Tensor, packed: PackedSeqParams) -> torch.Tensor:
    # The model hands every layer its RoPE table, which the per-layer RoPE replaces.
    rotary = module.rotary_pos_emb(packed.max_seqlen_q)
    out, bias = module(hidden, None, rotary_pos_emb=rotary, packed_seq_params=packed)
    assert bias is None
    return out


def _forward_backward(module, ranks: _Ranks, hidden: torch.Tensor, grad: torch.Tensor, packed) -> dict:
    module.zero_grad(set_to_none=True)
    x = ranks.tokens(hidden).clone().requires_grad_()
    out = _call(module, x, packed)
    out.backward(ranks.tokens(grad))
    return _collect_results(module, ranks, x, out)


def _collect_results(module, ranks: _Ranks, x: torch.Tensor, out: torch.Tensor) -> dict:
    qkv = ranks.gather(module.linear_qkv.weight.grad, dim=0)
    q, k, v, g = split_qkv_and_gate(
        qkv, num_query_groups=GROUPS, num_attention_heads=HEADS, head_dim=HEAD_DIM, tp_size=ranks.tp_size
    )
    return {
        "out": ranks.all_tokens(out.detach()),
        "d_hidden": ranks.all_tokens(x.grad),
        "d_q_proj": q,
        "d_k_proj": k,
        "d_v_proj": v,
        "d_g_proj": g,
        "d_o_proj": ranks.gather(module.linear_proj.weight.grad, dim=1),
        "d_q_norm": ranks.summed(module.q_layernorm.weight.grad),
        "d_k_norm": ranks.summed(module.k_layernorm.weight.grad),
    }


def _recomputed_selections(module, ranks: _Ranks, hidden, grad, packed) -> tuple[list[torch.Tensor], dict]:
    """Run checkpointed forward/backward and record Indexer block IDs."""
    recorded = []
    select_regions = module.indexer.select_regions

    def recording(hidden_states, packed_seq_params):
        selection = select_regions(hidden_states, packed_seq_params)
        recorded.append(selection.region_ids.clone())
        return selection

    module.indexer.select_regions = recording
    try:
        module.zero_grad(set_to_none=True)
        x = ranks.tokens(hidden).clone().requires_grad_()
        out = tensor_parallel.checkpoint(lambda h: _call(module, h, packed), False, x)
        out.backward(ranks.tokens(grad))
    finally:
        del module.indexer.select_regions
    return recorded, _collect_results(module, ranks, x, out)


def _reused_selections(module, ranks: _Ranks, hidden, grad, packed) -> tuple[list[torch.Tensor], dict]:
    """Reuse Indexer block IDs during backward recomputation and consume the cache."""
    with patch.object(module, "reuse_indexer_on_recompute", True):
        recorded, result = _recomputed_selections(module, ranks, hidden, grad, packed)
        assert not getattr(packed, "_step5_indexer_selections", {})
    return recorded, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    tp_size = dist.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_size)
    tensor_parallel.model_parallel_cuda_manual_seed(1234)
    ranks = _Ranks(tp_size, args.sequence_parallel)

    config = _config(tp_size, args.sequence_parallel)
    weights = _global_weights()
    dense = _build(DenseFullAttention, TEDotProductAttention, config)
    sparse = _build(Step5SparseAttention, Step5SparseCoreAttention, config)
    for module in (dense, sparse):
        _load_attention(module, weights, tp_size, ranks.rank)
    sparse.indexer.load_checkpoint_layer(_indexer_checkpoint())

    results = {}
    generator = torch.Generator(device="cuda").manual_seed(2)
    for name, lengths in PACKS.items():
        packed = _packed(lengths)
        tokens = sum(lengths)
        hidden = torch.randn(tokens, 1, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator)
        grad = torch.randn(tokens, 1, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.01
        selection = sparse.indexer.select_regions(ranks.tokens(hidden), packed)
        baseline_recorded, baseline = _recomputed_selections(sparse, ranks, hidden, grad, packed)

        reuse_packed = _packed(lengths)
        reuse_recorded, reuse_result = _reused_selections(sparse, ranks, hidden, grad, reuse_packed)

        sparse_result = _forward_backward(sparse, ranks, hidden, grad, packed)
        results[name] = {
            "lengths": lengths,
            "sparse": sparse_result,
            "dense": _forward_backward(dense, ranks, hidden, grad, packed),
            "selection": ranks.gather(selection.region_ids, dim=1),
            "selection_passes": len(baseline_recorded),
            "recomputed_selection_equal": all(torch.equal(ids, selection.region_ids) for ids in baseline_recorded),
            "checkpointed_out_equal": torch.equal(baseline["out"], sparse_result["out"]),
            "reuse_selection_passes": len(reuse_recorded),
            "reuse_selection_equal": all(torch.equal(ids, reuse_recorded[0]) for ids in baseline_recorded),
            "reuse_out_equal": torch.equal(reuse_result["out"], baseline["out"]),
            "reuse_gradient_errors": {
                key: (value.float() - baseline[key].float()).norm() / baseline[key].float().norm().clamp_min(1e-12)
                for key, value in reuse_result.items()
                if key.startswith("d_")
            },
        }

    if dist.get_rank() == 0:
        cpu = {name: {key: _to_cpu(value) for key, value in result.items()} for name, result in results.items()}
        torch.save(cpu, args.output)
        print(f"PASS tp{tp_size}{'-sp' if args.sequence_parallel else ''}", flush=True)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _to_cpu(value):
    if isinstance(value, dict):
        return {key: tensor.cpu() for key, tensor in value.items()}
    return value.cpu() if isinstance(value, torch.Tensor) else value


if __name__ == "__main__":
    main()
