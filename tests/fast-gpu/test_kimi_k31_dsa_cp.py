"""Kimi-K3.1's DSA attention under zigzag context parallelism against the same layer at CP 1.

    torchrun --nproc_per_node=2 tests/fast-gpu/test_kimi_k31_dsa_cp.py

One packed batch, with a sequence longer than the index top-k, runs at CP 1 on every rank and then split
over CP 2. Feeding both runs the CP-1 top-k pins the gathered KV to global packed key positions, the space
rollout indexer replay rebases into; the native run pins the gathered index-K.
"""

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.ci.ci_register import register_cuda_ci

from miles_plugins.models.kimi_k3.model import configure_kimi_k3
from miles_plugins.models.kimi_k31.model import KimiK31DSAAttention, configure_kimi_k31_dsa

register_cuda_ci(est_time=300, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper"], num_gpus=2)

SEQLENS = [3072, 1024]
TOLERANCE = 2e-2
GRAD_WEIGHTS = ("q_a_proj.weight", "kv_a_proj_with_mqa.weight", "kv_b_proj.weight", "o_proj.weight")
DSA_CONFIG = {
    "dsa_layers": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93],
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_num_heads": 32,
}


def _config(cp: int) -> MLATransformerConfig:
    config = MLATransformerConfig(
        num_layers=4,
        hidden_size=7168,
        ffn_hidden_size=33792,
        num_attention_heads=96,
        num_query_groups=96,
        kv_channels=256,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_moe_experts=64,
        moe_ffn_hidden_size=3072,
        moe_latent_size=3584,
        context_parallel_size=cp,
    )
    configure_kimi_k3(config)
    configure_kimi_k31_dsa(config, DSA_CONFIG, topk_backend="torch")
    return config


def _setup(cp: int):
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(context_parallel_size=cp)
    model_parallel_cuda_manual_seed(1234)


def _zigzag(full: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    parts = []
    for segment in full.split(SEQLENS):
        chunks = segment.chunk(2 * size)
        parts += [chunks[rank], chunks[2 * size - 1 - rank]]
    return torch.cat(parts)


def _rel_err(ref: torch.Tensor, out: torch.Tensor) -> float:
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


def _run(layer, x, grad, params):
    x = x.unsqueeze(1).contiguous().requires_grad_()
    out, _ = layer(x, packed_seq_params=params)
    out.backward(grad.unsqueeze(1))
    named = dict(layer.named_parameters())
    return out[:, 0].detach(), x.grad[:, 0], {name: named[name].grad.clone() for name in GRAD_WEIGHTS}


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    world = dist.get_world_size()
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQLENS).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max(SEQLENS),
        max_seqlen_kv=max(SEQLENS),
    )
    torch.manual_seed(0)
    x = torch.randn(sum(SEQLENS), 7168, device="cuda", dtype=torch.bfloat16) * 0.5
    grad = torch.randn_like(x)

    _setup(cp=1)
    reference = KimiK31DSAAttention(_config(cp=1), layer_number=4).cuda()
    state = {name: tensor.clone() for name, tensor in reference.state_dict().items()}
    native_topk_fn, recorded = reference.topk_fn, []
    reference.topk_fn = lambda logits, topk: recorded.append(native_topk_fn(logits, topk)) or recorded[-1]
    out_ref, dx_ref, grads_ref = _run(reference, x, grad, params)

    _setup(cp=world)
    rank, cp_group = parallel_state.get_context_parallel_rank(), parallel_state.get_context_parallel_group()
    errors = {}
    for case, topk_fn in (("replayed", lambda logits, topk: _zigzag(recorded[0], rank, world)), ("native", None)):
        layer = KimiK31DSAAttention(_config(cp=world), layer_number=4, cp_comm_type="p2p").cuda()
        layer.load_state_dict(state)
        if topk_fn is not None:
            layer.topk_fn = topk_fn
        out, dx, grads = _run(layer, _zigzag(x, rank, world), _zigzag(grad, rank, world), params)
        errors[f"{case}_out"] = _rel_err(_zigzag(out_ref, rank, world), out)
        errors[f"{case}_dx"] = _rel_err(_zigzag(dx_ref, rank, world), dx)
        for name, g in grads.items():
            dist.all_reduce(g, group=cp_group)
            errors[f"{case}_d[{name}]"] = _rel_err(grads_ref[name], g)
    parallel_state.destroy_model_parallel()
    if rank == 0:
        print("  ".join(f"{k}={v:.2e}" for k, v in errors.items()))
    bad = {k: v for k, v in errors.items() if v > TOLERANCE}
    dist.destroy_process_group()
    if bad:
        print(f"FAILED: {bad}")
        sys.exit(1)
    if rank == 0:
        print(f"Kimi-K3.1 DSA at CP {world} matches CP 1")


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=2", __file__])
    main()
