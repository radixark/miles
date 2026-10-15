"""Measure indexer, attention, and recomputation time and memory on captured layer inputs.

Sparse and dense layers share checkpoint weights and THD inputs. Rank 0 reports
CUDA-event medians and peak-memory increments. See the README for invocation.
"""

import argparse
import json
import os
import statistics
from collections.abc import Callable

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
from safetensors import safe_open

from miles.kernels.attention.region_sparse import sparse_region_attention
from miles_plugins.models.step5.indexer import read_checkpoint_layer
from miles_plugins.models.step5.layer_math import FULL_ATTENTION_ROPE
from miles_plugins.models.step5.layers import Step5Attention, Step5RMSNorm
from miles_plugins.models.step5.qkv_layout import fuse_qkv_and_gate
from miles_plugins.models.step5.sparse_attention import Step5SparseAttention, Step5SparseCoreAttention

HIDDEN, HEADS, GROUPS, HEAD_DIM = 4096, 64, 4, 192


class DenseFullAttention(Step5Attention):
    """Step5's full-attention layer before region selection: dense causal attention, full-attention RoPE."""

    rope = FULL_ATTENTION_ROPE


def _config(tp_size: int, sequence_parallel: bool, layer_number: int) -> TransformerConfig:
    return TransformerConfig(
        num_layers=layer_number,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_query_groups=GROUPS,
        kv_channels=HEAD_DIM,
        layernorm_epsilon=1e-5,
        qk_layernorm=True,
        head_wise_attn_gate=True,
        add_bias_linear=False,
        window_size=(511, 0),
        # only this layer's type matters: full attention
        window_attn_skip_freq=[1] * (layer_number - 1) + [0],
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=sequence_parallel,
        apply_rope_fusion=True,
        gradient_accumulation_fusion=False,
    )


def _build(module, core_attention, config: TransformerConfig, layer_number: int):
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
    return build_module(spec, config=config, layer_number=layer_number).cuda()


def _read(checkpoint: str, names: list[str]) -> dict[str, torch.Tensor]:
    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    tensors = {}
    for name in names:
        with safe_open(os.path.join(checkpoint, weight_map[name]), framework="pt") as f:
            tensors[name] = f.get_tensor(name)
    return tensors


def _load_layer(modules, checkpoint: str, layer_index: int, tp_size: int, tp_rank: int) -> None:
    prefix = f"model.layers.{layer_index}.self_attn"
    names = [f"{prefix}.{p}_proj.weight" for p in ("q", "k", "v", "g", "o")]
    tensors = _read(checkpoint, names + [f"{prefix}.q_norm.weight", f"{prefix}.k_norm.weight"])
    q, k, v, g, o = (tensors[name] for name in names)
    fused = fuse_qkv_and_gate(query=q, key=k, value=v, gate=g, num_query_groups=GROUPS, tp_size=tp_size)
    for module in modules:
        with torch.no_grad():
            module.linear_qkv.weight.copy_(fused.chunk(tp_size)[tp_rank])
            module.linear_proj.weight.copy_(o.chunk(tp_size, dim=1)[tp_rank])
            module.q_layernorm.weight.copy_(tensors[f"{prefix}.q_norm.weight"])
            module.k_layernorm.weight.copy_(tensors[f"{prefix}.k_norm.weight"])


def _median_ms(fn: Callable[[], dict[str, tuple[torch.cuda.Event, torch.cuda.Event]]], iters: int) -> dict:
    """``fn`` returns named (start, end) event pairs; the median of each over ``iters`` runs after a warmup."""
    fn()
    samples: dict[str, list[float]] = {}
    for _ in range(iters):
        torch.cuda.synchronize()
        events = fn()
        torch.cuda.synchronize()
        for name, (start, end) in events.items():
            samples.setdefault(name, []).append(start.elapsed_time(end))
    return {name: round(statistics.median(times), 2) for name, times in samples.items()}


def _peak_mib(fn: Callable[[], object]) -> float:
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return round((torch.cuda.max_memory_allocated() - base) / 2**20, 1)


def _events() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--inputs", required=True, help="indexer_inputs.py output")
    parser.add_argument("--prompt", default="text0-32768")
    parser.add_argument("--copies", type=int, default=1, help="sequences in the pack, each the whole prompt")
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    tp_size = dist.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_size)
    tensor_parallel.model_parallel_cuda_manual_seed(1234)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()

    captured = torch.load(args.inputs, weights_only=True)
    layer_index = captured["layer_index"]
    config = _config(tp_size, args.sequence_parallel, layer_index + 1)
    sparse = _build(Step5SparseAttention, Step5SparseCoreAttention, config, layer_index + 1)
    dense = _build(DenseFullAttention, TEDotProductAttention, config, layer_index + 1)
    _load_layer((sparse, dense), args.hf_checkpoint, layer_index, tp_size, tp_rank)
    sparse.indexer.load_checkpoint_layer(read_checkpoint_layer(args.hf_checkpoint, layer_index))

    prompt = captured["hidden"][args.prompt].cuda()
    length = prompt.shape[0]
    hidden = prompt.repeat(args.copies, 1)[:, None]
    cu_seqlens = torch.arange(args.copies + 1, dtype=torch.int32, device="cuda") * length
    packed = PackedSeqParams(
        cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens, max_seqlen_q=length, max_seqlen_kv=length, qkv_format="thd"
    )
    x = (hidden.chunk(tp_size)[tp_rank] if args.sequence_parallel else hidden).contiguous()
    grad = torch.randn(x.shape, device="cuda", dtype=torch.bfloat16) * 1e-2
    rotary = sparse.rotary_pos_emb(length)

    def call(module, h):
        return module(h, None, rotary_pos_emb=rotary, packed_seq_params=packed)[0]

    def indexer():
        start, end = _events()
        start.record()
        sparse.indexer.select_regions(x, packed)
        end.record()
        return {"indexer": (start, end)}

    def layer(module, recompute: bool):
        def run():
            h = x.detach().requires_grad_()
            fwd, bwd = _events(), _events()
            fwd[0].record()
            out = tensor_parallel.checkpoint(lambda t: call(module, t), False, h) if recompute else call(module, h)
            fwd[1].record()
            bwd[0].record()
            out.backward(grad)
            bwd[1].record()
            return {"fwd": fwd, "bwd": bwd, "total": (fwd[0], bwd[1])}

        return run

    captured_core = {}

    def capture(module, core_args):
        captured_core["args"] = tuple(a.detach() if torch.is_tensor(a) else a for a in core_args)

    handle = sparse.core_attention.register_forward_pre_hook(capture)
    call(sparse, x)
    handle.remove()
    q, k, v, selection = captured_core["args"]

    def kernel():
        qq, kk, vv = (t.clone().requires_grad_() for t in (q, k, v))
        fwd, bwd = _events(), _events()
        fwd[0].record()
        out = sparse_region_attention(
            qq, kk, vv, selection.region_ids, selection.cu_seqlens, sparse.core_attention.softmax_scale
        )
        fwd[1].record()
        bwd[0].record()
        out.backward(torch.ones_like(out))
        bwd[1].record()
        return {"fwd": fwd, "bwd": bwd}

    report = {
        "tokens": length * args.copies,
        "tp": tp_size,
        "sequence_parallel": args.sequence_parallel,
        "indexer_ms": _median_ms(indexer, args.iters),
        "indexer_peak_mib": _peak_mib(lambda: indexer()),
        "kernel_ms": _median_ms(kernel, args.iters),
        "kernel_peak_mib": _peak_mib(kernel),
    }
    for name, module in (("sparse", sparse), ("dense", dense)):
        for recompute in (False, True):
            key = f"{name}_recompute" if recompute else name
            report[f"{key}_ms"] = _median_ms(layer(module, recompute), args.iters)
            report[f"{key}_peak_mib"] = _peak_mib(layer(module, recompute))
    if dist.get_rank() == 0:
        print(json.dumps(report), flush=True)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
