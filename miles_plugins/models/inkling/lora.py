from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InklingLoRAAdapter(nn.Module):
    """One module's LoRA params keyed for HF export; invisible to Megatron dist-checkpointing."""

    def __init__(self, kind: str, hf_prefix: str) -> None:
        super().__init__()
        self.kind = kind
        self.hf_prefix = hf_prefix
        self.load_meta: dict[str, int] = {}

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        del prefix, sharded_offsets, metadata
        return {}


def _rmsnorm(inputs: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """Recompute the RMSNorm fused into TELayerNormColumnParallelLinear (eager, fp32 internals)."""
    x = inputs.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * gamma.float()).to(inputs.dtype)


def _new_param(
    ref_weight: torch.Tensor,
    shape: tuple[int, ...],
    *,
    init: str,
    grad_sum_group: str | None = None,
    expert: bool = False,
) -> nn.Parameter:
    tensor = torch.empty(*shape, dtype=ref_weight.dtype, device=ref_weight.device)
    if init == "zero":
        tensor.zero_()
    elif tensor.ndim == 2:
        nn.init.xavier_uniform_(tensor)
    else:
        for expert_tensor in tensor:
            nn.init.xavier_uniform_(expert_tensor)
    param = nn.Parameter(tensor)
    param.tensor_model_parallel = False
    param.partition_dim = -1
    param.partition_stride = 1
    if expert:
        param.allreduce = False
    if grad_sum_group is not None:
        # consumed by reduce_marked_lora_grads: replicated adapter grads need an
        # explicit sum over the tp/ep domain Megatron does not reduce for them
        param._lora_grad_sum_group = grad_sum_group
    return param


def _register_param(
    adapter: InklingLoRAAdapter,
    name: str,
    ref_weight: torch.Tensor,
    shape: tuple[int, ...],
    *,
    init: str = "zero",
    grad_sum_group: str | None = None,
    expert: bool = False,
) -> None:
    adapter.register_parameter(
        name, _new_param(ref_weight, shape, init=init, grad_sum_group=grad_sum_group, expert=expert)
    )


def _dropout(inputs: torch.Tensor, probability: float, training: bool) -> torch.Tensor:
    if probability and training:
        return F.dropout(inputs, p=probability, training=True)
    return inputs


def _grouped_linear(inputs: torch.Tensor, weights: torch.Tensor, tokens_per_expert) -> torch.Tensor:
    """Per-local-expert matmul over the permuted token buffer, one grouped GEMM."""
    if inputs.is_cuda:
        offsets = torch.as_tensor(list(tokens_per_expert), device=inputs.device, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        # F.grouped_mm is the public alias, added in torch 2.10; the ROCm image still
        # ships an older build that only exposes the private op. Same ATen kernel.
        grouped_mm = getattr(F, "grouped_mm", None) or torch._grouped_mm
        return grouped_mm(inputs, weights.transpose(1, 2), offs=offsets)
    segments = torch.split(inputs, list(tokens_per_expert), dim=0)
    return torch.cat([F.linear(segment, weights[idx]) for idx, segment in enumerate(segments)], dim=0)


def _gather_sequence_parallel(inputs: torch.Tensor, sequence_parallel: bool) -> torch.Tensor:
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    return gather_from_sequence_parallel_region(inputs) if sequence_parallel else inputs


def _reduce_row_parallel(partial: torch.Tensor, sequence_parallel: bool) -> torch.Tensor:
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.mappings import (
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
    )

    if parallel_state.get_tensor_model_parallel_world_size() <= 1:
        return partial
    if sequence_parallel:
        return reduce_scatter_to_sequence_parallel_region(partial)
    return reduce_from_tensor_model_parallel_region(partial)


def _apply_attention_lora(attention, args, hf_prefix: str, *, scale: float, dropout: float, a_init: str) -> None:
    from megatron.core import parallel_state

    config = attention.config
    rank = int(args.lora_rank)
    hidden_size = config.hidden_size
    eps = config.layernorm_epsilon
    sequence_parallel = bool(config.sequence_parallel)

    adapter = InklingLoRAAdapter("attn", hf_prefix + "attn.")
    qkv_ref = attention.linear_qkv.weight
    for name, out_rows in (
        ("wq", attention.nh_l * attention.hd),
        ("wk", attention.nkv_l * attention.hd),
        ("wv", attention.nkv_l * attention.hd),
        ("wr", attention.nh_l * attention.d_rel),
    ):
        _register_param(adapter, f"{name}_A", qkv_ref, (rank, hidden_size), init=a_init, grad_sum_group="tp")
        _register_param(adapter, f"{name}_B", qkv_ref, (out_rows, rank))
    proj_ref = attention.linear_proj.weight
    _register_param(adapter, "wo_A", proj_ref, (rank, attention.nh_l * attention.hd), init=a_init)
    _register_param(adapter, "wo_B", proj_ref, (hidden_size, rank), grad_sum_group="tp" if sequence_parallel else None)
    adapter.load_meta = dict(
        nh_l=attention.nh_l,
        nkv_l=attention.nkv_l,
        hd=attention.hd,
        d_rel=attention.d_rel,
        tp_rank=parallel_state.get_tensor_model_parallel_rank(),
    )
    attention.lora_adapter = adapter

    qkv = attention.linear_qkv
    original_qkv = qkv.forward

    def qkv_forward(inputs, *forward_args, **forward_kwargs):
        output, bias = original_qkv(inputs, *forward_args, **forward_kwargs)
        normed = _rmsnorm(inputs, qkv.layer_norm_weight, eps)
        normed = _dropout(_gather_sequence_parallel(normed, sequence_parallel), dropout, qkv.training)
        joint = F.linear(normed, torch.cat([adapter.wq_A, adapter.wk_A, adapter.wv_A, adapter.wr_A], dim=0))
        delta = torch.cat(
            [
                F.linear(joint[..., 0 * rank : 1 * rank], adapter.wq_B),
                F.linear(joint[..., 1 * rank : 2 * rank], adapter.wk_B),
                F.linear(joint[..., 2 * rank : 3 * rank], adapter.wv_B),
                F.linear(joint[..., 3 * rank : 4 * rank], adapter.wr_B),
            ],
            dim=-1,
        )
        return torch.add(output, delta, alpha=scale), bias

    qkv.forward = qkv_forward

    proj = attention.linear_proj
    original_proj = proj.forward

    def proj_forward(inputs, *forward_args, **forward_kwargs):
        output, bias = original_proj(inputs, *forward_args, **forward_kwargs)
        local = F.linear(_dropout(inputs, dropout, proj.training), adapter.wo_A)
        delta = F.linear(_reduce_row_parallel(local, sequence_parallel), adapter.wo_B)
        return torch.add(output, delta, alpha=scale), bias

    proj.forward = proj_forward


def _apply_dense_mlp_lora(mlp, args, hf_prefix: str, *, scale: float, dropout: float, a_init: str) -> None:
    from megatron.core import parallel_state

    config = mlp.config
    rank = int(args.lora_rank)
    hidden_size = config.hidden_size
    eps = config.layernorm_epsilon
    sequence_parallel = bool(config.sequence_parallel)
    dense_intermediate = config.ffn_hidden_size
    local_intermediate = dense_intermediate // parallel_state.get_tensor_model_parallel_world_size()

    adapter = InklingLoRAAdapter("dense_mlp", hf_prefix + "mlp.")
    fc1_ref, fc2_ref = mlp.linear_fc1.weight, mlp.linear_fc2.weight
    _register_param(adapter, "fc1_A", fc1_ref, (rank, hidden_size), init=a_init, grad_sum_group="tp")
    _register_param(adapter, "fc1_B", fc1_ref, (2 * local_intermediate, rank))
    _register_param(adapter, "fc2_A", fc2_ref, (rank, local_intermediate), init=a_init)
    _register_param(adapter, "fc2_B", fc2_ref, (hidden_size, rank), grad_sum_group="tp" if sequence_parallel else None)
    adapter.load_meta = dict(
        dense_i=dense_intermediate,
        i_loc=local_intermediate,
        tp_rank=parallel_state.get_tensor_model_parallel_rank(),
    )
    mlp.lora_adapter = adapter

    fc1 = mlp.linear_fc1
    original_fc1 = fc1.forward

    def fc1_forward(inputs, *forward_args, **forward_kwargs):
        output, bias = original_fc1(inputs, *forward_args, **forward_kwargs)
        normed = _rmsnorm(inputs, fc1.layer_norm_weight, eps)
        normed = _dropout(_gather_sequence_parallel(normed, sequence_parallel), dropout, fc1.training)
        delta = F.linear(F.linear(normed, adapter.fc1_A), adapter.fc1_B)
        return torch.add(output, delta, alpha=scale), bias

    fc1.forward = fc1_forward

    fc2 = mlp.linear_fc2
    original_fc2 = fc2.forward

    def fc2_forward(inputs, *forward_args, **forward_kwargs):
        output, bias = original_fc2(inputs, *forward_args, **forward_kwargs)
        local = F.linear(_dropout(inputs, dropout, fc2.training), adapter.fc2_A)
        delta = F.linear(_reduce_row_parallel(local, sequence_parallel), adapter.fc2_B)
        return torch.add(output, delta, alpha=scale), bias

    fc2.forward = fc2_forward


def _apply_expert_lora(moe, args, hf_prefix: str, *, scale: float, dropout: float, a_init: str) -> None:
    from megatron.core import parallel_state

    config = moe.config
    assert (getattr(config, "expert_tensor_parallel_size", 1) or 1) == 1, "Inkling LoRA assumes ETP=1"
    rank = int(args.lora_rank)
    hidden_size = config.hidden_size
    moe_intermediate = config.moe_ffn_hidden_size
    experts = moe.experts
    num_local_experts = experts.num_local_experts
    is_ep = parallel_state.get_expert_model_parallel_world_size() > 1
    ep_group = "ep" if is_ep else None

    adapter = InklingLoRAAdapter("experts", hf_prefix + "mlp.experts.")
    fc1_ref, fc2_ref = experts.linear_fc1.weight0, experts.linear_fc2.weight0
    _register_param(adapter, "w1_A", fc1_ref, (rank, hidden_size), init=a_init, grad_sum_group=ep_group, expert=is_ep)
    _register_param(adapter, "w3_A", fc1_ref, (rank, hidden_size), init=a_init, grad_sum_group=ep_group, expert=is_ep)
    _register_param(adapter, "w1_B", fc1_ref, (num_local_experts, moe_intermediate, rank), expert=is_ep)
    _register_param(adapter, "w3_B", fc1_ref, (num_local_experts, moe_intermediate, rank), expert=is_ep)
    _register_param(adapter, "w2_A", fc2_ref, (num_local_experts, rank, moe_intermediate), init=a_init, expert=is_ep)
    _register_param(adapter, "w2_B", fc2_ref, (hidden_size, rank), grad_sum_group=ep_group, expert=is_ep)
    adapter.load_meta = dict(
        e_local=num_local_experts,
        moe_i=moe_intermediate,
        ep_rank=parallel_state.get_expert_model_parallel_rank(),
    )
    experts.lora_adapter = adapter

    fc1 = experts.linear_fc1
    original_fc1 = fc1.forward

    def expert_fc1_forward(inputs, tokens_per_expert, *forward_args, **forward_kwargs):
        output, bias = original_fc1(inputs, tokens_per_expert, *forward_args, **forward_kwargs)
        dropped = _dropout(inputs, dropout, fc1.training)
        joint = F.linear(dropped, torch.cat([adapter.w1_A, adapter.w3_A], dim=0))
        gate = _grouped_linear(joint[..., :rank].contiguous(), adapter.w1_B, tokens_per_expert)
        up = _grouped_linear(joint[..., rank:].contiguous(), adapter.w3_B, tokens_per_expert)
        delta = torch.cat([gate, up], dim=-1)
        return torch.add(output, delta, alpha=scale), bias

    fc1.forward = expert_fc1_forward

    fc2 = experts.linear_fc2
    original_fc2 = fc2.forward

    def expert_fc2_forward(inputs, tokens_per_expert, *forward_args, **forward_kwargs):
        output, bias = original_fc2(inputs, tokens_per_expert, *forward_args, **forward_kwargs)
        inner = _grouped_linear(_dropout(inputs, dropout, fc2.training), adapter.w2_A, tokens_per_expert)
        delta = F.linear(inner, adapter.w2_B)
        return torch.add(output, delta, alpha=scale), bias

    fc2.forward = expert_fc2_forward


def _apply_shared_experts_lora(shared, args, hf_prefix: str, *, scale: float, dropout: float, a_init: str) -> None:
    from megatron.core import parallel_state

    config = shared.config
    rank = int(args.lora_rank)
    hidden_size = config.hidden_size
    sequence_parallel = bool(config.sequence_parallel)
    moe_intermediate = config.moe_ffn_hidden_size
    local_intermediate = moe_intermediate // parallel_state.get_tensor_model_parallel_world_size()
    num_shared = len(shared.experts)

    adapter = InklingLoRAAdapter("shared_experts", hf_prefix + "mlp.shared_experts.")
    fc1_ref = shared.experts[0].linear_fc1.weight
    fc2_ref = shared.experts[0].linear_fc2.weight
    _register_param(adapter, "w1_A", fc1_ref, (rank, hidden_size), init=a_init, grad_sum_group="tp")
    _register_param(adapter, "w3_A", fc1_ref, (rank, hidden_size), init=a_init, grad_sum_group="tp")
    _register_param(adapter, "w1_B", fc1_ref, (num_shared, local_intermediate, rank))
    _register_param(adapter, "w3_B", fc1_ref, (num_shared, local_intermediate, rank))
    _register_param(adapter, "w2_A", fc2_ref, (num_shared, rank, local_intermediate), init=a_init)
    _register_param(adapter, "w2_B", fc2_ref, (hidden_size, rank), grad_sum_group="tp" if sequence_parallel else None)
    adapter.load_meta = dict(
        ns=num_shared,
        moe_i=moe_intermediate,
        si_loc=local_intermediate,
        tp_rank=parallel_state.get_tensor_model_parallel_rank(),
    )
    shared.lora_adapter = adapter

    def patch_sub_expert(sub, idx: int) -> None:
        fc1 = sub.linear_fc1
        original_fc1 = fc1.forward

        def shared_fc1_forward(inputs, *forward_args, **forward_kwargs):
            output, bias = original_fc1(inputs, *forward_args, **forward_kwargs)
            dropped = _dropout(_gather_sequence_parallel(inputs, sequence_parallel), dropout, fc1.training)
            gate = F.linear(F.linear(dropped, adapter.w1_A), adapter.w1_B[idx])
            up = F.linear(F.linear(dropped, adapter.w3_A), adapter.w3_B[idx])
            delta = torch.cat([gate, up], dim=-1)
            return torch.add(output, delta, alpha=scale), bias

        fc1.forward = shared_fc1_forward

        fc2 = sub.linear_fc2
        original_fc2 = fc2.forward

        def shared_fc2_forward(inputs, *forward_args, **forward_kwargs):
            output, bias = original_fc2(inputs, *forward_args, **forward_kwargs)
            local = F.linear(_dropout(inputs, dropout, fc2.training), adapter.w2_A[idx])
            delta = F.linear(_reduce_row_parallel(local, sequence_parallel), adapter.w2_B)
            return torch.add(output, delta, alpha=scale), bias

        fc2.forward = shared_fc2_forward

    for idx, sub in enumerate(shared.experts):
        patch_sub_expert(sub, idx)


def _apply_lm_head_lora(model, args, *, scale: float, dropout: float, a_init: str) -> None:
    from megatron.core import parallel_state

    if not getattr(model, "post_process", False) or getattr(model, "output_layer", None) is None:
        return

    config = model.config
    rank = int(args.lora_rank)
    hidden_size = config.hidden_size
    sequence_parallel = bool(config.sequence_parallel)
    output_layer = model.output_layer
    vocab_local = output_layer.weight.shape[0]
    mup = getattr(config.inkling, "logits_mup_width_multiplier", None)
    mup = float(mup) if mup else None

    adapter = InklingLoRAAdapter("lm_head", "language_model.lm_head.")
    _register_param(adapter, "head_A", output_layer.weight, (rank, hidden_size), init=a_init, grad_sum_group="tp")
    _register_param(adapter, "head_B", output_layer.weight, (vocab_local, rank))
    adapter.load_meta = dict(vocab_local=vocab_local, tp_rank=parallel_state.get_tensor_model_parallel_rank())
    model.lora_lm_head_adapter = adapter

    original_forward = output_layer.forward

    def lm_head_forward(inputs, *forward_args, **forward_kwargs):
        output, bias = original_forward(inputs, *forward_args, **forward_kwargs)
        scaled = inputs / mup if mup else inputs
        scaled = _dropout(_gather_sequence_parallel(scaled, sequence_parallel), dropout, output_layer.training)
        delta = F.linear(F.linear(scaled, adapter.head_A), adapter.head_B)
        return torch.add(output, delta, alpha=scale), bias

    output_layer.forward = lm_head_forward


def apply_inkling_lora(model, args):
    """Attach Inkling LoRA to ONE built model chunk (before Float16Module / DDP wrapping)."""
    from miles.backends.megatron_utils.lora_utils import patch_param_grad_buffer_for_colocate_mode_lora

    from miles_plugins.models.inkling.layers import InklingDenseMLP, InklingSelfAttention, InklingSharedExperts

    if args.offload_train:
        # keep adapter param/grad buffers out of the pausable memory-saver region
        patch_param_grad_buffer_for_colocate_mode_lora()

    rank = int(args.lora_rank)
    assert rank > 0, "apply_inkling_lora requires --lora-rank > 0"
    scale = float(args.lora_alpha) / float(rank)
    dropout = float(getattr(args, "lora_dropout", 0.0) or 0.0)
    a_init = getattr(args, "lora_A_init_method", "xavier") or "xavier"
    lora_kwargs = dict(scale=scale, dropout=dropout, a_init=a_init)

    for param in model.parameters():
        param.requires_grad = False

    for layer in model.decoder.layers:
        layer_idx = layer.layer_number - 1
        hf_prefix = f"language_model.layers.{layer_idx}."

        attention = layer.self_attention
        assert isinstance(
            attention, InklingSelfAttention
        ), f"layer {layer_idx}: unexpected attention {type(attention)}"
        _apply_attention_lora(attention, args, hf_prefix, **lora_kwargs)

        mlp = layer.mlp
        if isinstance(mlp, InklingDenseMLP):
            _apply_dense_mlp_lora(mlp, args, hf_prefix, **lora_kwargs)
        else:
            _apply_expert_lora(mlp, args, hf_prefix, **lora_kwargs)
            if mlp.shared_experts is not None:
                assert isinstance(mlp.shared_experts, InklingSharedExperts)
                _apply_shared_experts_lora(mlp.shared_experts, args, hf_prefix, **lora_kwargs)

    _apply_lm_head_lora(model, args, **lora_kwargs)

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    logger.info(
        "[inkling-lora] applied: rank=%d alpha=%s scale=%.3f dropout=%s | trainable %s / %s (%.4f%%)",
        rank,
        args.lora_alpha,
        scale,
        dropout,
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(total, 1),
    )
    return model


def wrap_model_provider_with_inkling_lora(provider_func, args):
    """Wrap a miles model provider so every built chunk gets LoRA before DDP wrap."""

    def wrapped(*provider_args, **provider_kwargs):
        return apply_inkling_lora(provider_func(*provider_args, **provider_kwargs), args)

    return wrapped


def _iter_adapters(model_chunks):
    for chunk in model_chunks:
        module = chunk
        while hasattr(module, "module"):
            module = module.module
        yield from (m for m in module.modules() if isinstance(m, InklingLoRAAdapter))


def _load_attention_adapter(adapter, get_tensor, copy_param) -> None:
    meta = adapter.load_meta
    prefix = adapter.hf_prefix
    tp_rank = meta["tp_rank"]
    for hf_proj, a_name, b_name, rows in (
        ("wq_du", "wq_A", "wq_B", meta["nh_l"] * meta["hd"]),
        ("wk_dv", "wk_A", "wk_B", meta["nkv_l"] * meta["hd"]),
        ("wv_dv", "wv_A", "wv_B", meta["nkv_l"] * meta["hd"]),
        ("wr_du", "wr_A", "wr_B", meta["nh_l"] * meta["d_rel"]),
    ):
        copy_param(getattr(adapter, a_name), get_tensor(f"{prefix}{hf_proj}.lora_A.weight"))
        full_b = get_tensor(f"{prefix}{hf_proj}.lora_B.weight")
        copy_param(getattr(adapter, b_name), full_b[tp_rank * rows : (tp_rank + 1) * rows])
    full_a = get_tensor(f"{prefix}wo_ud.lora_A.weight")
    cols = meta["nh_l"] * meta["hd"]
    copy_param(adapter.wo_A, full_a[:, tp_rank * cols : (tp_rank + 1) * cols])
    copy_param(adapter.wo_B, get_tensor(f"{prefix}wo_ud.lora_B.weight"))


def _load_dense_mlp_adapter(adapter, get_tensor, copy_param) -> None:
    meta = adapter.load_meta
    prefix = adapter.hf_prefix
    tp_rank, dense_i, i_loc = meta["tp_rank"], meta["dense_i"], meta["i_loc"]
    copy_param(adapter.fc1_A, get_tensor(f"{prefix}gate_up_proj.lora_A.weight"))
    full_b = get_tensor(f"{prefix}gate_up_proj.lora_B.weight")
    gate = full_b[:dense_i][tp_rank * i_loc : (tp_rank + 1) * i_loc]
    up = full_b[dense_i:][tp_rank * i_loc : (tp_rank + 1) * i_loc]
    copy_param(adapter.fc1_B, torch.cat([gate, up], dim=0))
    full_a = get_tensor(f"{prefix}down_proj.lora_A.weight")
    copy_param(adapter.fc2_A, full_a[:, tp_rank * i_loc : (tp_rank + 1) * i_loc])
    copy_param(adapter.fc2_B, get_tensor(f"{prefix}down_proj.lora_B.weight"))


def _load_experts_adapter(adapter, get_tensor, copy_param) -> None:
    meta = adapter.load_meta
    prefix = adapter.hf_prefix
    lo = meta["ep_rank"] * meta["e_local"]
    hi = lo + meta["e_local"]
    copy_param(adapter.w1_A, get_tensor(f"{prefix}w1.lora_A.weight").squeeze(0))
    copy_param(adapter.w3_A, get_tensor(f"{prefix}w3.lora_A.weight").squeeze(0))
    copy_param(adapter.w1_B, get_tensor(f"{prefix}w1.lora_B.weight")[lo:hi])
    copy_param(adapter.w3_B, get_tensor(f"{prefix}w3.lora_B.weight")[lo:hi])
    copy_param(adapter.w2_A, get_tensor(f"{prefix}w2.lora_A.weight")[lo:hi])
    copy_param(adapter.w2_B, get_tensor(f"{prefix}w2.lora_B.weight").squeeze(0))


def _load_shared_experts_adapter(adapter, get_tensor, copy_param) -> None:
    meta = adapter.load_meta
    prefix = adapter.hf_prefix
    tp_rank, num_shared, moe_i, si_loc = meta["tp_rank"], meta["ns"], meta["moe_i"], meta["si_loc"]

    def local_rows(full: torch.Tensor, idx: int) -> torch.Tensor:
        return full[idx * moe_i + tp_rank * si_loc : idx * moe_i + (tp_rank + 1) * si_loc]

    copy_param(adapter.w1_A, get_tensor(f"{prefix}w1.lora_A.weight"))
    copy_param(adapter.w3_A, get_tensor(f"{prefix}w3.lora_A.weight"))
    full_b1 = get_tensor(f"{prefix}w1.lora_B.weight")
    full_b3 = get_tensor(f"{prefix}w3.lora_B.weight")
    copy_param(adapter.w1_B, torch.stack([local_rows(full_b1, idx) for idx in range(num_shared)]))
    copy_param(adapter.w3_B, torch.stack([local_rows(full_b3, idx) for idx in range(num_shared)]))
    full_a2 = get_tensor(f"{prefix}w2.lora_A.weight")
    copy_param(
        adapter.w2_A,
        torch.stack(
            [
                full_a2[:, idx * moe_i + tp_rank * si_loc : idx * moe_i + (tp_rank + 1) * si_loc]
                for idx in range(num_shared)
            ]
        ),
    )
    copy_param(adapter.w2_B, get_tensor(f"{prefix}w2.lora_B.weight"))


def _load_lm_head_adapter(adapter, get_tensor, copy_param) -> None:
    meta = adapter.load_meta
    prefix = adapter.hf_prefix
    tp_rank, vocab_local = meta["tp_rank"], meta["vocab_local"]
    copy_param(adapter.head_A, get_tensor(f"{prefix}lora_A.weight"))
    copy_param(
        adapter.head_B, get_tensor(f"{prefix}lora_B.weight")[tp_rank * vocab_local : (tp_rank + 1) * vocab_local]
    )


_ADAPTER_LOADERS = {
    "attn": _load_attention_adapter,
    "dense_mlp": _load_dense_mlp_adapter,
    "experts": _load_experts_adapter,
    "shared_experts": _load_shared_experts_adapter,
    "lm_head": _load_lm_head_adapter,
}


def load_inkling_lora_adapter(model_chunks, adapter_path):
    """Load the Inkling HF-format LoRA release into the applied lora params (call AFTER load_checkpoint)."""
    from safetensors import safe_open

    path = f"{adapter_path}/adapter_model.safetensors"
    n_loaded = 0
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())

        def get_tensor(name: str) -> torch.Tensor:
            assert name in keys, f"[inkling-lora] adapter tensor missing: {name}"
            return f.get_tensor(name)

        def copy_param(param: torch.Tensor, tensor: torch.Tensor) -> None:
            nonlocal n_loaded
            assert (
                param.shape == tensor.shape
            ), f"[inkling-lora] shape mismatch: param {tuple(param.shape)} vs adapter slice {tuple(tensor.shape)}"
            with torch.no_grad():
                param.copy_(tensor.to(dtype=param.dtype, device=param.device))
            n_loaded += 1

        for adapter in _iter_adapters(model_chunks):
            _ADAPTER_LOADERS[adapter.kind](adapter, get_tensor, copy_param)
    logger.info("[inkling-lora] loaded %d lora tensors from %s", n_loaded, adapter_path)
    return n_loaded


class _GatherBatch:
    """Coalesce the export's per-tensor all_gathers into ONE flat all_gather per (tp|ep) group."""

    class _Token:
        __slots__ = ("batch", "kind", "index")

        def __init__(self, batch: _GatherBatch, kind: str, index: int) -> None:
            self.batch = batch
            self.kind = kind
            self.index = index

        def get(self) -> torch.Tensor:
            return self.batch.resolved[self.kind][self.index]

    def __init__(self) -> None:
        self.requests: dict[str, list[tuple[torch.Tensor, int]]] = {"tp": [], "ep": []}
        self.resolved: dict[str, list[torch.Tensor]] = {"tp": [], "ep": []}

    def add(self, kind: str, local: torch.Tensor, dim: int) -> _Token:
        self.requests[kind].append((local, dim))
        return self._Token(self, kind, len(self.requests[kind]) - 1)

    def num_requests(self) -> int:
        return sum(len(requests) for requests in self.requests.values())

    def flush(self) -> int:
        from megatron.core import parallel_state

        groups = {
            "tp": (
                parallel_state.get_tensor_model_parallel_group,
                parallel_state.get_tensor_model_parallel_world_size,
            ),
            "ep": (
                parallel_state.get_expert_model_parallel_group,
                parallel_state.get_expert_model_parallel_world_size,
            ),
        }
        n_calls = 0
        for kind, requests in self.requests.items():
            if not requests:
                continue
            get_group, get_world_size = groups[kind]
            world_size = get_world_size()
            if world_size == 1:
                self.resolved[kind] = [local for local, _dim in requests]
                continue
            assert len({local.dtype for local, _dim in requests}) == 1, "mixed adapter dtypes"
            flat_parts = [local.detach().contiguous().reshape(-1) for local, _dim in requests]
            sizes = [part.numel() for part in flat_parts]
            flat_local = torch.cat(flat_parts)
            gathered = flat_local.new_empty(world_size * flat_local.numel())
            torch.distributed.all_gather_into_tensor(gathered, flat_local, group=get_group())
            per_rank = gathered.view(world_size, flat_local.numel())
            offset = 0
            resolved = []
            for (local, dim), size in zip(requests, sizes, strict=True):
                partitions = [per_rank[rank, offset : offset + size].view(local.shape) for rank in range(world_size)]
                resolved.append(torch.cat(partitions, dim=dim))
                offset += size
            self.resolved[kind] = resolved
            n_calls += 1
        return n_calls


_UNPADDED_VOCAB_CACHE: list = []


def _hf_unpadded_vocab_size():
    """True (unpadded) vocab size from the HF config, or None if absent."""
    if not _UNPADDED_VOCAB_CACHE:
        value = None
        try:
            from megatron.training import get_args

            with open(os.path.join(get_args().hf_checkpoint, "config.json"), encoding="utf-8") as f:
                config = json.load(f)
            value = (config.get("text_config") or config).get("unpadded_vocab_size")
        except Exception:
            value = None
        _UNPADDED_VOCAB_CACHE.append(value)
    return _UNPADDED_VOCAB_CACHE[0]


_ExportPlan = list[tuple[str, torch.Tensor | Callable[[], torch.Tensor]]]


def _export_attention(adapter: InklingLoRAAdapter, batch: _GatherBatch) -> _ExportPlan:
    prefix = adapter.hf_prefix
    plans: _ExportPlan = []
    for hf_proj, param_a, param_b in (
        ("wq_du", adapter.wq_A, adapter.wq_B),
        ("wk_dv", adapter.wk_A, adapter.wk_B),
        ("wv_dv", adapter.wv_A, adapter.wv_B),
        ("wr_du", adapter.wr_A, adapter.wr_B),
    ):
        plans.append((f"{prefix}{hf_proj}.lora_A.weight", param_a))
        plans.append((f"{prefix}{hf_proj}.lora_B.weight", batch.add("tp", param_b, 0).get))
    plans.append((f"{prefix}wo_ud.lora_A.weight", batch.add("tp", adapter.wo_A, 1).get))
    plans.append((f"{prefix}wo_ud.lora_B.weight", adapter.wo_B))
    return plans


def _export_dense_mlp(adapter: InklingLoRAAdapter, batch: _GatherBatch) -> _ExportPlan:
    prefix = adapter.hf_prefix
    i_loc = adapter.load_meta["i_loc"]
    gate_token = batch.add("tp", adapter.fc1_B[:i_loc], 0)
    up_token = batch.add("tp", adapter.fc1_B[i_loc:], 0)
    return [
        (f"{prefix}gate_up_proj.lora_A.weight", adapter.fc1_A),
        (f"{prefix}gate_up_proj.lora_B.weight", lambda: torch.cat([gate_token.get(), up_token.get()], dim=0)),
        (f"{prefix}down_proj.lora_A.weight", batch.add("tp", adapter.fc2_A, 1).get),
        (f"{prefix}down_proj.lora_B.weight", adapter.fc2_B),
    ]


def _export_experts(adapter: InklingLoRAAdapter, batch: _GatherBatch) -> _ExportPlan:
    prefix = adapter.hf_prefix
    return [
        (f"{prefix}w1.lora_A.weight", adapter.w1_A.unsqueeze(0)),
        (f"{prefix}w3.lora_A.weight", adapter.w3_A.unsqueeze(0)),
        (f"{prefix}w1.lora_B.weight", batch.add("ep", adapter.w1_B, 0).get),
        (f"{prefix}w3.lora_B.weight", batch.add("ep", adapter.w3_B, 0).get),
        (f"{prefix}w2.lora_A.weight", batch.add("ep", adapter.w2_A, 0).get),
        (f"{prefix}w2.lora_B.weight", adapter.w2_B.unsqueeze(0)),
    ]


def _export_shared_experts(adapter: InklingLoRAAdapter, batch: _GatherBatch) -> _ExportPlan:
    prefix = adapter.hf_prefix
    num_shared = adapter.load_meta["ns"]
    b1_tokens = [batch.add("tp", adapter.w1_B[idx], 0) for idx in range(num_shared)]
    b3_tokens = [batch.add("tp", adapter.w3_B[idx], 0) for idx in range(num_shared)]
    a2_tokens = [batch.add("tp", adapter.w2_A[idx], 1) for idx in range(num_shared)]
    return [
        (f"{prefix}w1.lora_A.weight", adapter.w1_A),
        (f"{prefix}w3.lora_A.weight", adapter.w3_A),
        (f"{prefix}w1.lora_B.weight", lambda: torch.cat([token.get() for token in b1_tokens], dim=0)),
        (f"{prefix}w3.lora_B.weight", lambda: torch.cat([token.get() for token in b3_tokens], dim=0)),
        (f"{prefix}w2.lora_A.weight", lambda: torch.cat([token.get() for token in a2_tokens], dim=1)),
        (f"{prefix}w2.lora_B.weight", adapter.w2_B),
    ]


def _export_lm_head(adapter: InklingLoRAAdapter, batch: _GatherBatch) -> _ExportPlan:
    prefix = adapter.hf_prefix
    head_b_token = batch.add("tp", adapter.head_B, 0)

    def head_b() -> torch.Tensor:
        full = head_b_token.get()
        unpadded = _hf_unpadded_vocab_size()
        if unpadded and unpadded < full.shape[0]:
            full = full[:unpadded]
        return full

    return [
        (f"{prefix}lora_A.weight", adapter.head_A),
        (f"{prefix}lora_B.weight", head_b),
    ]


_ADAPTER_EXPORTERS = {
    "attn": _export_attention,
    "dense_mlp": _export_dense_mlp,
    "experts": _export_experts,
    "shared_experts": _export_shared_experts,
    "lm_head": _export_lm_head,
}


def export_inkling_lora_hf_named(model_chunks):
    """Return (hf_name, full_tensor) for every applied lora param, gathered to full HF layout."""
    start = time.perf_counter()
    batch = _GatherBatch()
    plans: _ExportPlan = []
    for adapter in _iter_adapters(model_chunks):
        plans.extend(_ADAPTER_EXPORTERS[adapter.kind](adapter, batch))

    n_requests = batch.num_requests()
    n_calls = batch.flush()
    named_tensors = [
        (name, (value() if callable(value) else value).detach().to(torch.bfloat16).contiguous())
        for name, value in plans
    ]
    if torch.distributed.get_rank() == 0:
        logger.info(
            "[inkling-lora] adapter export: %d tensors, %d gathers -> %d flat all_gathers, %.1f ms",
            len(named_tensors),
            n_requests,
            n_calls,
            (time.perf_counter() - start) * 1e3,
        )
    return named_tensors
