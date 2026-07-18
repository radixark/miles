from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_SUPPORTED_TARGET_SUFFIXES = {
    "self_attention.o_proj",
    "self_attention.q_a_proj",
    "self_attention.kv_a_proj_with_mqa",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
    "mlp.experts.linear_fc1",
    "mlp.experts.linear_fc2",
}


class KimiK3LoRAAdapter(nn.Module):
    def __init__(self, kind: str, hf_prefix: str) -> None:
        super().__init__()
        self.kind = kind
        self.hf_prefix = hf_prefix
        self.load_meta: dict[str, int] = {}

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        del prefix, sharded_offsets, metadata
        return {}


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
    elif init == "xavier":
        if tensor.ndim == 2:
            nn.init.xavier_uniform_(tensor)
        else:
            for expert_tensor in tensor:
                nn.init.xavier_uniform_(expert_tensor)
    else:
        raise ValueError(f"Unsupported Kimi K3 LoRA init method: {init}")

    param = nn.Parameter(tensor)
    param.tensor_model_parallel = False
    param.partition_dim = -1
    param.partition_stride = 1
    if expert:
        param.allreduce = False
    if grad_sum_group is not None:
        param._lora_grad_sum_group = grad_sum_group
    return param


def _register_param(
    adapter: KimiK3LoRAAdapter,
    name: str,
    ref_weight: torch.Tensor,
    shape: tuple[int, ...],
    *,
    init: str,
    grad_sum_group: str | None = None,
    expert: bool = False,
) -> None:
    adapter.register_parameter(
        name,
        _new_param(
            ref_weight,
            shape,
            init=init,
            grad_sum_group=grad_sum_group,
            expert=expert,
        ),
    )


def _dropout(inputs: torch.Tensor, probability: float, training: bool) -> torch.Tensor:
    if probability and training:
        return F.dropout(inputs, p=probability, training=True)
    return inputs


def _grouped_linear(inputs: torch.Tensor, weights: torch.Tensor, tokens_per_expert: list[int]) -> torch.Tensor:
    if inputs.is_cuda:
        offsets = torch.as_tensor(tokens_per_expert, device=inputs.device, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        return F.grouped_mm(inputs, weights.transpose(1, 2), offs=offsets)
    segments = torch.split(inputs, tokens_per_expert, dim=0)
    return torch.cat([F.linear(segment, weights[idx]) for idx, segment in enumerate(segments)], dim=0)


def _validate_targets(args) -> None:
    targets = set(args.target_modules)
    suffixes = {target.split("decoder.layers.*.", 1)[-1] for target in targets}
    unsupported = suffixes - _SUPPORTED_TARGET_SUFFIXES
    missing = _SUPPORTED_TARGET_SUFFIXES - suffixes
    if unsupported or missing:
        raise NotImplementedError(
            "Kimi K3 native LoRA currently requires the verified target set; "
            f"unsupported={sorted(unsupported)}, missing={sorted(missing)}"
        )
    if not args.experts_shared_outer_loras:
        raise NotImplementedError("Kimi K3 native LoRA currently requires --experts-shared-outer-loras")


def _apply_attention_lora(attention, args, layer_idx: int, scale: float, dropout: float, a_init: str) -> None:
    from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region

    rank = int(args.lora_rank)
    hidden_size = attention.config.hidden_size
    adapter = KimiK3LoRAAdapter(
        "kda_attention" if attention.is_kda else "mla_attention",
        f"language_model.model.layers.{layer_idx}.self_attn.",
    )

    _register_param(
        adapter,
        "o_lora_A",
        attention.o_proj.weight,
        (rank, attention.o_proj.weight.shape[1]),
        init=a_init,
    )
    _register_param(
        adapter,
        "o_lora_B",
        attention.o_proj.weight,
        (hidden_size, rank),
        init="zero",
    )

    o_proj = attention.o_proj
    original_o_proj = o_proj.forward

    def o_proj_forward(inputs, *forward_args, **forward_kwargs):
        output, bias = original_o_proj(inputs, *forward_args, **forward_kwargs)
        local = F.linear(_dropout(inputs, dropout, o_proj.training), adapter.o_lora_A)
        reduced = reduce_from_tensor_model_parallel_region(local, group=attention.tp_group)
        delta = F.linear(reduced, adapter.o_lora_B)
        return torch.add(output, delta, alpha=scale), bias

    o_proj.forward = o_proj_forward

    if not attention.is_kda:
        for module_name, output_size in (
            ("q_a_proj", attention.q_lora_rank),
            ("kv_a_proj_with_mqa", attention.kv_lora_rank + attention.qk_extra_head_dim),
        ):
            module = getattr(attention, module_name)
            prefix = "q_a" if module_name == "q_a_proj" else "kv_a"
            _register_param(
                adapter,
                f"{prefix}_lora_A",
                module.weight,
                (rank, hidden_size),
                init=a_init,
            )
            _register_param(
                adapter,
                f"{prefix}_lora_B",
                module.weight,
                (output_size, rank),
                init="zero",
            )
            original_forward = module.forward
            lora_a = getattr(adapter, f"{prefix}_lora_A")
            lora_b = getattr(adapter, f"{prefix}_lora_B")

            def duplicated_forward(
                inputs,
                *forward_args,
                _module=module,
                _original=original_forward,
                _a=lora_a,
                _b=lora_b,
                **forward_kwargs,
            ):
                output, bias = _original(inputs, *forward_args, **forward_kwargs)
                delta = F.linear(F.linear(_dropout(inputs, dropout, _module.training), _a), _b)
                # q_b/kv_b TE column backward reduces latent dgrad over TP;
                # the key-extra slice is reduced by KimiK3Attention itself.
                return torch.add(output, delta, alpha=scale), bias

            module.forward = duplicated_forward

    attention.lora_adapter = adapter


def _apply_dense_mlp_lora(
    mlp,
    args,
    layer_idx: int,
    scale: float,
    dropout: float,
    a_init: str,
    *,
    adapter_kind: str = "dense_mlp",
    hf_prefix: str | None = None,
) -> None:
    from megatron.core.tensor_parallel.mappings import (
        gather_from_sequence_parallel_region,
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
    )

    rank = int(args.lora_rank)
    sequence_parallel = bool(mlp.config.sequence_parallel)
    tp_group = mlp.tp_group
    fc1 = mlp.linear_fc1
    fc2 = mlp.linear_fc2
    adapter = KimiK3LoRAAdapter(
        adapter_kind,
        hf_prefix or f"language_model.model.layers.{layer_idx}.mlp.",
    )
    _register_param(
        adapter,
        "fc1_lora_A",
        fc1.weight,
        (rank, mlp.config.hidden_size),
        init=a_init,
        grad_sum_group="tp",
    )
    _register_param(
        adapter,
        "fc1_lora_B",
        fc1.weight,
        (fc1.weight.shape[0], rank),
        init="zero",
    )
    _register_param(
        adapter,
        "fc2_lora_A",
        fc2.weight,
        (rank, fc2.weight.shape[1]),
        init=a_init,
    )
    _register_param(
        adapter,
        "fc2_lora_B",
        fc2.weight,
        (mlp.config.hidden_size, rank),
        init="zero",
        grad_sum_group="tp" if sequence_parallel else None,
    )

    original_fc1 = fc1.forward

    def fc1_forward(inputs, *forward_args, **forward_kwargs):
        output, bias = original_fc1(inputs, *forward_args, **forward_kwargs)
        adapter_inputs = gather_from_sequence_parallel_region(inputs, group=tp_group) if sequence_parallel else inputs
        delta = F.linear(
            F.linear(_dropout(adapter_inputs, dropout, fc1.training), adapter.fc1_lora_A),
            adapter.fc1_lora_B,
        )
        return torch.add(output, delta, alpha=scale), bias

    fc1.forward = fc1_forward
    original_fc2 = fc2.forward

    def fc2_forward(inputs, *forward_args, **forward_kwargs):
        output, bias = original_fc2(inputs, *forward_args, **forward_kwargs)
        local = F.linear(_dropout(inputs, dropout, fc2.training), adapter.fc2_lora_A)
        reduced = (
            reduce_scatter_to_sequence_parallel_region(local, group=tp_group)
            if sequence_parallel
            else reduce_from_tensor_model_parallel_region(local, group=tp_group)
        )
        delta = F.linear(reduced, adapter.fc2_lora_B)
        return torch.add(output, delta, alpha=scale), bias

    fc2.forward = fc2_forward
    mlp.lora_adapter = adapter


def _apply_expert_lora(moe, args, layer_idx: int, scale: float, dropout: float, a_init: str) -> None:
    experts = moe.experts
    if (moe.config.expert_tensor_parallel_size or 1) != 1:
        raise NotImplementedError("Kimi K3 native expert LoRA currently requires ETP=1")

    rank = int(args.lora_rank)
    num_local_experts = experts.num_local_experts
    latent_size = moe.config.moe_latent_size
    intermediate_size = moe.config.moe_ffn_hidden_size
    ref_fc1 = experts.linear_fc1.weight0
    ref_fc2 = experts.linear_fc2.weight0
    adapter = KimiK3LoRAAdapter(
        "experts",
        f"language_model.model.layers.{layer_idx}.block_sparse_moe.experts.",
    )
    _register_param(
        adapter,
        "w1_lora_A",
        ref_fc1,
        (rank, latent_size),
        init=a_init,
        grad_sum_group="ep",
        expert=True,
    )
    _register_param(
        adapter,
        "w3_lora_A",
        ref_fc1,
        (rank, latent_size),
        init=a_init,
        grad_sum_group="ep",
        expert=True,
    )
    _register_param(
        adapter,
        "w1_lora_B",
        ref_fc1,
        (num_local_experts, intermediate_size, rank),
        init="zero",
        expert=True,
    )
    _register_param(
        adapter,
        "w3_lora_B",
        ref_fc1,
        (num_local_experts, intermediate_size, rank),
        init="zero",
        expert=True,
    )
    _register_param(
        adapter,
        "w2_lora_A",
        ref_fc2,
        (num_local_experts, rank, intermediate_size),
        init=a_init,
        expert=True,
    )
    _register_param(
        adapter,
        "w2_lora_B",
        ref_fc2,
        (latent_size, rank),
        init="zero",
        grad_sum_group="ep",
        expert=True,
    )
    adapter.load_meta = {"num_local_experts": num_local_experts}

    fc1 = experts.linear_fc1
    original_fc1 = fc1.forward

    def expert_fc1_forward(inputs, tokens_per_expert, *forward_args, **forward_kwargs):
        output, bias = original_fc1(inputs, tokens_per_expert, *forward_args, **forward_kwargs)
        adapter_inputs = _dropout(inputs, dropout, fc1.training)
        shared = F.linear(
            adapter_inputs,
            torch.cat((adapter.w1_lora_A, adapter.w3_lora_A), dim=0),
        )
        w1_shared, w3_shared = shared.chunk(2, dim=-1)
        w1_delta = _grouped_linear(w1_shared.contiguous(), adapter.w1_lora_B, tokens_per_expert)
        w3_delta = _grouped_linear(w3_shared.contiguous(), adapter.w3_lora_B, tokens_per_expert)
        delta = torch.cat((w1_delta, w3_delta), dim=-1)
        return torch.add(output, delta, alpha=scale), bias

    fc1.forward = expert_fc1_forward
    fc2 = experts.linear_fc2
    original_fc2 = fc2.forward

    def expert_fc2_forward(inputs, tokens_per_expert, *forward_args, **forward_kwargs):
        output, bias = original_fc2(inputs, tokens_per_expert, *forward_args, **forward_kwargs)
        inner = _grouped_linear(
            _dropout(inputs, dropout, fc2.training),
            adapter.w2_lora_A,
            tokens_per_expert,
        )
        delta = F.linear(inner, adapter.w2_lora_B)
        return torch.add(output, delta, alpha=scale), bias

    fc2.forward = expert_fc2_forward
    experts.lora_adapter = adapter


def apply_kimi_k3_lora(model, args):
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.moe.moe_layer import MoELayer

    from .layers import KimiK3Attention

    _validate_targets(args)
    rank = int(args.lora_rank)
    if rank <= 0:
        raise ValueError("apply_kimi_k3_lora requires --lora-rank > 0")
    scale = float(args.lora_alpha) / rank
    dropout = float(args.lora_dropout or 0.0)
    a_init = "xavier"

    for parameter in model.parameters():
        parameter.requires_grad = False

    for layer in model.decoder.layers:
        layer_idx = layer.layer_number - 1
        if not isinstance(layer.self_attention, KimiK3Attention):
            raise TypeError(f"Kimi K3 layer {layer_idx} has unexpected attention type {type(layer.self_attention)}")
        _apply_attention_lora(layer.self_attention, args, layer_idx, scale, dropout, a_init)

        if isinstance(layer.mlp, MLP):
            _apply_dense_mlp_lora(layer.mlp, args, layer_idx, scale, dropout, a_init)
        elif isinstance(layer.mlp, MoELayer):
            _apply_expert_lora(layer.mlp, args, layer_idx, scale, dropout, a_init)
            if layer.mlp.shared_experts is None:
                raise RuntimeError(f"Kimi K3 MoE layer {layer_idx} is missing shared experts")
            _apply_dense_mlp_lora(
                layer.mlp.shared_experts,
                args,
                layer_idx,
                scale,
                dropout,
                a_init,
                adapter_kind="shared_experts",
                hf_prefix=(f"language_model.model.layers.{layer_idx}.block_sparse_moe.shared_experts."),
            )
        else:
            raise TypeError(f"Kimi K3 layer {layer_idx} has unexpected MLP type {type(layer.mlp)}")

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "Kimi K3 native LoRA applied: rank=%d alpha=%s trainable=%d total=%d ratio=%.6f%%",
        rank,
        args.lora_alpha,
        trainable,
        total,
        100.0 * trainable / total,
    )
    return model


def wrap_model_provider_with_kimi_k3_lora(provider_func, args):
    def wrapped(*provider_args, **provider_kwargs):
        return apply_kimi_k3_lora(provider_func(*provider_args, **provider_kwargs), args)

    return wrapped


class _GatherBatch:
    class _Token:
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

    def flush(self) -> int:
        from megatron.core import parallel_state

        groups = {
            "tp": (
                parallel_state.get_tensor_model_parallel_group,
                parallel_state.get_tensor_model_parallel_world_size(),
            ),
            "ep": (
                parallel_state.get_expert_model_parallel_group,
                parallel_state.get_expert_model_parallel_world_size(),
            ),
        }
        calls = 0
        for kind, requests in self.requests.items():
            if not requests:
                continue
            get_group, world_size = groups[kind]
            if world_size == 1:
                self.resolved[kind] = [local for local, _dim in requests]
                continue
            group = get_group()
            dtypes = {local.dtype for local, _dim in requests}
            if len(dtypes) != 1:
                raise TypeError(f"Kimi K3 LoRA {kind} gather has mixed dtypes: {dtypes}")
            flat_parts = [local.detach().contiguous().view(-1) for local, _dim in requests]
            sizes = [part.numel() for part in flat_parts]
            flat_local = torch.cat(flat_parts)
            gathered = flat_local.new_empty(world_size * flat_local.numel())
            torch.distributed.all_gather_into_tensor(gathered, flat_local, group=group)
            per_rank = gathered.view(world_size, flat_local.numel())
            offset = 0
            resolved = []
            for (local, dim), size in zip(requests, sizes, strict=True):
                partitions = [per_rank[rank, offset : offset + size].view(local.shape) for rank in range(world_size)]
                resolved.append(torch.cat(partitions, dim=dim))
                offset += size
            self.resolved[kind] = resolved
            calls += 1
        return calls


def _unwrap_model_chunks(model_chunks):
    for chunk in model_chunks:
        while hasattr(chunk, "module"):
            chunk = chunk.module
        yield chunk


def _validate_adapter_layout(models, adapters: list[KimiK3LoRAAdapter]) -> None:
    expected = []
    for model in models:
        for layer in model.decoder.layers:
            layer_idx = layer.layer_number - 1
            attention_kind = "kda_attention" if layer.self_attention.is_kda else "mla_attention"
            expected.append(
                (
                    attention_kind,
                    f"language_model.model.layers.{layer_idx}.self_attn.",
                )
            )
            if hasattr(layer.mlp, "experts"):
                expected.extend(
                    (
                        (
                            "experts",
                            f"language_model.model.layers.{layer_idx}.block_sparse_moe.experts.",
                        ),
                        (
                            "shared_experts",
                            f"language_model.model.layers.{layer_idx}.block_sparse_moe.shared_experts.",
                        ),
                    )
                )
            else:
                expected.append(
                    (
                        "dense_mlp",
                        f"language_model.model.layers.{layer_idx}.mlp.",
                    )
                )

    expected_counts = Counter(expected)
    actual_counts = Counter((adapter.kind, adapter.hf_prefix) for adapter in adapters)
    if actual_counts != expected_counts:
        missing = list((expected_counts - actual_counts).elements())
        unexpected = list((actual_counts - expected_counts).elements())
        raise RuntimeError(
            "Kimi K3 LoRA adapter layout is incomplete: "
            f"expected={sum(expected_counts.values())}, actual={sum(actual_counts.values())}, "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def _export_attention(adapter: KimiK3LoRAAdapter, materialize_parameter):
    batch = _GatherBatch()
    plans: list[tuple[str, torch.Tensor | Callable[[], torch.Tensor]]] = []
    prefix = adapter.hf_prefix
    if adapter.kind == "mla_attention":
        for hf_name, parameter_a, parameter_b in (
            ("q_a_proj", adapter.q_a_lora_A, adapter.q_a_lora_B),
            ("kv_a_proj_with_mqa", adapter.kv_a_lora_A, adapter.kv_a_lora_B),
        ):
            local_a = materialize_parameter(parameter_a)
            local_b = materialize_parameter(parameter_b)
            plans.append((f"{prefix}{hf_name}.lora_A.weight", local_a))
            plans.append((f"{prefix}{hf_name}.lora_B.weight", local_b))
    o_a = batch.add("tp", materialize_parameter(adapter.o_lora_A), 1)
    plans.append((f"{prefix}o_proj.lora_A.weight", o_a.get))
    plans.append((f"{prefix}o_proj.lora_B.weight", materialize_parameter(adapter.o_lora_B)))
    batch.flush()
    return plans


def _export_dense_mlp(adapter: KimiK3LoRAAdapter, materialize_parameter):
    batch = _GatherBatch()
    prefix = adapter.hf_prefix
    fc1_a = materialize_parameter(adapter.fc1_lora_A)
    gate_b_local, up_b_local = materialize_parameter(adapter.fc1_lora_B).chunk(2, dim=0)
    gate_b = batch.add("tp", gate_b_local, 0)
    up_b = batch.add("tp", up_b_local, 0)
    down_a = batch.add("tp", materialize_parameter(adapter.fc2_lora_A), 1)
    fc2_b = materialize_parameter(adapter.fc2_lora_B)
    batch.flush()
    return [
        (f"{prefix}gate_proj.lora_A.weight", fc1_a),
        (f"{prefix}gate_proj.lora_B.weight", gate_b.get),
        (f"{prefix}up_proj.lora_A.weight", fc1_a),
        (f"{prefix}up_proj.lora_B.weight", up_b.get),
        (f"{prefix}down_proj.lora_A.weight", down_a.get),
        (f"{prefix}down_proj.lora_B.weight", fc2_b),
    ]


def _export_experts(adapter: KimiK3LoRAAdapter, materialize_parameter):
    batch = _GatherBatch()
    prefix = adapter.hf_prefix
    w1_a = materialize_parameter(adapter.w1_lora_A)
    w3_a = materialize_parameter(adapter.w3_lora_A)
    w1_b = batch.add("ep", materialize_parameter(adapter.w1_lora_B), 0)
    w3_b = batch.add("ep", materialize_parameter(adapter.w3_lora_B), 0)
    w2_a = batch.add("ep", materialize_parameter(adapter.w2_lora_A), 0)
    w2_b = materialize_parameter(adapter.w2_lora_B)
    batch.flush()
    return [
        (f"{prefix}w1.lora_A.weight", w1_a.unsqueeze(0)),
        (f"{prefix}w1.lora_B.weight", w1_b.get),
        (f"{prefix}w3.lora_A.weight", w3_a.unsqueeze(0)),
        (f"{prefix}w3.lora_B.weight", w3_b.get),
        (f"{prefix}w2.lora_A.weight", w2_a.get),
        (f"{prefix}w2.lora_B.weight", w2_b.unsqueeze(0)),
    ]


def export_kimi_k3_lora_hf_chunks(model_chunks, materialize_parameter=lambda parameter: parameter):
    models = list(_unwrap_model_chunks(model_chunks))
    adapters: list[KimiK3LoRAAdapter] = []
    for model in models:
        adapters.extend(module for module in model.modules() if isinstance(module, KimiK3LoRAAdapter))
    if not adapters:
        raise RuntimeError("Kimi K3 native LoRA export found no adapters")
    _validate_adapter_layout(models, adapters)

    for adapter in adapters:
        if adapter.kind in ("kda_attention", "mla_attention"):
            plans = _export_attention(adapter, materialize_parameter)
        elif adapter.kind in ("dense_mlp", "shared_experts"):
            plans = _export_dense_mlp(adapter, materialize_parameter)
        elif adapter.kind == "experts":
            plans = _export_experts(adapter, materialize_parameter)
        else:
            raise ValueError(f"Unknown Kimi K3 LoRA adapter kind: {adapter.kind}")
        yield [
            (name, (value() if callable(value) else value).detach().to(torch.bfloat16).contiguous())
            for name, value in plans
        ]
