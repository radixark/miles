"""Exact HF naming, loading, and export for native LoRA."""

from __future__ import annotations

import logging
import os
import re
import time

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.lora.native.distributed import ParallelGather
from miles.backends.megatron_utils.lora.native.modules.linear import NativeLoRAAdapter, iter_adapters
from miles.backends.megatron_utils.lora.native.spec.base import ShardLayout

logger = logging.getLogger(__name__)


def export_lora_hf_named(model_chunks) -> list[tuple[str, torch.Tensor]]:
    """Materialize full HF/PEFT adapter tensors on every TP rank.

    This representation preserves the exact target set; the serving exporter adds the
    zero siblings SGLang's fused buffers need. The weight iterator gathers across PP.
    """
    started = time.perf_counter()
    gather = ParallelGather()
    plan: list[tuple[str, object]] = []

    for adapter in iter_adapters(model_chunks):
        custom = adapter.export_plan(gather)
        if custom is not None:
            plan.extend(custom)
            continue
        for export in adapter.exports():
            a: object = export.a
            b: object = export.b
            if export.layout == ShardLayout.COLUMN:
                b = gather.request(export.b, 0)
            elif export.layout == ShardLayout.ROW:
                a = gather.request(export.a, 1)
            plan.append((f"{export.hf_name}.lora_A.weight", a))
            plan.append((f"{export.hf_name}.lora_B.weight", b))

    gather.flush()
    exported = [
        (name, (source() if callable(source) else source).detach().to(torch.bfloat16).contiguous())
        for name, source in plan
    ]
    if not dist.is_initialized() or dist.get_rank() == 0:
        peak_b = max(
            (tensor.abs().max().item() for name, tensor in exported if name.endswith("lora_B.weight")),
            default=0.0,
        )
        logger.info(
            "[lora-native] exported %d adapter tensors in %.1f ms (max|lora_B|=%.3e)",
            len(exported),
            (time.perf_counter() - started) * 1e3,
            peak_b,
        )
    return exported


def load_lora_adapter_hf(model_chunks, adapter_path: str) -> int:
    """Load and slice an HF/PEFT adapter into attached native modules.

    Every full HF tensor is shape-checked before any TP slicing or parameter
    mutation. This rejects oversized global tensors whose valid-looking local
    prefix used to be silently accepted and also keeps a failed load locally
    atomic.
    """
    from safetensors import safe_open

    path = os.path.join(adapter_path, "adapter_model.safetensors")
    assert os.path.exists(path), (
        f"[lora-native] no adapter_model.safetensors under {adapter_path}; "
        "checkpoints written by save_lora_checkpoint use that name"
    )
    with safe_open(path, framework="pt") as adapter_file:
        keys = {re.sub(r"^base_model\.model\.", "", key): key for key in adapter_file.keys()}

        def take(name: str) -> torch.Tensor:
            if name not in keys:
                raise KeyError(f"[lora-native] adapter tensor missing: {name}")
            return adapter_file.get_tensor(keys[name])

        load_plan: list[tuple[torch.Tensor, torch.Tensor]] = []
        for adapter in iter_adapters(model_chunks):
            load_plan.extend(_load_adapter(adapter, take))

    with torch.no_grad():
        for parameter, tensor in load_plan:
            parameter.copy_(tensor.to(dtype=parameter.dtype, device=parameter.device))
    loaded = len(load_plan)
    logger.info("[lora-native] loaded %d adapter tensors from %s", loaded, adapter_path)
    return loaded


def _load_adapter(adapter: NativeLoRAAdapter, take) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build a fully shape-validated load plan for one native adapter."""
    custom = adapter.load_plan(take)
    if custom is not None:
        for parameter, tensor in custom:
            assert tuple(parameter.shape) == tuple(tensor.shape), (
                f"[lora-native] shape mismatch loading {adapter.hf_prefix!r}: "
                f"checkpoint slice {tuple(tensor.shape)} != parameter {tuple(parameter.shape)}"
            )
        return list(custom)
    tp_size = adapter.context.tp_size
    plan: list[tuple[torch.Tensor, torch.Tensor]] = []
    for projection in adapter.projection_specs:
        a_parameter = getattr(adapter, f"{projection.param_prefix}_A")
        b_parameter = getattr(adapter, f"{projection.param_prefix}_B")
        a_name = f"{adapter.hf_prefix}{projection.hf_leaf}.lora_A.weight"
        b_name = f"{adapter.hf_prefix}{projection.hf_leaf}.lora_B.weight"
        a_full = take(a_name)
        b_full = take(b_name)

        expected_a = tuple(a_parameter.shape)
        expected_b = tuple(b_parameter.shape)
        if projection.layout == ShardLayout.COLUMN:
            expected_b = (b_parameter.shape[0] * tp_size, b_parameter.shape[1])
        elif projection.layout == ShardLayout.ROW:
            expected_a = (a_parameter.shape[0], a_parameter.shape[1] * tp_size)
        if tuple(a_full.shape) != expected_a:
            raise ValueError(
                f"[lora-native] global shape mismatch for {a_name}: "
                f"checkpoint {tuple(a_full.shape)} != expected {expected_a}"
            )
        if tuple(b_full.shape) != expected_b:
            raise ValueError(
                f"[lora-native] global shape mismatch for {b_name}: "
                f"checkpoint {tuple(b_full.shape)} != expected {expected_b}"
            )

        if projection.layout != ShardLayout.REPLICATED:
            if projection.layout == ShardLayout.COLUMN:
                width = b_parameter.shape[0]
                span = slice(adapter.tp_rank * width, (adapter.tp_rank + 1) * width)
                b_full = b_full[span]
            else:
                width = a_parameter.shape[1]
                span = slice(adapter.tp_rank * width, (adapter.tp_rank + 1) * width)
                a_full = a_full[:, span]
        plan.append((a_parameter, a_full))
        plan.append((b_parameter, b_full))
    return plan
