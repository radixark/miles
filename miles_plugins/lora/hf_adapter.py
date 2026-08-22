"""Exact HF naming, loading, and export for native LoRA."""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import time
from collections.abc import Iterable

import torch
import torch.distributed as dist

from miles_plugins.lora.distributed import ParallelGather
from miles_plugins.lora.modules.linear import NativeLoRAAdapter, iter_adapters
from miles_plugins.lora.spec.base import ShardLayout

# Megatron -> HF module-name mapping, covering merged (LoRA), split
# (CanonicalLoRA), GDN, and MLA spellings.
MEGATRON_TO_HF_MODULES = {
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
    "linear_fc1": ["gate_proj", "up_proj"],
    "linear_fc2": ["down_proj"],
    "linear_q": ["q_proj"],
    "linear_k": ["k_proj"],
    "linear_v": ["v_proj"],
    "linear_fc1_gate": ["gate_proj"],
    "linear_fc1_up": ["up_proj"],
    "in_proj": ["in_proj_qkvz", "in_proj_ba"],
}

MEGATRON_MLA_TO_HF = {
    "linear_q_down_proj": "q_a_proj",
    "linear_kv_down_proj": "kv_a_proj_with_mqa",
    "linear_q_up_proj": "q_b_proj",
    "linear_kv_up_proj": "kv_b_proj",
    "linear_wq_b": "wq_b",
    "linear_wk": "wk",
    "linear_weights_proj": "weights_proj",
}


def convert_target_modules_to_hf(megatron_modules: list[str]) -> list[str]:
    """Convert Megatron LoRA target module names to HuggingFace format.

    Wildcards (``*.layers.2.mlp.experts.linear_fc1``) get the last dotted
    segment mapped to an HF leaf name; SGLang uses the result to choose
    adapter-buffer types, not to scope by layer.
    """
    if isinstance(megatron_modules, tuple):
        megatron_modules = list(megatron_modules)
    hf_modules: list[str] = []
    for module in megatron_modules:
        lookup_key = module.rsplit(".", 1)[-1] if "." in module else module
        if lookup_key in MEGATRON_MLA_TO_HF:
            hf_modules.append(MEGATRON_MLA_TO_HF[lookup_key])
        elif lookup_key in MEGATRON_TO_HF_MODULES:
            hf_modules.extend(MEGATRON_TO_HF_MODULES[lookup_key])
        else:
            hf_modules.append(lookup_key)
    seen: set[str] = set()
    unique: list[str] = []
    for m in hf_modules:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


logger = logging.getLogger(__name__)


def target_modules_from_hf_names(names: Iterable[str]) -> list[str]:
    """Return the exact logical projection leaves represented by HF LoRA tensors."""
    targets = set()
    for name in names:
        match = re.search(r"(?:^|\.)([^.]+)\.lora_[AB]\.weight$", name)
        if match:
            targets.add(match.group(1))
    return sorted(targets)


def resolve_hf_naming(hf_checkpoint: str | None) -> tuple[str, str]:
    """Read decoder-layer and shared-expert prefixes from the served checkpoint.

    Falls back to the common HF spelling (``model.layers.`` /
    ``mlp.shared_expert.``) when the checkpoint has no weight index.
    """
    default_layer_prefix, default_shared_expert = "model.layers.", "mlp.shared_expert."
    index_path = os.path.join(hf_checkpoint or "", "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return default_layer_prefix, default_shared_expert
    with open(index_path) as handle:
        names = json.load(handle).get("weight_map", {})

    prefixes: collections.Counter[str] = collections.Counter()
    for name in names:
        if name.startswith("mtp.") or "vision" in name:
            continue
        match = re.match(r"^((?:[\w.]+\.)?layers\.)\d+\.", name)
        if match:
            prefixes[match.group(1)] += 1
    layer_prefix = prefixes.most_common(1)[0][0] if prefixes else default_layer_prefix
    shared = "mlp.shared_experts." if any(".mlp.shared_experts." in name for name in names) else default_shared_expert
    return layer_prefix, shared


def _layer_prefix_from_mapping(mapping: dict) -> str | None:
    """Return the decoder-layer prefix declared by an mbridge mapping table."""
    for hf_names in mapping.values():
        names = hf_names if isinstance(hf_names, (list, tuple)) else [hf_names]
        for name in names:
            match = re.match(r"^((?:[\w.]+\.)?layers\.)\{layer_number\}", str(name))
            if match:
                return match.group(1)
    return None


def mbridge_cross_check(model_type: str | None, layer_prefix: str) -> None:
    """Warn if optional mbridge conversion metadata disagrees with HF naming."""
    if not model_type:
        return
    try:
        import miles_plugins.mbridge  # noqa: F401  (registers Miles bridge subclasses)
        from mbridge.core.bridge import _MODEL_REGISTRY
    except Exception:
        return
    bridge_cls = _MODEL_REGISTRY.get(model_type)
    if bridge_cls is None:
        return
    expected = _layer_prefix_from_mapping(getattr(bridge_cls, "_ATTENTION_MAPPING", None) or {})
    if expected is not None and expected != layer_prefix:
        logger.warning(
            "[lora-native] adapter layer prefix %r (from the checkpoint weight index) disagrees with "
            "mbridge's %s mapping (%r); trusting the checkpoint.",
            layer_prefix,
            model_type,
            expected,
        )


def export_lora_hf_named(model_chunks) -> list[tuple[str, torch.Tensor]]:
    """Materialize full HF/PEFT adapter tensors on every TP rank.

    This representation preserves the user's exact target set for checkpoint
    interoperability. The serving-only exporter consumes it and may add zero
    siblings required by SGLang's fused buffers. PP assembly remains with the
    shared Miles checkpoint orchestrator until the bridge path is split in a
    later refactor.
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
    custom = adapter.load_plan_custom(take)
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
        a_parameter = getattr(adapter, f"{projection.attr}_A")
        b_parameter = getattr(adapter, f"{projection.attr}_B")
        a_name = f"{adapter.hf_prefix}{projection.hf}.lora_A.weight"
        b_name = f"{adapter.hf_prefix}{projection.hf}.lora_B.weight"
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
