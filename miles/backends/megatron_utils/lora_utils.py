"""LoRA utilities for Megatron backend using Megatron-Bridge PEFT integration."""

import logging
import os
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu

logger = logging.getLogger(__name__)

LORA_ADAPTER_NAME = "miles_lora"

# ---------------------------------------------------------------------------
# Unified HF <-> Megatron module name mappings
# ---------------------------------------------------------------------------

# Standard LoRA: merged Q/K/V and merged up/gate
_STANDARD_LORA_HF_TO_MEGATRON = {
    "q_proj": "linear_qkv",
    "k_proj": "linear_qkv",
    "v_proj": "linear_qkv",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1",
    "up_proj": "linear_fc1",
    "down_proj": "linear_fc2",
}

_STANDARD_LORA_ALL_MODULES = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

# CanonicalLoRA: Split Q/K/V and up/gate
_CANONICAL_LORA_HF_TO_MEGATRON = {
    "q_proj": "linear_q",
    "k_proj": "linear_k",
    "v_proj": "linear_v",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1_gate",
    "up_proj": "linear_fc1_up",
    "down_proj": "linear_fc2",
}

_CANONICAL_LORA_ALL_MODULES = [
    "linear_q",
    "linear_k",
    "linear_v",
    "linear_proj",
    "linear_fc1_up",
    "linear_fc1_gate",
    "linear_fc2",
]

# Megatron -> HF (inverse mapping, one-to-many)
# Covers both standard LoRA (merged) and CanonicalLoRA (split) module names.
_MEGATRON_TO_HF_MODULES = {
    # Standard LoRA (merged layers)
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
    "linear_fc1": ["gate_proj", "up_proj"],
    "linear_fc2": ["down_proj"],
    # CanonicalLoRA (split layers)
    "linear_q": ["q_proj"],
    "linear_k": ["k_proj"],
    "linear_v": ["v_proj"],
    "linear_fc1_gate": ["gate_proj"],
    "linear_fc1_up": ["up_proj"],
}

_HF_MODULE_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def is_lora_enabled(args: Namespace) -> bool:
    """Check if LoRA is enabled based on arguments."""
    return getattr(args, "lora_rank", 0) > 0 or getattr(args, "lora_adapter_path", None) is not None


def is_lora_model(model: Sequence[torch.nn.Module]) -> bool:
    """Check if model has LoRA layers applied."""
    for model_chunk in model:
        if hasattr(model_chunk.module, "peft_config"):
            return True
        for name, _ in model_chunk.named_parameters():
            if "lora_" in name or "adapter" in name:
                return True
    return False


def is_lora_weight_name(name: str) -> bool:
    """Check if a weight name corresponds to a LoRA adapter weight."""
    return ".lora_A." in name or ".lora_B." in name


def _is_adapter_param_name(name: str) -> bool:
    """Check if a parameter name belongs to a LoRA adapter (Megatron internal naming)."""
    return "lora_" in name or (".adapter." in name and ("linear_in" in name or "linear_out" in name))


# ---------------------------------------------------------------------------
# Module name conversion
# ---------------------------------------------------------------------------


def _get_lora_class_name(lora_type: type | object | None) -> str:
    """Resolve LoRA type to its class name string."""
    if lora_type is None:
        return "CanonicalLoRA"
    if isinstance(lora_type, type):
        return lora_type.__name__
    return type(lora_type).__name__


def convert_target_modules_to_megatron(
    hf_modules: str | list[str],
    lora_type: type | object | None = None,
) -> list[str]:
    """Convert HuggingFace LoRA target module names to Megatron format.

    HF:  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    Megatron (LoRA):          linear_qkv, linear_proj, linear_fc1, linear_fc2
    Megatron (CanonicalLoRA): linear_q, linear_k, linear_v, linear_proj,
                              linear_fc1_up, linear_fc1_gate, linear_fc2

    Special values: "all", "all-linear", "all_linear" -> all standard linear modules.
    If input is already in Megatron format, returns as-is.
    """
    class_name = _get_lora_class_name(lora_type)
    is_canonical = class_name == "CanonicalLoRA"

    all_modules = _CANONICAL_LORA_ALL_MODULES if is_canonical else _STANDARD_LORA_ALL_MODULES
    hf_to_megatron = _CANONICAL_LORA_HF_TO_MEGATRON if is_canonical else _STANDARD_LORA_HF_TO_MEGATRON

    # Handle special "all-linear" variants
    if isinstance(hf_modules, str):
        if hf_modules in ("all", "all-linear", "all_linear"):
            return list(all_modules)
        hf_modules = [hf_modules]
    elif isinstance(hf_modules, list) and len(hf_modules) == 1:
        if hf_modules[0] in ("all", "all-linear", "all_linear"):
            return list(all_modules)

    # Check if already in Megatron format
    if all(m not in _HF_MODULE_NAMES for m in hf_modules if "*" not in m):
        return hf_modules

    # Convert HF names to Megatron names (dedup while preserving order)
    megatron_modules: list[str] = []
    for module in hf_modules:
        megatron_name = hf_to_megatron.get(module, module)
        if megatron_name not in megatron_modules:
            megatron_modules.append(megatron_name)

    return megatron_modules


def convert_target_modules_to_hf(megatron_modules: list[str]) -> list[str]:
    """Convert Megatron LoRA target module names to HuggingFace format.

    Supports both standard LoRA and CanonicalLoRA module names.

    Megatron standard:   linear_qkv, linear_proj, linear_fc1, linear_fc2
    Megatron canonical:  linear_q, linear_k, linear_v, linear_proj,
                         linear_fc1_up, linear_fc1_gate, linear_fc2
    HF:                  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    """
    hf_modules: list[str] = []
    for module in megatron_modules:
        if module in _MEGATRON_TO_HF_MODULES:
            hf_modules.extend(_MEGATRON_TO_HF_MODULES[module])
        else:
            hf_modules.append(module)
    return hf_modules


# ---------------------------------------------------------------------------
# Model setup helpers (used by model.py)
# ---------------------------------------------------------------------------


def parse_exclude_modules(args: Namespace, lora_type=None) -> list[str]:
    """Parse and convert exclude_modules argument."""
    exclude_modules: list[str] = []
    raw = getattr(args, "exclude_modules", None)
    if raw:
        if isinstance(raw, str):
            exclude_modules = [m.strip() for m in raw.split(",")]
        else:
            exclude_modules = list(raw)
        exclude_modules = convert_target_modules_to_megatron(exclude_modules, lora_type=lora_type)
    return exclude_modules


def create_lora_instance(args: Namespace):
    """Create a LoRA or CanonicalLoRA instance based on args.

    Returns:
        A LoRA/CanonicalLoRA dataclass instance ready to be applied to a model.
    """
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.peft.lora import LoRA

    lora_type_name = getattr(args, "lora_type", "lora").lower()

    if lora_type_name == "canonical_lora":
        lora_cls = CanonicalLoRA
    else:
        lora_cls = LoRA

    target_modules = convert_target_modules_to_megatron(args.target_modules, lora_type=lora_cls)
    exclude_modules = parse_exclude_modules(args, lora_type=lora_cls)

    lora = lora_cls(
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )

    logger.info(
        f"Created {lora_cls.__name__}: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}, target_modules={target_modules}, "
        f"exclude_modules={exclude_modules}"
    )
    return lora


# ---------------------------------------------------------------------------
# Parameter freezing / trainable param helpers
# ---------------------------------------------------------------------------


def freeze_base_model(model: Sequence[torch.nn.Module]) -> None:
    """Freeze base model parameters, only keep LoRA trainable."""
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            if "lora_" not in name and "adapter" not in name:
                param.requires_grad = False


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------


def save_lora_checkpoint(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    save_dir: str,
) -> str:
    """Save LoRA adapter checkpoint to disk.

    Saves in two formats:
    1. **HF PEFT format** (``adapter_model.bin`` + ``adapter_config.json``) for
       external tool compatibility. Uses Megatron-Bridge's ``export_adapter_weights``
       which correctly handles fused QKV / gate-up weight splitting and TP gathering.
    2. **Megatron-native format** (``adapter_megatron_tp{tp}_pp{pp}.pt``) for fast
       checkpoint resume without name/weight conversion. Each TP/PP rank saves its
       own shard with original parameter names.

    This function is collective: **all ranks must call it** because the bridge
    export performs TP all-gather internally. Only ``dp_rank == 0`` writes files.
    """
    import json

    from megatron.bridge import AutoBridge

    from miles.utils import megatron_bridge_utils

    save_path = Path(save_dir)
    is_dp_rank_0 = mpu.get_data_parallel_rank() == 0
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    # Create directory on dp_rank=0, then synchronize
    if is_dp_rank_0:
        save_path.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    # ---- Megatron-native format (per TP/PP rank, fast resume) ----
    if is_dp_rank_0:
        adapter_state: dict[str, torch.Tensor] = {}
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if _is_adapter_param_name(name):
                    adapter_state[name] = param.data.cpu()

        native_path = save_path / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
        torch.save(adapter_state, native_path)
        logger.info(f"Saved {len(adapter_state)} adapter tensors (native) to {native_path}")

    # ---- HF PEFT format (uses bridge for correct name/weight conversion) ----
    # Bridge export is collective: all TP ranks participate in the all-gather,
    # so every rank must call export_adapter_weights.
    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)

    lora_state_dict: dict[str, torch.Tensor] = {}
    with megatron_bridge_utils.patch_megatron_model(model):
        for hf_name, weight, _megatron_name in bridge.export_adapter_weights(
            model,
            cpu=True,
            show_progress=False,
        ):
            lora_state_dict[hf_name] = weight

    # Only one rank writes the HF PEFT files (bridge already gathered across TP)
    if is_dp_rank_0 and tp_rank == 0:
        torch.save(lora_state_dict, save_path / "adapter_model.bin")

        target_modules_hf = (
            convert_target_modules_to_hf(list(args.target_modules))
            if args.target_modules
            else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        config = {
            "peft_type": "LORA",
            "r": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": target_modules_hf,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        with open(save_path / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)

        os.sync()
        logger.info(f"Saved HF PEFT adapter to {save_path} with {len(lora_state_dict)} tensors")

    if dist.is_initialized():
        dist.barrier()

    return str(save_path)


def load_lora_adapter(
    model: Sequence[torch.nn.Module],
    adapter_path: str,
) -> bool:
    """Load LoRA adapter weights from a saved checkpoint into the model.

    Attempts to load from Megatron-native format first (per-rank ``.pt`` files),
    which preserves the exact TP/PP sharding and requires no name conversion.
    Falls back to HF PEFT ``adapter_model.bin`` if native files are not found
    (not yet implemented for HF PEFT format).

    Args:
        model: List of DDP-wrapped model chunks with LoRA layers already applied.
        adapter_path: Path to the adapter checkpoint directory.

    Returns:
        True if adapter weights were successfully loaded, False otherwise.
    """
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        logger.warning(f"LoRA adapter path does not exist: {adapter_dir}")
        return False

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    # ---- Try Megatron-native format first (fast, no conversion needed) ----
    native_path = adapter_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
    if native_path.exists():
        state_dict = torch.load(native_path, map_location="cpu")
        loaded = 0
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if name in state_dict:
                    param.data.copy_(state_dict[name].to(device=param.device))
                    loaded += 1
        logger.info(f"Loaded {loaded} adapter tensors from Megatron-native checkpoint: {native_path}")
        return True

    # ---- HF PEFT format (future work) ----
    hf_path = adapter_dir / "adapter_model.bin"
    if hf_path.exists():
        logger.warning(
            f"Found HF PEFT adapter at {hf_path} but direct HF PEFT loading into "
            f"Megatron is not yet supported. Please save using Megatron-native format "
            f"(adapter_megatron_tp*_pp*.pt files) for checkpoint resume."
        )
        return False

    logger.warning(f"No adapter checkpoint found at {adapter_dir}")
    return False


# ---------------------------------------------------------------------------
# LoRA config dict for weight sync to SGLang
# ---------------------------------------------------------------------------


def build_lora_sync_config(args: Namespace) -> dict[str, Any]:
    """Build LoRA config dict for syncing weights to SGLang engines."""
    target_modules_hf = (
        convert_target_modules_to_hf(list(args.target_modules))
        if args.target_modules
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    return {
        "peft_type": "LORA",
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules_hf,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
