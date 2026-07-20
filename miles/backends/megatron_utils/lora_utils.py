"""LoRA utilities for Megatron backend using Megatron-Bridge PEFT integration."""

import logging
import os
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.lora import is_lora_enabled

logger = logging.getLogger(__name__)

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
    # GDN (Qwen3.5/Qwen3-Next): both slices live in the single fused megatron in_proj
    "in_proj_qkvz": "in_proj",
    "in_proj_ba": "in_proj",
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
    "in_proj_qkvz": "in_proj",
    "in_proj_ba": "in_proj",
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
    # GDN linear attention: SGLang serves the fused in_proj as two modules
    "in_proj": ["in_proj_qkvz", "in_proj_ba"],
}

_HF_MODULE_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkvz",
    "in_proj_ba",
}

# DeepSeek / Kimi MLA (HF names on checkpoint; Megatron uses linear_* from Megatron-Bridge mappings).
_MLA_HF_TO_MEGATRON = {
    "q_a_proj": "linear_q_down_proj",
    "kv_a_proj_with_mqa": "linear_kv_down_proj",
    "q_b_proj": "linear_q_up_proj",
    "kv_b_proj": "linear_kv_up_proj",
    # DSA indexer (GLM-5 / DeepSeek-V3.2): HF/SGLang leaf names vs Megatron-Bridge linear_* names.
    "wq_b": "linear_wq_b",
    "wk": "linear_wk",
    "weights_proj": "linear_weights_proj",
}
_MEGATRON_MLA_TO_HF = {v: k for k, v in _MLA_HF_TO_MEGATRON.items()}

# Empty: dropping a module here makes sglang silently skip its shipped adapter tensors.
_SGLANG_UNSUPPORTED_HF_TARGETS = frozenset()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def lora_base_cpu_backup_enabled(args: Namespace) -> bool:
    """LoRA + --colocate + --lora-base-cpu-backup all set."""
    return is_lora_enabled(args) and getattr(args, "colocate", False) and getattr(args, "lora_base_cpu_backup", False)


def lora_rollout_base_retained(args: Namespace) -> bool:
    return (
        getattr(args, "reload_rollout_weights_from_disk", False)
        or not getattr(args, "offload_rollout", False)
        or "weight" not in args.offload_rollout_level
    )


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


_marked_lora_grad_params_cache: dict[int, list[tuple[torch.nn.Parameter, str]]] = {}


def reduce_marked_lora_grads(model: Sequence[torch.nn.Module]) -> None:
    """Sum native-LoRA partial gradients over their TP or EP domain."""
    from megatron.core import parallel_state

    key = id(model[0]) if model else 0
    marked = _marked_lora_grad_params_cache.get(key)
    if marked is None:
        marked = []
        for model_chunk in model:
            for parameter in model_chunk.parameters():
                group_name = getattr(parameter, "_lora_grad_sum_group", None)
                if group_name is not None and parameter.requires_grad:
                    marked.append((parameter, group_name))
        _marked_lora_grad_params_cache[key] = marked

    groups = {
        "tp": (
            parallel_state.get_tensor_model_parallel_group(),
            parallel_state.get_tensor_model_parallel_world_size(),
        ),
        "ep": (
            parallel_state.get_expert_model_parallel_group(),
            parallel_state.get_expert_model_parallel_world_size(),
        ),
    }
    for group_name, (group, world_size) in groups.items():
        if world_size <= 1:
            continue
        gradients = []
        for parameter, parameter_group_name in marked:
            if parameter_group_name != group_name:
                continue
            gradient = getattr(parameter, "main_grad", None)
            if gradient is None:
                gradient = parameter.grad
            if gradient is not None:
                gradients.append(gradient)
        for dtype in {gradient.dtype for gradient in gradients}:
            same_dtype = [gradient for gradient in gradients if gradient.dtype == dtype]
            flat = torch._utils._flatten_dense_tensors(same_dtype)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
            for gradient, reduced in zip(
                same_dtype,
                torch._utils._unflatten_dense_tensors(flat, same_dtype),
                strict=True,
            ):
                gradient.copy_(reduced)


def validate_lora_gradients(
    model: Sequence[torch.nn.Module],
    *,
    stage: str = "finalized",
    require_nonzero: bool = True,
) -> tuple[str, torch.nn.Parameter, torch.Tensor] | None:
    """Validate finalized LoRA gradients and retain one B-factor update probe."""
    param_count = 0
    grad_count = 0
    b_param_count = 0
    grad_max_values = []
    grad_l2_squared_values = []
    b_candidates = []

    for model_chunk in model:
        for name, parameter in model_chunk.named_parameters():
            if not parameter.requires_grad or not _is_adapter_param_name(name):
                continue
            is_b_factor = "lora_B" in name
            param_count += 1
            b_param_count += int(is_b_factor)
            gradient = getattr(parameter, "main_grad", None)
            if gradient is None:
                gradient = parameter.grad
            if gradient is None:
                continue
            grad_count += 1
            grad_max_abs = gradient.detach().abs().max().double()
            grad_norm = torch.linalg.vector_norm(gradient.detach()).double()
            grad_max_values.append(grad_max_abs)
            grad_l2_squared_values.append(grad_norm.square())
            if is_b_factor:
                b_candidates.append((name, parameter, grad_max_abs))

    device = next(model[0].parameters()).device
    grad_max_values_tensor = (
        torch.stack(grad_max_values) if grad_max_values else torch.zeros(1, dtype=torch.float64, device=device)
    )
    b_max_values_tensor = (
        torch.stack([candidate[2] for candidate in b_candidates])
        if b_candidates
        else torch.zeros(1, dtype=torch.float64, device=device)
    )
    local_counts = torch.stack(
        [
            torch.tensor(param_count, dtype=torch.float64, device=device),
            torch.tensor(grad_count, dtype=torch.float64, device=device),
            (grad_max_values_tensor > 0).sum().double(),
            torch.tensor(b_param_count, dtype=torch.float64, device=device),
            (b_max_values_tensor > 0).sum().double(),
        ]
    )
    local_l2_squared = (
        torch.stack(grad_l2_squared_values).sum()
        if grad_l2_squared_values
        else torch.zeros(1, dtype=torch.float64, device=device)
    ).reshape(1)
    local_max_abs = grad_max_values_tensor.max().reshape(1)
    local_nonfinite = (
        torch.stack(
            [
                (~torch.isfinite(grad_max_values_tensor)).any(),
                (~torch.isfinite(local_l2_squared)).any(),
            ]
        )
        .any()
        .to(torch.int32)
    )

    probe = None
    if b_candidates:
        probe_idx = b_max_values_tensor.argmax().item()
        probe_name, probe_parameter, _ = b_candidates[probe_idx]
        probe = (probe_name, probe_parameter, probe_parameter.detach().clone())

    torch.distributed.all_reduce(local_counts, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_l2_squared, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_max_abs, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(local_nonfinite, op=torch.distributed.ReduceOp.MAX)
    if torch.distributed.get_rank() == 0:
        logger.info(
            "LoRA gradient check stage=%s: params=%d with_grad=%d nonzero=%d "
            "b_params=%d nonzero_b=%d l2=%.8g max_abs=%.8g",
            stage,
            *(int(value.item()) for value in local_counts),
            local_l2_squared.sqrt().item(),
            local_max_abs.item(),
        )
    if local_nonfinite.item():
        raise RuntimeError("LoRA gradient check found a non-finite gradient after backward")
    if local_counts[0].item() == 0:
        raise RuntimeError("LoRA gradient check found no trainable adapter parameters")
    if not require_nonzero:
        return None
    if local_counts[2].item() == 0:
        raise RuntimeError("LoRA gradient check found no nonzero gradient after backward")
    if local_counts[4].item() == 0:
        raise RuntimeError("LoRA gradient check found no nonzero B-factor gradient after backward")
    assert probe is not None
    return probe


def validate_lora_optimizer_update(probe: tuple[str, torch.nn.Parameter, torch.Tensor]) -> None:
    """Verify that a nonzero-gradient B-factor changed in the optimizer step."""
    probe_name, parameter, before = probe
    local_max_delta = (parameter.detach() - before).abs().max().double().reshape(1)
    local_changed = (local_max_delta > 0).to(torch.int64)

    torch.distributed.all_reduce(local_changed, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_max_delta, op=torch.distributed.ReduceOp.MAX)
    if torch.distributed.get_rank() == 0:
        logger.info(
            "LoRA optimizer update check: changed_ranks=%d max_abs_delta=%.8g rank0_probe=%s",
            local_changed.item(),
            local_max_delta.item(),
            probe_name,
        )
    if local_changed.item() == 0:
        raise RuntimeError("LoRA optimizer update check found no changed B-factor after optimizer step")


def _is_adapter_param_name(name: str) -> bool:
    """Check if a parameter name belongs to a LoRA adapter (Megatron internal naming)."""
    return "lora_" in name or (".adapter." in name and ("linear_in" in name or "linear_out" in name))


class LoRABackwardDiagnostics:
    _PROBE_SUFFIXES = (
        "o_lora_B",
        "q_a_lora_B",
        "kv_a_lora_B",
        "fc1_lora_B",
        "fc2_lora_B",
        "w1_lora_B",
        "w2_lora_B",
        "w3_lora_B",
    )

    def __init__(self, model: Sequence[torch.nn.Module]) -> None:
        named_parameters = {
            name: parameter for model_chunk in model for name, parameter in model_chunk.named_parameters()
        }
        candidates: dict[str, tuple[str, torch.nn.Parameter]] = {}
        for name, parameter in named_parameters.items():
            if not parameter.requires_grad or not _is_adapter_param_name(name):
                continue
            for suffix in self._PROBE_SUFFIXES:
                if name.endswith(suffix) and (suffix not in candidates or name > candidates[suffix][0]):
                    candidates[suffix] = (name, parameter)

        if not candidates:
            raise RuntimeError("LoRA backward diagnostics found no native B-factor parameters")

        self._device = next(iter(candidates.values()))[1].device
        self._probe_names: dict[str, str] = {}
        self._initial_b_max: dict[str, torch.Tensor] = {}
        self._initial_a_max: dict[str, torch.Tensor] = {}
        self._grad_max_values: dict[str, list[torch.Tensor]] = {"logits": []}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for suffix, (name, parameter) in candidates.items():
            self._probe_names[suffix] = name
            self._initial_b_max[suffix] = parameter.detach().abs().max().double()
            a_name = name.removesuffix("_B") + "_A"
            if a_name not in named_parameters:
                raise RuntimeError(f"LoRA backward diagnostics found no A factor for {name}")
            self._initial_a_max[suffix] = named_parameters[a_name].detach().abs().max().double()
            self._grad_max_values[suffix] = []

            def capture(gradient: torch.Tensor, *, key: str = suffix) -> torch.Tensor:
                self._grad_max_values[key].append(gradient.detach().abs().max().double())
                return gradient

            self._handles.append(parameter.register_hook(capture))

    def register_logits(self, logits: torch.Tensor) -> None:
        if not logits.requires_grad:
            raise RuntimeError("LoRA backward diagnostics found logits without an autograd graph")

        def capture(gradient: torch.Tensor) -> torch.Tensor:
            self._grad_max_values["logits"].append(gradient.detach().abs().max().double())
            return gradient

        self._handles.append(logits.register_hook(capture))

    def report(self) -> dict[str, tuple[int, int, float, float, float]]:
        keys = ["logits", *self._probe_names]
        counts = torch.tensor(
            [
                value
                for key in keys
                for value in (
                    len(self._grad_max_values[key]),
                    sum(gradient.item() > 0 for gradient in self._grad_max_values[key]),
                )
            ],
            dtype=torch.int64,
            device=self._device,
        )
        max_values = torch.stack(
            [
                (
                    torch.stack(self._grad_max_values[key]).max()
                    if self._grad_max_values[key]
                    else torch.zeros((), dtype=torch.float64, device=self._device)
                )
                for key in keys
            ]
        )
        initial_b_values = torch.stack(
            [
                (
                    torch.zeros((), dtype=torch.float64, device=self._device)
                    if key == "logits"
                    else self._initial_b_max[key]
                )
                for key in keys
            ]
        )
        initial_a_values = torch.stack(
            [
                (
                    torch.zeros((), dtype=torch.float64, device=self._device)
                    if key == "logits"
                    else self._initial_a_max[key]
                )
                for key in keys
            ]
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_values, op=dist.ReduceOp.MAX)
        dist.all_reduce(initial_b_values, op=dist.ReduceOp.MAX)
        dist.all_reduce(initial_a_values, op=dist.ReduceOp.MAX)

        summary = {
            key: (
                counts[2 * index].item(),
                counts[2 * index + 1].item(),
                max_values[index].item(),
                initial_b_values[index].item(),
                initial_a_values[index].item(),
            )
            for index, key in enumerate(keys)
        }
        if dist.get_rank() == 0:
            logger.info(
                "LoRA raw backward diagnostics "
                "(hook_calls, nonzero_hook_calls, raw_grad_max_abs, initial_B_max_abs, initial_A_max_abs): %s",
                summary,
            )
            logger.info("LoRA raw backward probe names: %s", self._probe_names)
        return summary

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


_param_grad_buffer_patched = False


def _configure_lora_buffer_cpu_backup(kwargs: dict) -> None:
    kwargs["disable_grad_buffers_cpu_backup"] = True


def patch_param_grad_buffer_for_colocate_mode_lora() -> None:
    """Patch _ParamAndGradBuffer to disable CPU backup for gradient buffers.

    In colocate mode with offload_train, torch_memory_saver.pause(tag="default")
    offloads default-region GPU memory.  During LoRA training, base weights are
    frozen (requires_grad=False) so DDP only creates buffers for adapter params.

    Adapter parameter buffers must remain in the default region so their CPU
    backups are available to update_weights while the trainer sleeps. Gradient
    buffers can be discarded and rebuilt, so they use a separate no-backup
    region that sleep()/wake_up() pauses and resumes explicitly.

    The patch is idempotent and only takes effect once.
    """
    global _param_grad_buffer_patched
    if _param_grad_buffer_patched:
        return
    _param_grad_buffer_patched = True

    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer

    _original_init = _ParamAndGradBuffer.__init__

    def _patched_init(self, *args, **kwargs):
        _configure_lora_buffer_cpu_backup(kwargs)
        _original_init(self, *args, **kwargs)

    _ParamAndGradBuffer.__init__ = _patched_init
    logger.info("Patched _ParamAndGradBuffer.__init__ for LoRA colocate mode (discard gradient buffers on sleep)")


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

    if isinstance(hf_modules, tuple):
        hf_modules = list(hf_modules)

    # Check if already in Megatron format (standard / canonical / Kimi MLA linear_*).
    if all(m not in _HF_MODULE_NAMES and m not in _MLA_HF_TO_MEGATRON for m in hf_modules if "*" not in m):
        return list(hf_modules)

    # Convert HF names to Megatron names (dedup while preserving order)
    megatron_modules: list[str] = []
    for module in hf_modules:
        if module in _MLA_HF_TO_MEGATRON:
            megatron_name = _MLA_HF_TO_MEGATRON[module]
        else:
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
    Kimi MLA Megatron:   linear_q_down_proj -> q_a_proj, linear_kv_down_proj -> kv_a_proj_with_mqa, ...

    Wildcards (``*.layers.2.mlp.experts.linear_fc1``) get the last dotted
    segment mapped to an HF leaf name; SGLang uses the result to choose
    adapter-buffer types, not to scope by layer.
    """
    if isinstance(megatron_modules, tuple):
        megatron_modules = list(megatron_modules)
    hf_modules: list[str] = []
    for module in megatron_modules:
        lookup_key = module.rsplit(".", 1)[-1] if "." in module else module
        if lookup_key in _MEGATRON_MLA_TO_HF:
            hf_modules.append(_MEGATRON_MLA_TO_HF[lookup_key])
        elif lookup_key in _MEGATRON_TO_HF_MODULES:
            hf_modules.extend(_MEGATRON_TO_HF_MODULES[lookup_key])
        else:
            # same-name passthrough; SGLang needs the leaf, not a path or pattern
            hf_modules.append(lookup_key)
    seen: set[str] = set()
    unique: list[str] = []
    for m in hf_modules:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def target_modules_hf_for_sglang_rollout(args: Namespace) -> list[str]:
    """HF target_modules for SGLang LoRA init/sync (minus _SGLANG_UNSUPPORTED_HF_TARGETS, currently empty)."""
    raw = list(args.target_modules) if args.target_modules else []
    hf = convert_target_modules_to_hf(raw)
    out = [m for m in hf if m not in _SGLANG_UNSUPPORTED_HF_TARGETS]
    dropped = set(hf) - set(out)
    if dropped:
        logger.warning(
            "target_modules_hf_for_sglang_rollout: omitting %s for SGLang (unsupported by default "
            "get_hidden_dim); Megatron should not train LoRA on these if rollout sync is required.",
            sorted(dropped),
        )
    return out


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

    lora_kwargs = dict(
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )
    # shared-outer grouped-expert LoRA (SGLang PR #21466); per-expert is the default
    if getattr(args, "experts_shared_outer_loras", False):
        assert lora_cls is LoRA, "--experts-shared-outer-loras requires the standard LoRA adapter type"
        lora_kwargs["experts_shared_outer_loras"] = True

    lora = lora_cls(**lora_kwargs)

    logger.info(
        f"Created {lora_cls.__name__}: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}, target_modules={target_modules}, "
        f"exclude_modules={exclude_modules}"
    )
    return lora


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------


def save_lora_checkpoint(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    save_dir: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    iteration: int | None = None,
) -> str:
    """Save LoRA adapter checkpoint to disk.

    Bridge LoRA saves both HF PEFT and Megatron-native formats. Native Kimi K3
    LoRA saves one Megatron shard per global rank; materializing its full 896-expert
    HF adapter on every rank is intentionally kept out of the training checkpoint
    path.

    When ``optimizer`` is provided, training state (optimizer + LR scheduler) is
    also saved per-rank for checkpoint resume. Base model weights are frozen and
    never change, so they are not saved.

    This function is collective: all ranks must call it.
    """
    import json

    save_path = Path(save_dir)
    native_kimi_k3 = args.megatron_to_hf_mode == "raw" and "kimi_k3" in (args.model_name or "").lower()
    is_dp_rank_0 = get_parallel_state().effective_dp.rank == 0
    tp_rank = get_parallel_state().tp.rank
    pp_rank = get_parallel_state().pp.rank
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    if is_dp_rank_0 or native_kimi_k3:
        save_path.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    # ---- Megatron-native format (per TP/PP rank, fast resume) ----
    if is_dp_rank_0 or native_kimi_k3:
        adapter_state: dict[str, torch.Tensor] = {}
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if _is_adapter_param_name(name):
                    adapter_state[name] = param.data.cpu()

        if native_kimi_k3:
            native_path = save_path / f"adapter_megatron_rank{global_rank}.pt"
        else:
            native_path = save_path / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
        torch.save(adapter_state, native_path)
        logger.info(f"Saved {len(adapter_state)} adapter tensors (native) to {native_path}")

    if native_kimi_k3:
        if global_rank == 0:
            target_modules_hf = convert_target_modules_to_hf(list(args.target_modules))
            config = {
                "peft_type": "LORA",
                "r": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "target_modules": target_modules_hf,
                "lora_dropout": args.lora_dropout,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "experts_shared_outer_loras": True,
                "format": "megatron_rank_sharded",
            }
            with open(save_path / "adapter_config.json", "w") as f:
                json.dump(config, f, indent=2)
    else:
        from megatron.bridge import AutoBridge

        from miles.utils import megatron_bridge_utils

        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        lora_state_dict: dict[str, torch.Tensor] = {}
        with megatron_bridge_utils.patch_megatron_model(model):
            for hf_name, weight, _megatron_name in bridge.export_adapter_weights(
                model,
                cpu=True,
                show_progress=False,
            ):
                lora_state_dict[hf_name] = weight

        if is_dp_rank_0 and tp_rank == 0:
            torch.save(lora_state_dict, save_path / "adapter_model.bin")

            target_modules_hf = (
                convert_target_modules_to_hf(list(args.target_modules))
                if args.target_modules
                else [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            )
            config = {
                "peft_type": "LORA",
                "r": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "target_modules": target_modules_hf,
                "lora_dropout": args.lora_dropout,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
            with open(save_path / "adapter_config.json", "w") as f:
                json.dump(config, f, indent=2)

            os.sync()
            logger.info(f"Saved HF PEFT adapter to {save_path} with " f"{len(lora_state_dict)} tensors")

    if native_kimi_k3 and global_rank == 0:
        os.sync()
        logger.info(f"Saved rank-sharded Kimi K3 adapter to {save_path}")

    # ---- Training state (optimizer + scheduler) for resume ----
    if optimizer is not None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        torch.save(
            {
                "iteration": iteration,
                "optimizer": optimizer.state_dict(),
                "opt_param_scheduler": opt_param_scheduler.state_dict() if opt_param_scheduler else None,
            },
            save_path / f"training_state_rank{rank}.pt",
        )
        logger.info(f"Saved optimizer/scheduler state to {save_path}")

    if dist.is_initialized():
        dist.barrier()

    return str(save_path)


def load_lora_adapter(
    model: Sequence[torch.nn.Module],
    adapter_path: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
) -> tuple[bool, int | None]:
    """Load LoRA adapter weights from a saved checkpoint into the model.

    Attempts to load from Megatron-native format first (per-rank ``.pt`` files),
    which preserves the exact TP/PP sharding and requires no name conversion.
    Falls back to HF PEFT ``adapter_model.bin`` if native files are not found
    (not yet implemented for HF PEFT format).

    When ``optimizer`` is provided, also restores training state (optimizer +
    LR scheduler) from a co-located ``training_state_rank*.pt`` file.

    Args:
        model: List of DDP-wrapped model chunks with LoRA layers already applied.
        adapter_path: Path to the adapter checkpoint directory.
        optimizer: If provided, restore optimizer state for training resume.
        opt_param_scheduler: If provided, restore LR scheduler state.

    Returns:
        ``(loaded, iteration)`` — *loaded* is True if adapter weights were
        successfully loaded; *iteration* is the saved iteration number (or None
        if no training state was found).
    """
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        logger.warning(f"LoRA adapter path does not exist: {adapter_dir}")
        return False, None

    tp_rank = get_parallel_state().tp.rank
    pp_rank = get_parallel_state().pp.rank

    # ---- Try Megatron-native format first (fast, no conversion needed) ----
    global_rank = dist.get_rank() if dist.is_initialized() else 0
    native_path = adapter_dir / f"adapter_megatron_rank{global_rank}.pt"
    if not native_path.exists():
        config_path = adapter_dir / "adapter_config.json"
        if config_path.exists():
            import json

            with open(config_path) as f:
                adapter_config = json.load(f)
            if adapter_config.get("format") == "megatron_rank_sharded":
                raise FileNotFoundError(
                    f"Missing Kimi K3 adapter shard for global rank {global_rank}: " f"{native_path}"
                )
        native_path = adapter_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
    if native_path.exists():
        state_dict = torch.load(native_path, map_location="cpu", weights_only=True)
        adapter_params = {
            name: param
            for model_chunk in model
            for name, param in model_chunk.named_parameters()
            if _is_adapter_param_name(name)
        }
        missing = adapter_params.keys() - state_dict.keys()
        unexpected = state_dict.keys() - adapter_params.keys()
        if missing or unexpected:
            raise RuntimeError(
                f"Adapter checkpoint parameter mismatch: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if name in state_dict:
                    param.data.copy_(state_dict[name].to(device=param.device))
        logger.info(f"Loaded {len(adapter_params)} adapter tensors from " f"Megatron-native checkpoint: {native_path}")

        iteration = _load_training_state(adapter_dir, optimizer, opt_param_scheduler)
        return True, iteration

    # ---- HF PEFT format (future work) ----
    hf_path = adapter_dir / "adapter_model.bin"
    if hf_path.exists():
        logger.warning(
            f"Found HF PEFT adapter at {hf_path} but direct HF PEFT loading into "
            f"Megatron is not yet supported. Please save using Megatron-native format "
            f"(adapter_megatron_tp*_pp*.pt files) for checkpoint resume."
        )
        return False, None

    logger.warning(f"No adapter checkpoint found at {adapter_dir}")
    return False, None


def _load_training_state(
    adapter_dir: Path,
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
) -> int | None:
    """Restore optimizer/scheduler state saved alongside a LoRA adapter checkpoint."""
    if optimizer is None:
        return None

    rank = dist.get_rank() if dist.is_initialized() else 0
    state_path = adapter_dir / f"training_state_rank{rank}.pt"
    if not state_path.exists():
        return None

    # Optimizer state dicts may contain non-tensor objects (e.g. step counts,
    # param group metadata), so full unpickling is required here.
    training_state = torch.load(state_path, map_location="cpu", weights_only=False)

    optimizer.load_state_dict(training_state["optimizer"])
    logger.info("Restored optimizer state from LoRA checkpoint")

    if opt_param_scheduler is not None and training_state.get("opt_param_scheduler") is not None:
        opt_param_scheduler.load_state_dict(training_state["opt_param_scheduler"])
        logger.info("Restored LR scheduler state from LoRA checkpoint")

    iteration = training_state.get("iteration")
    if iteration is not None:
        logger.info(f"Resuming LoRA training from iteration {iteration}")
    return iteration


# ---------------------------------------------------------------------------
# LoRA config dict for weight sync to SGLang
# ---------------------------------------------------------------------------


def build_lora_sync_config(args: Namespace) -> dict[str, Any]:
    """Build LoRA config dict for syncing weights to SGLang engines."""
    target_modules_hf = (
        target_modules_hf_for_sglang_rollout(args)
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
