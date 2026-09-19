"""Native adapter checkpointing: the MCore dist-checkpoint write format plus read-only legacy shards."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from miles_plugins.lora.modules.linear import NATIVE_LORA_SHARDED_STATE_FLAG, iter_adapters

logger = logging.getLogger(__name__)

NATIVE_DIST_CKPT_DIRNAME = "torch_dist"

_OPTIMIZER_SHARDING_METADATA = {"distrib_optim_sharding_type": "fully_reshardable"}


def _unwrap_model_chunk(chunk: nn.Module) -> nn.Module:
    """Strip DDP/Float16 wrappers; MCore's save flow calls sharded_state_dict unwrapped."""
    module = chunk
    while hasattr(module, "module"):
        module = module.module
    return module


def _sharded_entry_dict_key(entry: Any) -> str:
    """Unique merge key: MCore dict keys are chunk-local (VPP chunks collide) and
    ``.key`` alone repeats across homogeneous layers; the (key, offset) pair is unique."""
    return f"{entry.key}|{tuple(entry.global_offset)}"


def native_adapter_sharded_state_dict(model_chunks: Sequence[nn.Module]) -> dict[str, Any]:
    """Adapter-only sharded state dict with layout-invariant (global) keys.

    Standard MCore walk with the opt-in flag, filtered to adapter parameters by
    identity and merged via ``_sharded_entry_dict_key``. Asserts that every
    attached adapter surfaced — a walk that skips adapter children must fail
    loudly, not silently drop weights.
    """
    adapter_parameter_names = {
        id(parameter): f"{type(adapter).__name__}({adapter.hf_prefix}).{name}"
        for adapter in iter_adapters(model_chunks)
        for name, parameter in adapter.named_parameters(recurse=False)
    }
    sharded_state: dict[str, Any] = {}
    for chunk in model_chunks:
        chunk_sharded = _unwrap_model_chunk(chunk).sharded_state_dict(metadata={NATIVE_LORA_SHARDED_STATE_FLAG: True})
        for entry in chunk_sharded.values():
            if id(getattr(entry, "data", None)) not in adapter_parameter_names:
                continue
            merged_key = _sharded_entry_dict_key(entry)
            assert merged_key not in sharded_state, f"duplicate adapter checkpoint shard: {merged_key}"
            sharded_state[merged_key] = entry
    collected = {id(entry.data) for entry in sharded_state.values()}
    missing = sorted(name for pid, name in adapter_parameter_names.items() if pid not in collected)
    assert not missing, (
        f"{len(missing)} adapter parameter(s) never surfaced in the model's sharded_state_dict walk "
        f"(the provider's module tree must recurse into adapter submodules): {missing[:8]}"
    )
    return sharded_state


def save_native_adapter_dist_checkpoint(
    model_chunks: Sequence[nn.Module],
    checkpoint_dir: str | Path,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    iteration: int | None = None,
) -> None:
    """Write adapters (+ optimizer/scheduler state) as one MCore dist checkpoint.

    Collective — every rank must call it.
    """
    from megatron.core import dist_checkpointing

    assert dist.is_initialized(), "native adapter dist checkpointing requires torch.distributed"
    sharded_model_state = native_adapter_sharded_state_dict(model_chunks)
    state_dict: dict[str, Any] = {
        "model": sharded_model_state,
        "iteration": iteration,
        "has_optimizer_state": optimizer is not None,
    }
    if optimizer is not None:
        state_dict["optimizer"] = optimizer.sharded_state_dict(
            sharded_model_state, is_loading=False, metadata=dict(_OPTIMIZER_SHARDING_METADATA)
        )
        state_dict["opt_param_scheduler"] = (
            opt_param_scheduler.state_dict() if opt_param_scheduler is not None else None
        )
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    dist_checkpointing.save(state_dict, str(checkpoint_path))
    logger.info(
        "Saved %d adapter tensors (+ optimizer state: %s) to dist checkpoint %s",
        len(sharded_model_state),
        optimizer is not None,
        checkpoint_path,
    )


def load_native_adapter_dist_checkpoint(
    model_chunks: Sequence[nn.Module],
    checkpoint_dir: str | Path,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
) -> int | None:
    """Load an MCore dist adapter checkpoint, resharding to the current layout.

    Collective — every rank must call it. Incompatible contents (e.g. a
    different ``--target-modules`` set) raise instead of silently
    reinitializing. Returns the saved iteration when optimizer state was
    restored, else None (fresh schedule; fp32 mains realigned via
    ``reload_model_params``).
    """
    from megatron.core import dist_checkpointing

    assert dist.is_initialized(), "native adapter dist checkpointing requires torch.distributed"
    common_state = dist_checkpointing.load_common_state_dict(str(checkpoint_dir))
    restore_optimizer = optimizer is not None and bool(common_state.get("has_optimizer_state"))

    sharded_model_state = native_adapter_sharded_state_dict(model_chunks)
    parameters_by_key = {key: entry.data for key, entry in sharded_model_state.items()}
    state_dict: dict[str, Any] = {"model": sharded_model_state}
    if restore_optimizer:
        state_dict["optimizer"] = optimizer.sharded_state_dict(
            sharded_model_state, is_loading=True, metadata=dict(_OPTIMIZER_SHARDING_METADATA)
        )
    loaded = dist_checkpointing.load(state_dict, str(checkpoint_dir))

    with torch.no_grad():
        for key, tensor in loaded["model"].items():
            parameter = parameters_by_key[key]
            if tensor is not parameter:
                parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    logger.info("Loaded %d adapter tensors from dist checkpoint %s", len(parameters_by_key), checkpoint_dir)

    if restore_optimizer:
        optimizer.load_state_dict(loaded["optimizer"])
        logger.info("Restored optimizer state from dist checkpoint")
        scheduler_state = common_state.get("opt_param_scheduler")
        if opt_param_scheduler is not None and scheduler_state is not None:
            opt_param_scheduler.load_state_dict(scheduler_state)
            logger.info("Restored LR scheduler state from dist checkpoint")
        return common_state.get("iteration")

    if optimizer is not None:
        logger.warning(
            "Dist adapter checkpoint at %s has no optimizer state; resuming with fresh training state.",
            checkpoint_dir,
        )
        reload_model_params = getattr(optimizer, "reload_model_params", None)
        if callable(reload_model_params):
            reload_model_params()
    return None


def is_native_adapter_dist_checkpoint(checkpoint_dir: str | Path) -> bool:
    """Whether ``checkpoint_dir`` holds an MCore distributed checkpoint."""
    path = Path(checkpoint_dir)
    if not path.is_dir():
        return False
    from megatron.core.dist_checkpointing import check_is_distributed_checkpoint

    return check_is_distributed_checkpoint(str(path))


@dataclass
class AdapterLoadPlan:
    """A validated, not-yet-applied adapter state-dict load.

    Keeping validation separate from mutation lets the checkpoint orchestrator
    reach a collective agreement across ranks before any parameter changes.
    """

    assignments: list[tuple[str, nn.Parameter, torch.Tensor]]
    unexpected: list[str]
    missing: list[str]
    shape_mismatches: list[str]

    @property
    def compatible(self) -> bool:
        return not (self.unexpected or self.missing or self.shape_mismatches)

    def apply(self) -> int:
        """Copy every prevalidated tensor and return the number loaded."""
        with torch.no_grad():
            for _, parameter, tensor in self.assignments:
                parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
        return len(self.assignments)


_LEGACY_MODEL_CHUNK_PREFIX = "_miles_model_chunks."


def adapter_load_plan(
    model_chunks: Sequence[nn.Module],
    state_dict: dict[str, torch.Tensor],
    include_parameter: Callable[[str, nn.Parameter], bool],
) -> AdapterLoadPlan:
    """Preflight a legacy adapter state dict without mutating parameters.

    Reads both legacy schemes: ``_miles_model_chunks.{chunk}.{name}`` keys map
    back exactly; flat keys are accepted when they identify exactly one
    parameter, and VPP-shared names are rejected as ambiguous (flat writers
    overwrote them).
    """
    expected: dict[str, nn.Parameter] = {}
    ambiguous: set[str] = set()
    namespaced = any(isinstance(key, str) and key.startswith(_LEGACY_MODEL_CHUNK_PREFIX) for key in state_dict)
    if namespaced:
        for chunk_index, chunk in enumerate(model_chunks):
            for name, parameter in chunk.named_parameters():
                if include_parameter(name, parameter):
                    expected[f"{_LEGACY_MODEL_CHUNK_PREFIX}{chunk_index}.{name}"] = parameter
    else:
        candidates: dict[str, list[nn.Parameter]] = {}
        for chunk in model_chunks:
            for name, parameter in chunk.named_parameters():
                if include_parameter(name, parameter):
                    candidates.setdefault(name, []).append(parameter)
        for name, parameters in candidates.items():
            if len(parameters) == 1:
                expected[name] = parameters[0]
            else:
                ambiguous.add(name)

    state_names = set(state_dict)
    expected_names = set(expected)
    unexpected = sorted(str(name) for name in state_names - expected_names - ambiguous)
    missing = sorted(expected_names - state_names)
    assignments: list[tuple[str, nn.Parameter, torch.Tensor]] = []
    shape_mismatches = [
        f"{name}: unqualified legacy key is ambiguous across multiple model chunks" for name in sorted(ambiguous)
    ]
    for name in sorted(state_names & expected_names):
        tensor = state_dict[name]
        parameter = expected[name]
        if not isinstance(tensor, torch.Tensor):
            shape_mismatches.append(f"{name}: checkpoint value is {type(tensor).__name__}, expected a tensor")
            continue
        if tuple(tensor.shape) != tuple(parameter.shape):
            shape_mismatches.append(f"{name}: checkpoint {tuple(tensor.shape)} != parameter {tuple(parameter.shape)}")
            continue
        assignments.append((name, parameter, tensor))

    return AdapterLoadPlan(assignments, unexpected, missing, shape_mismatches)


def native_adapter_shard_name(tp_rank: int, pp_rank: int) -> str:
    """Return the legacy per-rank native adapter filename.

    Native adapter state is EP-invariant, so the expert-parallel rank is
    deliberately absent from the key: routed/grouped experts carry no native
    adapter (see ``miles_plugins.lora.spec.moe``), and the MoE shared expert is
    sharded over the *attention* tensor-parallel group, not the expert one
    (mcore ``transformer/moe/shared_experts.py`` builds it with
    ``tp_group=pg_collection.tp``). Every EP rank therefore holds byte-identical
    adapter state for a given ``(tp_rank, pp_rank)``.
    """
    return f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"


def native_adapter_load_plan(
    model_chunks: Sequence[nn.Module],
    state_dict: dict[str, torch.Tensor],
) -> AdapterLoadPlan:
    """Preflight a legacy native shard for exact target names and tensor shapes."""
    native_parameter_ids = {
        id(parameter) for adapter in iter_adapters(model_chunks) for parameter in adapter.parameters(recurse=False)
    }
    return adapter_load_plan(
        model_chunks,
        state_dict,
        lambda _name, parameter: id(parameter) in native_parameter_ids,
    )


def has_native_adapters(model_chunks: Sequence[nn.Module]) -> bool:
    """Whether the built-in provider marked this model-chunk collection.

    Concrete ``NativeLoRAAdapter`` classes are deliberately not enough: a
    custom provider may compose one with additional adapter types that the
    native checkpoint helpers cannot serialize. ``apply_native_lora`` marks every
    successful chunk, including legitimate PP/VPP chunks with zero local
    targets, so save and load share one explicit provider contract.
    """
    if not model_chunks:
        return False
    for chunk in model_chunks:
        module = chunk
        marked = False
        while True:
            if getattr(module, "_miles_native_lora_provider", False):
                marked = True
                break
            if not hasattr(module, "module"):
                break
            module = module.module
        if not marked:
            return False
    return True
