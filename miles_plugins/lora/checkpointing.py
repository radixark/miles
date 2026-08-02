"""Native-specific adapter checkpoint helpers.

Shared save/load orchestration and PP assembly live in
``miles.backends.megatron_utils.lora_utils``, still serving Megatron-Bridge too.

TODO:

- Revisit this split once the bridge refactor lands.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from miles_plugins.lora.modules.linear import iter_adapters

_MODEL_CHUNK_PREFIX = "_miles_model_chunks."


def model_chunk_state_key(chunk_index: int, parameter_name: str) -> str:
    """Return an unambiguous checkpoint key for a parameter in one model chunk."""
    return f"{_MODEL_CHUNK_PREFIX}{chunk_index}.{parameter_name}"


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


def adapter_state_dict(
    model_chunks: Sequence[nn.Module],
    include_parameter: Callable[[str, nn.Parameter], bool],
) -> dict[str, torch.Tensor]:
    """Collect selected parameters using model-chunk-qualified keys.

    Model chunks commonly expose identical parameter-tree names under virtual
    pipeline parallelism. Qualifying every key prevents later chunks from
    silently overwriting earlier ones in the Python dictionary.
    """
    state: dict[str, torch.Tensor] = {}
    for chunk_index, chunk in enumerate(model_chunks):
        for name, parameter in chunk.named_parameters():
            if include_parameter(name, parameter):
                state[model_chunk_state_key(chunk_index, name)] = parameter.detach().cpu()
    return state


def adapter_load_plan(
    model_chunks: Sequence[nn.Module],
    state_dict: dict[str, torch.Tensor],
    include_parameter: Callable[[str, nn.Parameter], bool],
) -> AdapterLoadPlan:
    """Preflight an adapter state dict without mutating model parameters.

    New checkpoints always qualify keys by model-chunk index. For backwards
    compatibility, unqualified legacy keys are accepted whenever they identify
    exactly one parameter across all current chunks. A key shared by multiple
    chunks is rejected as ambiguous: old VPP writers overwrote those values and
    there is no safe way to reconstruct which chunk the surviving tensor came
    from.
    """
    namespaced = any(isinstance(key, str) and key.startswith(_MODEL_CHUNK_PREFIX) for key in state_dict)

    expected: dict[str, nn.Parameter] = {}
    ambiguous_legacy: set[str] = set()
    if namespaced:
        for chunk_index, chunk in enumerate(model_chunks):
            for name, parameter in chunk.named_parameters():
                if include_parameter(name, parameter):
                    expected[model_chunk_state_key(chunk_index, name)] = parameter
    else:
        legacy_candidates: dict[str, list[nn.Parameter]] = {}
        for chunk in model_chunks:
            for name, parameter in chunk.named_parameters():
                if include_parameter(name, parameter):
                    legacy_candidates.setdefault(name, []).append(parameter)
        for name, candidates in legacy_candidates.items():
            if len(candidates) == 1:
                expected[name] = candidates[0]
            else:
                ambiguous_legacy.add(name)

    state_names = set(state_dict)
    expected_names = set(expected)
    unexpected = sorted(str(name) for name in state_names - expected_names - ambiguous_legacy)
    missing = sorted(expected_names - state_names)
    assignments: list[tuple[str, nn.Parameter, torch.Tensor]] = []
    shape_mismatches = [
        f"{name}: unqualified legacy key is ambiguous across multiple model chunks"
        for name in sorted(ambiguous_legacy)
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
    """Return the per-rank native adapter filename.

    Native adapter state is EP-invariant, so the expert-parallel rank is
    deliberately absent from the key: routed/grouped experts carry no native
    adapter (see ``miles_plugins.lora.spec.moe``), and the MoE shared expert is
    sharded over the *attention* tensor-parallel group, not the expert one
    (mcore ``transformer/moe/shared_experts.py`` builds it with
    ``tp_group=pg_collection.tp``). Every EP rank therefore holds byte-identical
    adapter state for a given ``(tp_rank, pp_rank)``.
    """
    return f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"


def native_adapter_state_dict(model_chunks: Sequence[nn.Module]) -> dict[str, torch.Tensor]:
    """Collect local native-LoRA parameters without model-chunk key collisions."""
    native_parameter_ids = {
        id(parameter) for adapter in iter_adapters(model_chunks) for parameter in adapter.parameters(recurse=False)
    }
    return adapter_state_dict(model_chunks, lambda _name, parameter: id(parameter) in native_parameter_ids)


def native_adapter_load_plan(
    model_chunks: Sequence[nn.Module],
    state_dict: dict[str, torch.Tensor],
) -> AdapterLoadPlan:
    """Preflight a native shard for exact target names and tensor shapes."""
    native_parameter_ids = {
        id(parameter) for adapter in iter_adapters(model_chunks) for parameter in adapter.parameters(recurse=False)
    }
    return adapter_load_plan(
        model_chunks,
        state_dict,
        lambda _name, parameter: id(parameter) in native_parameter_ids,
    )


def load_native_adapter_state_dict(
    model_chunks: Sequence[nn.Module],
    state_dict: dict[str, torch.Tensor],
) -> tuple[int, list[str], list[str]]:
    """Load a local native shard and report both target-set mismatch directions.

    Returns ``(loaded, unexpected, missing)``: *unexpected* are checkpoint
    tensors absent from the current exact target set, *missing* are current
    adapter parameters the checkpoint has no tensor for (they keep their fresh
    initialization). Either direction means the shard was saved for a different
    ``--target-modules`` set.
    """
    plan = native_adapter_load_plan(model_chunks, state_dict)
    if plan.shape_mismatches:
        details = "; ".join(plan.shape_mismatches[:8])
        raise ValueError(f"native adapter tensor shape mismatch: {details}")
    loaded = plan.apply()
    return loaded, plan.unexpected, plan.missing


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
