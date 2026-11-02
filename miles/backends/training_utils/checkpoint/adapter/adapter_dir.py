"""The save / load flow, written once; backends only provide their checkpoint members."""

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import torch

from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    AdapterState,
    LocalShardSpec,
    save_adapter_state,
)
from miles.backends.training_utils.checkpoint.adapter.commit import (
    UncommittedAdapterDir,
    raise_if_any_rank_failed,
    write_adapter_dir,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    AdapterParametrization,
    write_parametrization,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization_conversion import (
    apply_parametrization_conversion,
    assert_parametrization_matches_run,
    plan_parametrization_conversion,
)
from miles.backends.training_utils.checkpoint.adapter.readers import (
    AdapterSourceKind,
    ExternalWeightsReader,
    MilesAdapterDirReader,
    PeftAdapterReader,
    classify_adapter_source,
)
from miles.utils.lora.hf_lora_targets import HfLoraTargets

TRAINING_STATE_FILENAME = "training_state.json"


class AdapterCheckpointMembers(Protocol):
    """The four checkpoint members of a backend's AdapterSlots (design section 4)."""

    def export_bf16_adapter(self, slot: int) -> Iterable[tuple[str, torch.Tensor]]: ...

    def get_local_shard_specs(self, slot: int) -> list[LocalShardSpec]: ...

    def export_adapter_state(self, slot: int) -> AdapterState: ...

    def import_adapter_state(self, slot: int, state: AdapterState) -> None: ...


def save_adapter_dir(
    root: Path,
    name: str,
    members: AdapterCheckpointMembers,
    slot: int,
    parametrization: AdapterParametrization,
    *,
    tenant: str | None = None,
    training_state: dict | None = None,
) -> UncommittedAdapterDir:
    """All ranks call; the caller's committer then runs ``commit_adapter_dir``."""
    with write_adapter_dir(root, name, tenant=tenant) as written:
        state, export_error = None, None
        try:
            state = members.export_adapter_state(slot)
        except Exception as exc:
            export_error = exc
        # agree before the collective DCP write, so one rank's failure cannot leave the others waiting in it
        raise_if_any_rank_failed(export_error)
        save_adapter_state(written.path, state)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return written
        if training_state is not None:
            (written.path / TRAINING_STATE_FILENAME).write_text(json.dumps(training_state, indent=2))
        write_parametrization(written.path, parametrization)
    return written


def _slice(full: torch.Tensor, spec: LocalShardSpec) -> torch.Tensor:
    assert tuple(full.shape) == spec.global_shape, f"{spec.key}: source {tuple(full.shape)} != run {spec.global_shape}"
    return full[tuple(slice(o, o + n) for o, n in zip(spec.offsets, spec.local_shape, strict=True))].clone()


def _weights_only(specs: list[LocalShardSpec], pieces: list[torch.Tensor]) -> AdapterState:
    return AdapterState(
        specs, pieces, [torch.zeros_like(p) for p in pieces], [torch.zeros_like(p) for p in pieces], step=0
    )


def _weights_only_state(weights: dict[str, torch.Tensor], specs: list[LocalShardSpec]) -> AdapterState:
    missing = {s.key for s in specs} - set(weights)
    assert not missing, f"the source lacks tensors the run trains: {sorted(missing)}"
    return _weights_only(specs, [_slice(weights[s.key], s).float() for s in specs])


def _with_run_alpha_if_unknown(source: AdapterParametrization, run: AdapterParametrization) -> AdapterParametrization:
    if source.lora_alpha is not None:
        return source
    # the run's --lora-alpha was required for such a source (resolved in miles_validate_args)
    return replace(source, lora_alpha=run.lora_alpha)


def load_adapter_dir(
    path: Path,
    members: AdapterCheckpointMembers,
    slot: int,
    run: AdapterParametrization,
    targets: HfLoraTargets,
    *,
    load_optimizer: bool,
    in_model_loaders_by_kind: Mapping[AdapterSourceKind, Callable[[Path], AdapterParametrization]] | None = None,
) -> AdapterParametrization:
    """Load any recognized source into ``slot``; returns the source parametrization.

    Alpha is not checked here: the caller owns it (Tinker adopts the source's, RL compares it to ``--lora-alpha``).
    """
    kind = classify_adapter_source(path, targets, load_optimizer=load_optimizer)
    if kind.is_read_into_model():
        assert (
            in_model_loaders_by_kind is not None and kind in in_model_loaders_by_kind
        ), f"{path} is a {kind.value} source, which this backend cannot load"
        source = _with_run_alpha_if_unknown(in_model_loaders_by_kind[kind](path), run)
        assert_parametrization_matches_run(
            source, run, [], load_optimizer=load_optimizer, has_optimizer_state=load_optimizer
        )
        return source
    if kind == AdapterSourceKind.ADAPTER_DIR:
        reader = MilesAdapterDirReader(path, targets)
    elif kind == AdapterSourceKind.EXTERNAL_WEIGHTS:
        reader = ExternalWeightsReader(path, run, targets)
    else:
        reader = PeftAdapterReader(path, run, targets)
    source = _with_run_alpha_if_unknown(reader.source.parametrization, run)
    steps = plan_parametrization_conversion(source, run, source_dtype=reader.source.weight_dtype)
    assert_parametrization_matches_run(
        source, run, steps, load_optimizer=load_optimizer, has_optimizer_state=reader.source.has_optimizer_state
    )
    specs = members.get_local_shard_specs(slot)
    if isinstance(reader, MilesAdapterDirReader) and reader.has_adapter_state() and not steps:
        # same parametrization: every rank reads only its own pieces, and moments only when it loads them
        template = AdapterState(
            specs,
            *[[torch.empty(s.local_shape) for s in specs] for _ in range(3)],
            step=None,
            has_optimizer_state=load_optimizer,
        )
        template.step = reader.load_state(template)
        if not load_optimizer:
            template = _weights_only(specs, template.weights)
        members.import_adapter_state(slot, template)
    else:
        num_experts = next(
            (s.global_shape[0] for s in specs if ".experts." in s.key and len(s.global_shape) == 3), None
        )
        weights = apply_parametrization_conversion(reader.read_weights(), steps, num_experts=num_experts)
        members.import_adapter_state(slot, _weights_only_state(weights, specs))
    return source
