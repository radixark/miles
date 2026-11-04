"""The save / load flow, written once; backends only provide their checkpoint members."""

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from miles.backends.training_utils.checkpoint.adapter.adapter_exports import (
    has_plain_peft_form,
    write_peft_adapter,
    write_sampler_adapter,
)
from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    AdapterState,
    LocalShardSpec,
    _save_adapter_state,
    assert_adapter_state_keys,
    gloo_group,
    raise_if_any_rank_failed,
)
from miles.backends.training_utils.checkpoint.adapter.commit import (
    CommittedAdapterDir,
    UncommittedAdapterDir,
    commit_adapter_dir,
    delete_retired_adapter_dirs,
    write_adapter_dir,
)
from miles.backends.training_utils.checkpoint.adapter.key_mapping import (
    map_named_tensors_to_hf_keys,
    map_peft_keys_to_hf_keys,
    map_sampler_keys_to_hf_keys,
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
    collapse_shared_lora_a,
)
from miles.utils.lora.hf_lora_targets import HfLoraTargets

TRAINING_STATE_FILENAME = "training_state.json"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeadRows:
    """The head's lora_B, whose rows follow a vocabulary: a source may hold more rows than the run trains (an HF
    export over ``vocab_size``, a backend's padding), and the run keeps its first ones, as its exports trim the rest.
    """

    key: str  # the HF key adapter_state/ holds it under, e.g. "lm_head.lora_B"
    max_source_rows: int


class InModelLoader(Protocol):
    """A source a backend reads straight into its model, in the run's own parametrization.

    Every rank calls each step; ``load_adapter_dir`` agrees across ranks after each. A step that enters a collective
    agrees its own rank-local failures before it (``raise_if_any_rank_failed``), so one rank never fails alone while
    the others wait inside it.
    """

    # reads and checks the source without changing the model, so a refused source leaves the model as it was
    def parametrization(self, path: Path) -> AdapterParametrization: ...

    # every read and check left before the model is written; returns what apply needs
    def prepare(self, path: Path) -> Any: ...

    # writes the prepared source into the model
    def apply(self, prepared: Any) -> None: ...


class AdapterCheckpointMembers(Protocol):
    """The checkpoint members every backend's AdapterSlots implements; they never touch files."""

    # naming of export_bf16_adapter's names; picks the mapper to HF keys
    bf16_adapter_key_style: Literal["peft", "sampler"]
    # the trained head's lora_B, or None when the adapter has no head
    head_rows: HeadRows | None

    def export_bf16_adapter(self, slot: int) -> Iterable[tuple[str, torch.Tensor]]:
        """Collective; whole bf16 tensors on rank 0, nothing elsewhere."""

    def get_local_shard_specs(self, slot: int) -> list[LocalShardSpec]: ...

    def export_adapter_state(self, slot: int) -> AdapterState: ...

    # rank-local, no collectives: the library agrees across ranks after it, before finish_import_adapter_state
    def import_adapter_state(self, slot: int, state: AdapterState) -> None: ...

    # the collectives that complete an import (e.g. syncing params from the written masters); every rank calls it
    def finish_import_adapter_state(self, slot: int) -> None: ...


def save_adapter_dir(
    root: Path,
    name: str,
    members: AdapterCheckpointMembers,
    slot: int,
    parametrization: AdapterParametrization,
    *,
    tenant_digest: str | None,
    training_state: dict | None = None,
) -> UncommittedAdapterDir:
    """All ranks call; the caller's committer then runs ``commit_adapter_dir``."""
    assert parametrization.lora_alpha is not None, "lora_alpha must be resolved before an adapter dir is saved"
    with write_adapter_dir(root, name, tenant_digest=tenant_digest) as written:
        state, error = None, None
        try:
            state = members.export_adapter_state(slot)
        except Exception as exc:
            error = exc
        # agree before any collective, so one rank's failure cannot leave the others waiting in it
        raise_if_any_rank_failed(error)
        # DCP writes primary pieces only; replicas cannot stand in for a missing primary
        keys = _all_ranks_keys([s for s in state.shard_specs if s.is_primary])
        try:
            _assert_real_rank(state.shard_specs, parametrization.lora_rank)
            expected = parametrization.adapter_state_keys()
            assert keys == expected, (
                f"the backend's adapter state does not match the parametrization: "
                f"missing {sorted(expected - keys)[:8]}, unexpected {sorted(keys - expected)[:8]}"
            )
        except Exception as exc:
            error = exc
        raise_if_any_rank_failed(error)
        _save_adapter_state(written.path, state)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return written
        if training_state is not None:
            (written.path / TRAINING_STATE_FILENAME).write_text(json.dumps(training_state, indent=2))
        write_parametrization(written.path, parametrization)
    return written


def read_training_state(path: Path) -> dict | None:
    """The RL loop's iteration and scheduler state, if the source has them."""
    state_file = Path(path) / TRAINING_STATE_FILENAME
    return json.loads(state_file.read_text()) if state_file.is_file() else None


def _assert_real_rank(specs: list[LocalShardSpec], lora_rank: int) -> None:
    """adapter_state/ holds the real rank: A's rows and B's columns, never a multi-LoRA slot's padding."""
    for spec in specs:
        rank_axis = -2 if spec.key.endswith(".lora_A") else -1
        assert (
            spec.global_shape[rank_axis] == lora_rank
        ), f"{spec.key}: rank {spec.global_shape[rank_axis]} != {lora_rank}"


def _hf_key_mapper(members: AdapterCheckpointMembers, targets: HfLoraTargets):
    return {
        "peft": map_peft_keys_to_hf_keys,
        "sampler": partial(map_sampler_keys_to_hf_keys, targets=targets),
    }[members.bf16_adapter_key_style]


def _hf_keyed_bf16_adapter(
    named_tensors: list[tuple[str, torch.Tensor]],
    members: AdapterCheckpointMembers,
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
) -> dict[str, torch.Tensor]:
    weights, fused_groups = map_named_tensors_to_hf_keys(named_tensors, _hf_key_mapper(members, targets), targets)
    # the export names a shared A once per module; the writers expect it once per group
    return collapse_shared_lora_a(weights, parametrization, fused_groups)


def _stacked_expert_keys(named_tensors: list[tuple[str, torch.Tensor]], map_name) -> frozenset[str]:
    """The HF keys the export names as one stacked experts tensor: the base checkpoint's packed form. A shared-outer
    side also comes unindexed, as ``[1, ...]``; it is one tensor for every expert, not a stack, so it is left out."""
    keys = set()
    for name, tensor in named_tensors:
        hf_key = map_name(name)
        if hf_key.expert_index is None and ".experts." in hf_key.key and tensor.dim() == 3 and tensor.shape[0] > 1:
            keys.add(hf_key.key)
    return frozenset(keys)


def _is_writer() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def write_peft_export(
    directory: Path,
    members: AdapterCheckpointMembers,
    slot: int,
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
) -> None:
    """All ranks call; rank 0 writes ``slot``'s bf16 PEFT export into ``directory``."""
    _write_bf16_export(
        directory,
        members,
        slot,
        parametrization,
        targets,
        lambda *view, stacked_keys: write_peft_adapter(*view, stacked_keys=stacked_keys),
    )


def write_sampler_view(
    directory: Path,
    members: AdapterCheckpointMembers,
    slot: int,
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
) -> None:
    """All ranks call; rank 0 writes ``slot``'s bf16 sampler export into ``directory``, outside any commit."""
    # the sampler layout keeps every expert stacked
    _write_bf16_export(
        directory, members, slot, parametrization, targets, lambda *view, stacked_keys: write_sampler_adapter(*view)
    )


def _write_bf16_export(
    directory: Path,
    members: AdapterCheckpointMembers,
    slot: int,
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
    write: Callable[..., None],
) -> None:
    error = None
    try:
        # the gathered adapter leaves the GPU before rank 0 splits and copies it
        named_tensors = [(key, tensor.cpu()) for key, tensor in members.export_bf16_adapter(slot)]
        if _is_writer():
            weights = _hf_keyed_bf16_adapter(named_tensors, members, parametrization, targets)
            # Bridge names experts as its checkpoint stores them; native always stacks them, so the family decides
            stacked_keys = (
                _stacked_expert_keys(named_tensors, _hf_key_mapper(members, targets))
                if members.bf16_adapter_key_style == "peft"
                else None
            )
            write(directory, weights, parametrization, targets, stacked_keys=stacked_keys)
    except Exception as exc:
        error = exc
    raise_if_any_rank_failed(error)


def save_and_commit_adapter_dir(
    root: Path,
    name: str,
    members: AdapterCheckpointMembers,
    slot: int,
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
    *,
    training_state: dict | None = None,
) -> CommittedAdapterDir:
    """RL's ``--save``: all ranks write, rank 0 commits over any earlier save of ``name``, then the PEFT export.

    A failed PEFT export fails every rank but keeps the committed checkpoint; its view can be rebuilt offline. A
    parametrization with no plain-PEFT form gets no PEFT view: the adapter dir is the checkpoint.
    """
    written = save_adapter_dir(
        root, name, members, slot, parametrization, tenant_digest=None, training_state=training_state
    )
    error = None
    if _is_writer():
        try:
            commit_adapter_dir(root, name, written, overwrite=True)
        except Exception as exc:
            error = exc
    raise_if_any_rank_failed(error)
    if has_plain_peft_form(parametrization, targets):
        write_peft_export(written.path, members, slot, parametrization, targets)
    elif _is_writer():
        logger.info(f"{written.path}: no PEFT view, as this parametrization has no plain-PEFT form")
    if _is_writer():
        delete_retired_adapter_dirs(root)
    return CommittedAdapterDir(written.path, written.commit_id)


def write_sampler_export(
    root: Path,
    name: str,
    members: AdapterCheckpointMembers,
    slot: int,
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
    *,
    tenant_digest: str | None,
) -> UncommittedAdapterDir:
    """All ranks call; rank 0 writes ``slot``'s bf16 sampler export. The caller's committer then commits it."""
    with write_adapter_dir(root, name, tenant_digest=tenant_digest) as written:
        # the gathered adapter leaves the GPU before rank 0 splits and copies it
        named_tensors = [(key, tensor.cpu()) for key, tensor in members.export_bf16_adapter(slot)]
        if _is_writer():
            weights = _hf_keyed_bf16_adapter(named_tensors, members, parametrization, targets)
            write_sampler_adapter(written.path, weights, parametrization, targets)
    return written


def _slice(full: torch.Tensor, spec: LocalShardSpec) -> torch.Tensor:
    assert tuple(full.shape) == spec.global_shape, f"{spec.key}: source {tuple(full.shape)} != run {spec.global_shape}"
    return full[tuple(slice(o, o + n) for o, n in zip(spec.offsets, spec.local_shape, strict=True))].clone()


def _with_zero_moments(specs: list[LocalShardSpec], pieces: list[torch.Tensor]) -> AdapterState:
    return AdapterState(
        specs, pieces, [torch.zeros_like(p) for p in pieces], [torch.zeros_like(p) for p in pieces], step=0
    )


def _assert_has_keys(weights: dict[str, torch.Tensor], keys: set[str]) -> None:
    missing = keys - set(weights)
    assert not missing, f"the source lacks tensors the run trains: {sorted(missing)}"


def _weights_only_state(
    weights: dict[str, torch.Tensor], specs: list[LocalShardSpec], head_rows: HeadRows | None
) -> AdapterState:
    _assert_has_keys(weights, {s.key for s in specs})
    return _with_zero_moments(specs, [_slice(_served_rows(weights[s.key], s, head_rows), s).float() for s in specs])


def _served_rows(full: torch.Tensor, spec: LocalShardSpec, head_rows: HeadRows | None) -> torch.Tensor:
    if head_rows is None or spec.key != head_rows.key:
        return full
    rows, served = full.shape[0], spec.global_shape[0]
    assert (
        served <= rows <= head_rows.max_source_rows
    ), f"{spec.key}: the source has {rows} rows; this run serves {served} and reads up to {head_rows.max_source_rows}"
    return full[:served]


def _with_run_alpha_if_unknown(source: AdapterParametrization, run: AdapterParametrization) -> AdapterParametrization:
    if source.lora_alpha is not None:
        return source
    # a source that records no alpha is taken to use the run's
    return replace(source, lora_alpha=run.lora_alpha)


def _open_source(
    path: Path, kind: AdapterSourceKind, run: AdapterParametrization, targets: HfLoraTargets, *, load_optimizer: bool
):
    """-> (reader, source parametrization, conversion steps), asserted loadable into ``run``."""
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
    return reader, source, steps


def _reads_adapter_state_pieces(reader, steps) -> bool:
    return isinstance(reader, MilesAdapterDirReader) and reader.has_adapter_state() and not steps


def check_adapter_source(
    path: Path, run: AdapterParametrization, targets: HfLoraTargets, *, load_optimizer: bool
) -> AdapterSourceKind:
    """Rank-local dry run of every check ``load_adapter_dir`` makes on the files of a source it reads itself.

    Raises what that load would; an in-model source is returned unchecked, its loader's checks are the backend's.
    """
    kind = classify_adapter_source(path, targets, load_optimizer=load_optimizer)
    if kind.is_read_into_model():
        return kind
    reader, _, steps = _open_source(path, kind, run, targets, load_optimizer=load_optimizer)
    if _reads_adapter_state_pieces(reader, steps):
        assert_adapter_state_keys(path, run.adapter_state_keys(), with_optimizer=load_optimizer)
    else:
        _assert_has_keys(apply_parametrization_conversion(reader.read_weights(), steps), run.adapter_state_keys())
    return kind


def load_adapter_dir(
    path: Path,
    members: AdapterCheckpointMembers,
    slot: int,
    run: AdapterParametrization,
    targets: HfLoraTargets,
    *,
    load_optimizer: bool,
    in_model_loaders_by_kind: Mapping[AdapterSourceKind, InModelLoader] | None = None,
) -> AdapterParametrization:
    """Load any recognized source into ``slot``; returns the source parametrization.

    All ranks call. A failure on one rank fails every rank before the next collective, provided a backend's in-model
    loader agrees its own failures before each collective it enters (see ``InModelLoader``).
    Alpha is not checked here: the caller owns it (Tinker adopts the source's, RL compares it to ``--lora-alpha``).

    A source that needs conversion (PEFT, external weights, another parametrization) is read whole on every rank:
    about 6 bytes of host memory per source element for a bf16 file (its tensors and their fp32 upcast), e.g.
    ~41 GB per rank and ~330 GB per 8-rank node for a rank-16 adapter on every DeepSeek-V3 expert.
    """
    kind, error = None, None
    try:
        kind = classify_adapter_source(path, targets, load_optimizer=load_optimizer)
        assert not kind.is_read_into_model() or (
            in_model_loaders_by_kind is not None and kind in in_model_loaders_by_kind
        ), f"{path} is a {kind.value} source, which this backend cannot load"
    except Exception as exc:
        error = exc
    raise_if_any_rank_failed(error)
    if kind.is_read_into_model():
        loader, source = in_model_loaders_by_kind[kind], None
        try:
            source = _with_run_alpha_if_unknown(loader.parametrization(path), run)
            steps = plan_parametrization_conversion(source, run, source_dtype=torch.float32)
            # an in-model loader reads straight into the run's parametrization; it cannot convert
            assert not steps, f"{path} needs {steps} to match the run, which an in-model loader cannot do"
            assert_parametrization_matches_run(
                source, run, steps, load_optimizer=load_optimizer, has_optimizer_state=True
            )
        except Exception as exc:
            error = exc
        raise_if_any_rank_failed(error)
        prepared = None
        try:
            prepared = loader.prepare(path)
        except Exception as exc:
            error = exc
        raise_if_any_rank_failed(error)
        try:
            loader.apply(prepared)
        except Exception as exc:
            error = exc
        raise_if_any_rank_failed(error)
        return source

    specs, reader, source, steps = None, None, None, None
    try:
        specs = members.get_local_shard_specs(slot)
        reader, source, steps = _open_source(path, kind, run, targets, load_optimizer=load_optimizer)
    except Exception as exc:
        error = exc
    raise_if_any_rank_failed(error)

    if _reads_adapter_state_pieces(reader, steps):
        run_keys = _all_ranks_keys(specs)
        try:
            assert_adapter_state_keys(path, run_keys, with_optimizer=load_optimizer)
        except Exception as exc:
            error = exc
        raise_if_any_rank_failed(error)
        # same parametrization: every rank reads only its own pieces, and moments only when it loads them
        weights = [torch.empty(s.local_shape, dtype=torch.float32) for s in specs]
        moments = (
            [[torch.empty(s.local_shape, dtype=torch.float32) for s in specs] for _ in range(2)]
            if load_optimizer
            else [[], []]
        )
        state = AdapterState(specs, weights, *moments, step=None, has_optimizer_state=load_optimizer)
        state.step = reader.load_state(state)
        if not load_optimizer:
            state = _with_zero_moments(specs, state.weights)
    else:
        state = None
        try:
            weights = apply_parametrization_conversion(reader.read_weights(), steps)
            state = _weights_only_state(weights, specs, members.head_rows)
        except Exception as exc:
            error = exc
        raise_if_any_rank_failed(error)
    try:
        members.import_adapter_state(slot, state)
    except Exception as exc:
        error = exc
    raise_if_any_rank_failed(error)
    try:
        members.finish_import_adapter_state(slot)
    except Exception as exc:
        error = exc
    raise_if_any_rank_failed(error)
    return source


def _all_ranks_keys(specs: list[LocalShardSpec]) -> set[str]:
    keys = [{s.key for s in specs}]
    if torch.distributed.is_initialized():
        keys = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(keys, {s.key for s in specs}, group=gloo_group())
    return set().union(*keys)
