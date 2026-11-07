"""The save / load flow, written once; backends only provide their checkpoint members."""

import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from miles.backends.training_utils.checkpoint.adapter.adapter_exports import (
    has_plain_peft_form,
    write_external_adapter,
    write_peft_adapter,
    write_sampler_adapter,
)
from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    AdapterState,
    LocalShardSpec,
    _save_adapter_state,
    agree_on_failure_context,
    assert_adapter_state_keys,
    broadcast_from_rank0,
    gloo_group,
    is_rank0,
    is_vocab_padding,
    read_adapter_state_shapes,
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
from miles.backends.training_utils.checkpoint.adapter.source_error import AdapterSourceError
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

    Every rank calls each step; ``load_adapter_source`` agrees across ranks after each. A step that enters a collective
    agrees its own rank-local failures before it (``agree_on_failure_context``), so one rank never fails alone while
    the others wait inside it.
    """

    # reads and checks the source without changing the model, so a refused source leaves the model as it was;
    # returns the source's parametrization and whether it holds optimizer state
    def parametrization(self, path: Path) -> tuple[AdapterParametrization, bool]: ...

    # every read and check left before the model is written; returns what apply needs
    def prepare(self, path: Path) -> Any: ...

    # writes the prepared source into the model
    def apply(self, prepared: Any) -> None: ...


class AdapterCheckpointMembers(Protocol):
    """The checkpoint members every backend's AdapterSlots implements; they never touch files.

    The library agrees across ranks after each member it calls. A member that enters a collective agrees its own
    rank-local failures before it (``agree_on_failure_context``), so one rank never fails alone while the others wait
    inside it.
    """

    # naming of export_bf16_adapter's names; picks the mapper to HF keys
    bf16_adapter_key_style: Literal["peft", "sampler"]
    # the trained head's lora_B, or None when the adapter has no head
    head_rows: HeadRows | None

    def export_bf16_adapter(self, slot: int) -> Iterable[tuple[str, torch.Tensor]]:
        """Collective; whole tensors on rank 0, nothing elsewhere. Any float dtype; the exports cast to bf16."""

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
        # agree before any collective, so one rank's failure cannot leave the others waiting in it
        with agree_on_failure_context():
            state = members.export_adapter_state(slot)
        # DCP writes primary pieces only; replicas cannot stand in for a missing primary
        keys = _all_ranks_keys([s for s in state.shard_specs if s.is_primary and not is_vocab_padding(s.key)])
        with agree_on_failure_context():
            _assert_real_rank(state.shard_specs, parametrization.lora_rank)
            expected = parametrization.adapter_state_keys()
            assert keys == expected, (
                f"the backend's adapter state does not match the parametrization: "
                f"missing {sorted(expected - keys)[:8]}, unexpected {sorted(keys - expected)[:8]}"
            )
        _save_adapter_state(written.path, state)
        if not is_rank0():
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


def write_peft_export(
    directory: Path,
    members: AdapterCheckpointMembers,
    slot: int,
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
    *,
    lora_dropout: float,
) -> None:
    """All ranks call; rank 0 writes ``slot``'s bf16 PEFT export into ``directory``."""
    _write_bf16_export(
        directory,
        members,
        slot,
        parametrization,
        targets,
        lambda *view, stacked_keys: write_peft_adapter(*view, stacked_keys=stacked_keys, lora_dropout=lora_dropout),
    )


def write_sampler_export(
    directory: Path,
    members: AdapterCheckpointMembers,
    slot: int,
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
    *,
    lora_dropout: float,
) -> None:
    """All ranks call; rank 0 writes ``slot``'s bf16 sampler export into ``directory``, outside any commit."""
    # the sampler layout keeps every expert stacked
    _write_bf16_export(
        directory,
        members,
        slot,
        parametrization,
        targets,
        lambda *view, stacked_keys: write_sampler_adapter(*view, lora_dropout=lora_dropout),
    )


def _write_bf16_export(
    directory: Path,
    members: AdapterCheckpointMembers,
    slot: int,
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
    write: Callable[..., None],
) -> None:
    with agree_on_failure_context():
        # the gathered adapter leaves the GPU before rank 0 splits and copies it
        named_tensors = [(key, tensor.cpu()) for key, tensor in members.export_bf16_adapter(slot)]
        if is_rank0():
            weights = _hf_keyed_bf16_adapter(named_tensors, members, parametrization, targets)
            # Bridge names experts as its checkpoint stores them; native always stacks them, so the family decides
            stacked_keys = (
                _stacked_expert_keys(named_tensors, _hf_key_mapper(members, targets))
                if members.bf16_adapter_key_style == "peft"
                else None
            )
            write(directory, weights, parametrization, targets, stacked_keys=stacked_keys)


def save_and_commit_adapter_dir(
    root: Path,
    name: str,
    members: AdapterCheckpointMembers,
    slot: int,
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
    *,
    training_state: dict | None = None,
    lora_dropout: float,
) -> CommittedAdapterDir:
    """RL's ``--save``: all ranks write, rank 0 commits over any earlier save of ``name``, then the PEFT export.

    A failed PEFT export fails every rank but keeps the committed checkpoint; its view can be rebuilt offline. A
    parametrization with no plain-PEFT form gets no PEFT view: the adapter dir is the checkpoint.
    """
    written = save_adapter_dir(
        root, name, members, slot, parametrization, tenant_digest=None, training_state=training_state
    )
    with agree_on_failure_context():
        if is_rank0():
            commit_adapter_dir(root, name, written, overwrite=True)
    if has_plain_peft_form(parametrization, targets):
        write_peft_export(written.path, members, slot, parametrization, targets, lora_dropout=lora_dropout)
    elif is_rank0():
        logger.info(f"{written.path}: no PEFT view, as this parametrization has no plain-PEFT form")
    if is_rank0():
        delete_retired_adapter_dirs(root)
    return CommittedAdapterDir(written.path, written.commit_id)


def save_sampler_export(
    root: Path,
    name: str,
    members: AdapterCheckpointMembers,
    slot: int,
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
    *,
    tenant_digest: str | None,
    lora_dropout: float,
) -> UncommittedAdapterDir:
    """All ranks call; rank 0 writes ``slot``'s bf16 sampler export. The caller's committer then commits it."""
    with write_adapter_dir(root, name, tenant_digest=tenant_digest) as written:
        write_sampler_export(written.path, members, slot, parametrization, targets, lora_dropout=lora_dropout)
    return written


def save_external_export(
    root: Path,
    name: str,
    members: AdapterCheckpointMembers,
    slot: int,
    parametrization: AdapterParametrization,
    targets: HfLoraTargets,
    *,
    tenant_digest: str | None,
    age_encryption_recipients: Sequence[str] = (),
    lora_dropout: float,
) -> UncommittedAdapterDir:
    """All ranks call; rank 0 writes ``slot``'s external export from the fp32 masters (no parametrization: it is a
    download, not a checkpoint). The caller's committer then commits it."""
    with write_adapter_dir(root, name, tenant_digest=tenant_digest) as written:
        # agree before the gather, so one rank's failure cannot leave the others waiting in it
        with agree_on_failure_context():
            state = members.export_adapter_state(slot)
        weights = _gather_whole_weights(state)
        # rank 0's own pieces are in ``weights`` now; the export would otherwise hold them twice
        del state
        if is_rank0():
            keys, expected = set(weights), parametrization.adapter_state_keys()
            assert keys == expected, (
                f"the backend's adapter state does not match the parametrization: "
                f"missing {sorted(expected - keys)[:8]}, unexpected {sorted(keys - expected)[:8]}"
            )
            write_external_adapter(
                written.path,
                weights,
                parametrization,
                targets,
                age_encryption_recipients=age_encryption_recipients,
                lora_dropout=lora_dropout,
            )
    return written


def _gather_whole_weights(state: AdapterState) -> dict[str, torch.Tensor] | None:
    """Collective: each key's whole fp32 weight, assembled on rank 0 from the primary pieces; None elsewhere."""
    pieces = [(s, w.cpu()) for s, w in zip(state.shard_specs, state.weights, strict=True) if s.is_primary]
    specs_by_rank = [[spec for spec, _ in pieces]]
    if torch.distributed.is_initialized():
        # only the specs go through gather_object, which pads every rank's payload to the largest one
        specs_by_rank = [None] * torch.distributed.get_world_size() if is_rank0() else None
        torch.distributed.gather_object([spec for spec, _ in pieces], specs_by_rank, dst=0, group=gloo_group())
    if not is_rank0():
        for _, piece in pieces:
            torch.distributed.send(piece.float().contiguous(), dst=0, group=gloo_group())
        return None
    whole, filled = {}, {}
    # pieces arrive one at a time, so rank 0 holds the whole adapter plus one received piece
    for rank, specs in enumerate(specs_by_rank):
        for index, spec in enumerate(specs):
            if rank == 0:
                piece = pieces[index][1]
            else:
                piece = torch.empty(spec.local_shape, dtype=torch.float32)
                torch.distributed.recv(piece, src=rank, group=gloo_group())
            full = whole.setdefault(spec.key, torch.empty(spec.global_shape, dtype=torch.float32))
            full[tuple(slice(o, o + n) for o, n in zip(spec.offsets, spec.local_shape, strict=True))] = piece
            filled[spec.key] = filled.get(spec.key, 0) + piece.numel()
    # primary pieces never overlap, so a tensor is whole exactly when they cover all of its elements
    partial_keys = sorted(key for key, tensor in whole.items() if filled[key] != tensor.numel())
    assert not partial_keys, f"the primary pieces do not cover {partial_keys[:8]}"
    return whole


def _assert_source_shape(full: torch.Tensor, spec: LocalShardSpec) -> None:
    if tuple(full.shape) != spec.global_shape:
        raise AdapterSourceError(f"{spec.key}: source {tuple(full.shape)} != run {spec.global_shape}")


def _slice(full: torch.Tensor, spec: LocalShardSpec) -> torch.Tensor:
    _assert_source_shape(full, spec)
    return full[tuple(slice(o, o + n) for o, n in zip(spec.offsets, spec.local_shape, strict=True))].clone()


def _with_zero_moments(specs: list[LocalShardSpec], pieces: list[torch.Tensor]) -> AdapterState:
    return AdapterState(
        specs, pieces, [torch.zeros_like(p) for p in pieces], [torch.zeros_like(p) for p in pieces], step=0
    )


def _assert_has_keys(weights: dict[str, torch.Tensor], keys: set[str]) -> None:
    missing = keys - set(weights)
    if missing:
        raise AdapterSourceError(f"the source lacks tensors the run trains: {sorted(missing)}")


def _zero_filled(specs: list[LocalShardSpec], loaded: AdapterState) -> AdapterState:
    """``loaded`` (pieces of ``specs``) with zeros for the pieces it left unread, and zero moments if it read none."""
    index = {id(s): i for i, s in enumerate(loaded.shard_specs)}

    def aligned(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        return [tensors[index[id(s)]] if id(s) in index else torch.zeros(s.local_shape) for s in specs]

    if not loaded.has_optimizer_state:
        return _with_zero_moments(specs, aligned(loaded.weights))
    return AdapterState(
        specs, aligned(loaded.weights), aligned(loaded.exp_avgs), aligned(loaded.exp_avg_sqs), step=loaded.step
    )


def _assert_pieces_readable(
    weights: dict[str, torch.Tensor], specs: list[LocalShardSpec], head_rows: HeadRows | None
) -> None:
    _assert_has_keys(weights, {s.key for s in specs if not is_vocab_padding(s.key)})
    for s in specs:
        if not is_vocab_padding(s.key):
            _assert_source_shape(_served_rows(weights[s.key], s, head_rows), s)


def _scatter_weights_only_state(
    weights: dict[str, torch.Tensor] | None, specs: list[LocalShardSpec], head_rows: HeadRows | None
) -> AdapterState:
    """Collective: each rank's pieces of the source rank 0 read and converted, sent to it point-to-point."""
    specs_by_rank = [specs]
    if torch.distributed.is_initialized():
        # only the specs go through gather_object, which pads every rank's payload to the largest one
        specs_by_rank = [None] * torch.distributed.get_world_size() if is_rank0() else None
        torch.distributed.gather_object(specs, specs_by_rank, dst=0, group=gloo_group())
    with agree_on_failure_context():
        if is_rank0():
            for rank_specs in specs_by_rank:
                _assert_pieces_readable(weights, rank_specs, head_rows)
    # no export or conversion carries vocab padding: it stays zero, as at init
    if not is_rank0():
        pieces = [torch.zeros(s.local_shape) for s in specs]
        for spec, piece in zip(specs, pieces, strict=True):
            if not is_vocab_padding(spec.key):
                torch.distributed.recv(piece, src=0, group=gloo_group())
        return _with_zero_moments(specs, pieces)
    own = []
    for rank, rank_specs in enumerate(specs_by_rank):
        for spec in rank_specs:
            if is_vocab_padding(spec.key):
                piece = torch.zeros(spec.local_shape)
            else:
                piece = _slice(_served_rows(weights[spec.key], spec, head_rows), spec).float().contiguous()
                if rank != 0:
                    torch.distributed.send(piece, dst=rank, group=gloo_group())
            if rank == 0:
                own.append(piece)
    return _with_zero_moments(specs, own)


def _served_rows(full: torch.Tensor, spec: LocalShardSpec, head_rows: HeadRows | None) -> torch.Tensor:
    if head_rows is None or spec.key != head_rows.key:
        return full
    _assert_head_rows(spec.key, full.shape[0], spec.global_shape[0], head_rows)
    return full[: spec.global_shape[0]]


def _assert_head_rows(key: str, rows: int, served: int, head_rows: HeadRows) -> None:
    if not served <= rows <= head_rows.max_source_rows:
        raise AdapterSourceError(
            f"{key}: the source has {rows} rows; this run serves {served} and reads up to {head_rows.max_source_rows}"
        )


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


@dataclass(frozen=True)
class CheckedSource:
    """What ``check_adapter_source`` opened; ``load_adapter_source`` takes it instead of reading the source again."""

    kind: AdapterSourceKind
    reader: Any = None
    source: AdapterParametrization | None = None
    steps: list | None = None
    # the converted tensors of a source read as weights
    weights: dict[str, torch.Tensor] | None = None


def check_adapter_source(
    path: Path,
    run: AdapterParametrization,
    targets: HfLoraTargets,
    *,
    load_optimizer: bool,
    run_shapes: dict[str, tuple[int, ...]] | None = None,
    head_rows: HeadRows | None = None,
) -> CheckedSource:
    """Rank-local dry run of every check ``load_adapter_source`` makes on the files of a source it reads itself.

    Raises what that load would; an in-model source is returned unchecked, its loader's checks are the backend's.
    ``run_shapes``: the whole shape of each HF-keyed tensor this rank loads, checked on a source read as weights;
    ``head_rows``: the members' head, whose source may hold more rows than the run serves, as the load reads it.
    """
    kind = classify_adapter_source(path, targets, load_optimizer=load_optimizer)
    if kind.is_read_into_model():
        return CheckedSource(kind)
    reader, source, steps = _open_source(path, kind, run, targets, load_optimizer=load_optimizer)
    if _reads_adapter_state_pieces(reader, steps):
        assert_adapter_state_keys(path, run.adapter_state_keys(), with_optimizer=load_optimizer)
        return CheckedSource(kind, reader, source, steps)
    weights = apply_parametrization_conversion(reader.read_weights(), steps)
    _assert_has_keys(weights, run.adapter_state_keys())
    if run_shapes is not None:
        if head_rows is not None and head_rows.key in run_shapes:
            source_rows, run_shape = weights[head_rows.key].shape[0], run_shapes[head_rows.key]
            _assert_head_rows(head_rows.key, source_rows, run_shape[0], head_rows)
            run_shapes = {**run_shapes, head_rows.key: (source_rows, *run_shape[1:])}
        differing = [
            f"{k}: {tuple(weights[k].shape)} != {s}" for k, s in run_shapes.items() if tuple(weights[k].shape) != s
        ]
        if differing:
            raise AdapterSourceError(f"tensor shapes differ from this model's, e.g. {differing[:3]}")
    return CheckedSource(kind, reader, source, steps, weights)


def load_adapter_source(
    path: Path,
    members: AdapterCheckpointMembers,
    slot: int,
    run: AdapterParametrization,
    targets: HfLoraTargets,
    *,
    load_optimizer: bool,
    in_model_loaders_by_kind: Mapping[AdapterSourceKind, InModelLoader] | None = None,
    checked: CheckedSource | None = None,
) -> AdapterParametrization:
    """Load any recognized source into ``slot``; returns the source parametrization.

    ``checked``: rank 0's ``check_adapter_source`` of the same source and run, read instead of the source.

    All ranks call. A failure on one rank fails every rank before the next collective, provided a backend's in-model
    loader agrees its own failures before each collective it enters (see ``InModelLoader``).
    Alpha is not checked here: the caller owns it (Tinker adopts the source's, RL compares it to ``--lora-alpha``).

    A source that needs conversion (PEFT, external weights, another parametrization) is read and converted once, on
    rank 0, which sends every rank its pieces: rank 0 holds about 6 bytes per source element for a bf16 file (its
    tensors and their fp32 upcast), e.g. ~41 GB for a rank-16 adapter on every DeepSeek-V3 expert.
    """
    with agree_on_failure_context():
        kind = classify_adapter_source(path, targets, load_optimizer=load_optimizer)
        if kind.is_read_into_model() and (in_model_loaders_by_kind is None or kind not in in_model_loaders_by_kind):
            raise AdapterSourceError(f"{path} is a {kind.value} source, which this backend cannot load")
    if kind.is_read_into_model():
        loader = in_model_loaders_by_kind[kind]
        with agree_on_failure_context():
            source, has_optimizer_state = loader.parametrization(path)
            source = _with_run_alpha_if_unknown(source, run)
            steps = plan_parametrization_conversion(source, run, source_dtype=torch.float32)
            # an in-model loader reads straight into the run's parametrization; it cannot convert
            if steps:
                raise AdapterSourceError(f"{path} needs {steps} to match the run, which an in-model loader cannot do")
            assert_parametrization_matches_run(
                source, run, steps, load_optimizer=load_optimizer, has_optimizer_state=has_optimizer_state
            )
        with agree_on_failure_context():
            prepared = loader.prepare(path)
        with agree_on_failure_context():
            loader.apply(prepared)
        return source

    with agree_on_failure_context():
        specs = members.get_local_shard_specs(slot)
        # PEFT and external readers load every tensor, so only rank 0 opens such a source; an adapter dir's
        # metadata is cheap, and every rank reads its own pieces of adapter_state/
        reader, source, steps = None, None, None
        if checked is not None and is_rank0():
            reader, source, steps = checked.reader, checked.source, checked.steps
        elif kind == AdapterSourceKind.ADAPTER_DIR or is_rank0():
            reader, source, steps = _open_source(path, kind, run, targets, load_optimizer=load_optimizer)
    source, reads_pieces = broadcast_from_rank0((source, _reads_adapter_state_pieces(reader, steps)))

    if reads_pieces:
        run_keys = _all_ranks_keys(specs)
        with agree_on_failure_context():
            assert_adapter_state_keys(path, run_keys, with_optimizer=load_optimizer)
        # same parametrization: every rank reads only its own pieces, and moments only when it loads them
        stored_shapes = read_adapter_state_shapes(path)
        # vocab padding of another size, or none in the source, stays zero, as at init
        read = [s for s in specs if not is_vocab_padding(s.key) or stored_shapes.get(s.key) == s.global_shape]
        weights = [torch.empty(s.local_shape, dtype=torch.float32) for s in read]
        moments = (
            [[torch.empty(s.local_shape, dtype=torch.float32) for s in read] for _ in range(2)]
            if load_optimizer
            else [[], []]
        )
        loaded = AdapterState(read, weights, *moments, step=None, has_optimizer_state=load_optimizer)
        loaded.step = reader.load_state(loaded)
        state = _zero_filled(specs, loaded)
    else:
        with agree_on_failure_context():
            weights = None
            if checked is not None and is_rank0():
                weights = checked.weights
            elif is_rank0():
                weights = apply_parametrization_conversion(reader.read_weights(), steps)
        state = _scatter_weights_only_state(weights, specs, members.head_rows)
    with agree_on_failure_context():
        members.import_adapter_state(slot, state)
    with agree_on_failure_context():
        members.finish_import_adapter_state(slot)
    return source


def _all_ranks_keys(specs: list[LocalShardSpec]) -> set[str]:
    keys = [{s.key for s in specs}]
    if torch.distributed.is_initialized():
        keys = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(keys, {s.key for s in specs}, group=gloo_group())
    return set().union(*keys)
