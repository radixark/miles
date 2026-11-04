"""Megatron's adapter optimizer state behind the library's checkpoint members, and the legacy Tinker loader.

A multi-LoRA slot's LayerWiseDistributedOptimizer keeps each parameter whole on one DP rank: that rank holds the
fp32 master (a Float16 child's ``fp32_from_float16_groups``) and its Adam state; ``step`` lives per param group.
"""

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import apply_factories
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.utils import unwrap_model
from safetensors import SafetensorError

from miles.backends.megatron_utils.lora.bridge_checkpoint import BridgeAdapterSlots
from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import (
    HeadRows,
    check_adapter_source,
    load_adapter_dir,
    save_adapter_dir,
)
from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    AdapterState,
    gloo_group,
    raise_if_any_rank_failed,
)
from miles.backends.training_utils.checkpoint.adapter.commit import UncommittedAdapterDir
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.checkpoint.adapter.readers import AdapterSourceKind
from miles.backends.training_utils.checkpoint.adapter.source_error import AdapterSourceError

_WEIGHTS_KEY = "adapter_weights"
_OPTIM_KEY = "adapter_optimizer"
# a ChainedOptimizer prefixes each child's keys with chained_<i>.
_LEGACY_OPTIMIZER_KEY = re.compile(r"(chained_\d+\.)?optimizer\.")


class LayerWiseSlotMembers:
    """Checkpoint members of Bridge multi-LoRA slots trained by per-slot LayerWise optimizers."""

    bf16_adapter_key_style = "peft"

    def __init__(self, adapter_slots: BridgeAdapterSlots, slot_optimizers: dict[int, SlotOptimizer]):
        self._slots, self._optimizers = adapter_slots, slot_optimizers

    @property
    def head_rows(self) -> HeadRows | None:
        return self._slots.head_rows

    def export_bf16_adapter(self, slot: int) -> Iterable[tuple[str, torch.Tensor]]:
        return self._slots.export_bf16_adapter(slot)

    def _owned(self, slot: int) -> dict[int, tuple[torch.Tensor, dict, dict]]:
        """Index in ``slot_parameters(slot)`` -> (fp32 master, Adam state, its param group), for parameters held here."""
        index_of = {id(p): i for i, p in enumerate(self._slots.slot_parameters(slot))}
        owned = {}
        for child in self._optimizers[slot].chained_optimizers:
            for g, group in enumerate(child.optimizer.param_groups):
                # an fp32 parameter is its own master
                pairs = [*zip(child.float16_groups[g], child.fp32_from_float16_groups[g], strict=True)]
                pairs += [(param, param) for param in child.fp32_from_fp32_groups[g]]
                for param, master in pairs:
                    owned[index_of[id(param)]] = (master, child.optimizer.state[master], group)
        return owned

    def _holds(self, slot: int) -> tuple[dict[int, tuple[torch.Tensor, dict, dict]], list]:
        """``_owned(slot)`` and this rank's shard specs, agreed by every rank first."""
        error, owned = None, {}
        try:
            owned = self._owned(slot)
        except Exception as exc:
            error = exc
        # the specs gather over Gloo: a rank that cannot list its ownership must not leave the others there
        raise_if_any_rank_failed(error)
        return owned, self._slots.local_shard_specs(slot, frozenset(owned))

    def get_local_shard_specs(self, slot: int):
        return self._holds(slot)[1]

    def export_adapter_state(self, slot: int) -> AdapterState:
        owned, specs = self._holds(slot)
        holds = frozenset(owned)
        params = self._slots.slot_parameters(slot)
        # each piece is cut from the one held parameter it comes from; parameters held elsewhere are never read
        sources = self._slots.piece_parameter(slot, holds)

        def cut(tensor: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
            return tensor.detach().reshape(-1)[offsets.flatten().to(tensor.device)]

        def pieces(pick) -> list[torch.Tensor]:
            return [
                cut(pick(*owned[index]), offsets).reshape(spec.local_shape)
                for (index, offsets), spec in zip(sources, specs, strict=True)
            ]

        weights = pieces(lambda master, adam, group: master)
        for weight, (index, offsets) in zip(weights, sources, strict=True):
            bf16 = cut(params[index], offsets)
            # NaN-safe: a slot that diverged still saves what it holds
            assert torch.allclose(
                weight.to(bf16.dtype), bf16.reshape(weight.shape), rtol=0, atol=0, equal_nan=True
            ), "a bf16 adapter parameter is not its master"
        steps = {group.get("step", 0) for _, _, group in owned.values()}
        assert len(steps) <= 1, f"a slot's param groups disagree on the Adam step: {steps}"
        return AdapterState(
            specs,
            weights,
            pieces(lambda master, adam, group: _moment(adam, master, "exp_avg")),
            pieces(lambda master, adam, group: _moment(adam, master, "exp_avg_sq")),
            step=steps.pop() if steps else None,
        )

    def import_adapter_state(self, slot: int, state: AdapterState) -> None:
        """Owners write masters and moments into place and refresh their bf16 parameters from them."""
        optimizer = self._optimizers[slot]
        owned = self._owned(slot)
        targets = self._slots.piece_parameter(slot, frozenset(owned))
        for (index, offsets), weight, exp_avg, exp_avg_sq in zip(
            targets, state.weights, state.exp_avgs, state.exp_avg_sqs, strict=True
        ):
            master, adam, _ = owned[index]
            if not adam:
                adam["exp_avg"], adam["exp_avg_sq"] = torch.zeros_like(master), torch.zeros_like(master)
            for destination, piece in (
                (master, weight),
                (adam["exp_avg"], exp_avg),
                (adam["exp_avg_sq"], exp_avg_sq),
            ):
                destination.view(-1)[offsets.flatten().to(destination.device)] = piece.flatten().to(destination)
        for child in optimizer.chained_optimizers:
            for group in child.optimizer.param_groups:
                if state.step:
                    group["step"] = state.step
                else:
                    group.pop("step", None)
        for child in optimizer.chained_optimizers:
            child._copy_main_params_to_model_params()

    def finish_import_adapter_state(self, slot: int) -> None:
        """Every rank gathers the refreshed bf16 parameters of the owners."""
        self._optimizers[slot].allgather_params()


def _moment(state: dict, master: torch.Tensor, key: str) -> torch.Tensor:
    # Adam state appears at the first step; before it the moments are zero
    return state[key] if state else torch.zeros_like(master)


def save_slot(
    members: LayerWiseSlotMembers,
    slot: int,
    root: str,
    name: str,
    tenant_digest: str,
    parametrization: AdapterParametrization,
) -> UncommittedAdapterDir:
    return save_adapter_dir(Path(root), name, members, slot, parametrization, tenant_digest=tenant_digest)


def load_slot(
    model: Sequence[DDP],
    members: LayerWiseSlotMembers,
    slot_optimizer: SlotOptimizer,
    path: str,
    run: AdapterParametrization,
    targets,
    load_optimizer: bool,
) -> AdapterParametrization:
    """Any recognized source; a checkpoint written before the adapter format goes through the legacy loader."""
    legacy = LegacyTinkerCheckpointLoader(model, slot_optimizer, run, load_optimizer)
    return load_adapter_dir(
        Path(path),
        members,
        slot_optimizer.slot,
        run,
        targets,
        load_optimizer=load_optimizer,
        in_model_loaders_by_kind={AdapterSourceKind.LEGACY_TINKER_CHECKPOINT: legacy},
    )


def find_load_mismatch(
    model: Sequence[DDP],
    slot: int,
    path: str,
    run: AdapterParametrization,
    targets,
    load_optimizer: bool,
) -> str | None:
    """Collective: why ``path`` cannot load into ``slot`` (every rank gets the same answer), or None.

    A legacy checkpoint records no adapter layout in META.json, so its tensor keys are compared with the slot's,
    e.g. one fused GDN in_proj adapter saved before the canonical in_proj_q/k/v/z split.
    """
    path = Path(path)
    mismatch, saved, server_error = None, None, None
    # a source the user named that is missing, malformed or refused is theirs; raised after the slot reset, it kills
    # the trainer cell
    try:
        kind = check_adapter_source(path, run, targets, load_optimizer=load_optimizer)
        if kind == AdapterSourceKind.LEGACY_TINKER_CHECKPOINT:
            _read_legacy_meta(path, run)
            state_dict_metadata = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
            _assert_has_legacy_optimizer_state(path, state_dict_metadata)
            # TE's _extra_state objects are stored per shard (<key>/shard_i_j); the tensors decide the layout
            saved = {
                k: tuple(metadata.size)
                for k, metadata in state_dict_metadata.items()
                if ".adapter." in k and "_extra_state" not in k and not _LEGACY_OPTIMIZER_KEY.match(k)
            }
        elif kind.is_read_into_model():
            # load_adapter_dir refuses it only after the slot reset
            mismatch = f"{path} is a {kind.value} source, which this backend cannot load"
    except _SOURCE_ERRORS as error:
        mismatch = f"{path} cannot load into this model: {error}"
    except Exception as error:
        # storage faults (EIO, ESTALE) and bugs are the gateway's
        server_error = error
    # the slot's own layout is this gateway's: a failure computing it is a server error on every rank
    expected = None
    if saved is not None and server_error is None:
        try:
            expected = _legacy_slot_layout(model, slot)
        except Exception as error:
            server_error = error
    raise_if_any_rank_failed(server_error)
    if expected is not None:
        if expected.keys() != saved.keys():
            mismatch = (
                f"{path} was saved with another adapter layout than this gateway trains "
                f"(differing keys e.g. {sorted(expected.keys() ^ saved.keys())[:3]}); "
                "load it on a gateway with the layout it was saved with"
            )
        elif expected != saved:
            key = next(k for k in sorted(expected) if expected[k] != saved[k])
            mismatch = (
                f"{path} was saved with other tensor shapes than this gateway's slots ({key}: saved "
                f"{saved[key]}, slot {expected[key]}); legacy tensors are padded to the gateway's --lora-rank, "
                "so load it on a gateway launched with the one it was saved with"
            )
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, mismatch, group=gloo_group())
    return next((m for m in gathered if m is not None), None)


# what the source checks raise for a source the user named: absent, unreadable, malformed or refused
_SOURCE_ERRORS = (
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
    PermissionError,
    ValueError,
    KeyError,
    AssertionError,
    SafetensorError,
)


def _legacy_slot_layout(model, slot: int) -> dict[str, tuple[int, ...]]:
    """The slot's legacy torch_dist keys and global shapes."""
    slot_state = _canonicalize_slot_keys(_slot_weights_sharded_state_dict(model, slot), slot)
    # shared grouped-expert adapters are factories; built, they carry the global expert axis
    apply_factories(slot_state)
    return {
        sharded.key: tuple(sharded.global_shape)
        for sharded in nested_values(slot_state)
        if "_extra_state" not in sharded.key
    }


class LegacyTinkerCheckpointLoader:
    """A slot saved before the adapter format (torch_dist + META.json), read straight into the run's slot.

    Its parametrization is the run's; the old META.json only has to agree with it.
    """

    def __init__(self, model, slot_optimizer: SlotOptimizer, run: AdapterParametrization, load_optimizer: bool):
        self._model, self._optimizer, self._run, self._load_optimizer = model, slot_optimizer, run, load_optimizer

    def parametrization(self, path: Path) -> tuple[AdapterParametrization, bool]:
        alpha = _read_legacy_meta(path, self._run)["lora_alpha"]
        _assert_has_legacy_optimizer_state(path, dcp.FileSystemReader(path).read_metadata().state_dict_metadata)
        return replace(self._run, lora_alpha=alpha), True

    def prepare(self, path: Path) -> tuple[Path, dict]:
        error, weights = None, None
        try:
            weights = _slot_weights_sharded_state_dict(self._model, self._optimizer.slot)
        except Exception as exc:
            error = exc
        # the optimizer shells gather over the default group: a rank without slot tensors must not leave the others there
        raise_if_any_rank_failed(error)
        shells = {_WEIGHTS_KEY: weights, _OPTIM_KEY: self._optimizer.sharded_state(weights, is_loading=True)}
        _canonicalize_slot_keys(shells, self._optimizer.slot)
        return path, shells

    def apply(self, prepared: tuple[Path, dict]) -> None:
        path, shells = prepared
        _load_legacy_slot(shells, self._optimizer, str(path), self._load_optimizer)


def _assert_has_legacy_optimizer_state(path: Path, state_dict_metadata: Iterable[str]) -> None:
    # main always saved it; a weights-only load reads the saved masters too (see _load_legacy_slot)
    if not any(_LEGACY_OPTIMIZER_KEY.match(k) for k in state_dict_metadata):
        raise AdapterSourceError(f"{path} holds no optimizer state, which every legacy slot load reads")


# a gateway-wide layout setting -> the launch flag that restores a checkpoint saved under `value`
_GATEWAY_LAYOUT_FLAGS = {
    "lora_type": lambda value: f"--lora-type {value}",
    "experts_shared_outer_loras": lambda value: (
        "--experts-shared-outer-loras" if value else "no --experts-shared-outer-loras"
    ),
}


def _read_legacy_meta(path: Path, run: AdapterParametrization) -> dict:
    """META.json, checked to agree with ``run`` on what fixes the slot's tensor shapes."""
    # checkpoints written before the layout was recorded are all fused, per-expert
    meta = {
        "lora_type": "lora",
        "experts_shared_outer_loras": False,
        **json.loads((path / "META.json").read_text()),
    }
    run = asdict(run)
    for field in ("base_model", "lora_rank", "lora_type", "experts_shared_outer_loras"):
        if meta[field] != run[field]:
            message = f"{path} was saved with {field}={meta[field]!r}; this run has {run[field]!r}"
            if field in _GATEWAY_LAYOUT_FLAGS:
                message += (
                    f". The adapter layout is set per gateway: restore it on one launched with "
                    f"{_GATEWAY_LAYOUT_FLAGS[field](meta[field])}"
                )
            raise ValueError(message)
    return meta


def _slot_weights_sharded_state_dict(model: Sequence[DDP], slot: int) -> dict:
    sharded: dict = {}
    for chunk in unwrap_model(model):
        sharded |= chunk.sharded_state_dict()
    marker = f".adapters.{slot}."
    slot_sharded = {key: value for key, value in sharded.items() if marker in key}
    assert slot_sharded, f"slot {slot} exposed no adapter tensors"
    return slot_sharded


def _canonicalize_slot_keys(tree: dict, slot: int) -> dict:
    """Rewrite storage keys so a checkpoint saved from slot i loads into slot j."""
    marker, canonical = f".adapters.{slot}.", ".adapter."
    for sharded in nested_values(tree):
        if hasattr(sharded, "key"):  # config-access-exempt: nested checkpoint values differ in key support
            sharded.key = sharded.key.replace(marker, canonical)
    return tree


def _load_legacy_slot(shells: dict, slot_optimizer: SlotOptimizer, path: str, load_optimizer: bool) -> None:
    loaded = dist_checkpointing.load(shells, path)
    error = None
    try:
        slot_optimizer.load_sharded_state(loaded[_OPTIM_KEY])
        if not load_optimizer:
            slot_optimizer.reset_adam_state()
        # factory-built (grouped-expert) adapter weights come back unwritten (likely: dist_checkpointing.load writes
        # them only into the returned dict); the loaded masters are authoritative
        for child in slot_optimizer.chained_optimizers:
            child._copy_main_params_to_model_params()
    except Exception as exc:
        error = exc
    raise_if_any_rank_failed(error)
    slot_optimizer.allgather_params()
