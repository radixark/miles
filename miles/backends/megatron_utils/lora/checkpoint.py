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
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.utils import unwrap_model

from miles.backends.megatron_utils.lora.bridge_checkpoint import BridgeAdapterSlots
from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import (
    check_adapter_source,
    load_adapter_dir,
    save_adapter_dir,
)
from miles.backends.training_utils.checkpoint.adapter.adapter_state import AdapterState, gloo_group
from miles.backends.training_utils.checkpoint.adapter.commit import UncommittedAdapterDir
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.checkpoint.adapter.readers import AdapterSourceKind

_WEIGHTS_KEY = "adapter_weights"
_OPTIM_KEY = "adapter_optimizer"
# a ChainedOptimizer prefixes each child's keys with chained_<i>.
_LEGACY_OPTIMIZER_KEY = re.compile(r"(chained_\d+\.)?optimizer\.")


class LayerWiseSlotMembers:
    """Checkpoint members of Bridge multi-LoRA slots trained by per-slot LayerWise optimizers."""

    bf16_adapter_key_style = "peft"

    def __init__(self, adapter_slots: BridgeAdapterSlots, slot_optimizers: dict[int, SlotOptimizer]):
        self._slots, self._optimizers = adapter_slots, slot_optimizers

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

    def get_local_shard_specs(self, slot: int):
        return self._slots.local_shard_specs(slot, frozenset(self._owned(slot)))

    def export_adapter_state(self, slot: int) -> AdapterState:
        owned = self._owned(slot)
        holds = frozenset(owned)
        params = self._slots.slot_parameters(slot)
        specs = self._slots.local_shard_specs(slot, holds)

        def pieces(pick) -> list[torch.Tensor]:
            # parameters held elsewhere only pad the index space; none of their pieces is taken
            aligned = [pick(*owned[i]) if i in owned else p.detach().float() for i, p in enumerate(params)]
            return [
                piece.reshape(spec.local_shape)
                for piece, spec in zip(self._slots.local_pieces(slot, aligned, holds), specs, strict=True)
            ]

        weights = pieces(lambda master, adam, group: master)
        for weight, bf16 in zip(weights, self._slots.local_pieces(slot, params, holds), strict=True):
            assert torch.equal(
                weight.to(bf16.dtype), bf16.reshape(weight.shape)
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
        """Owners write masters and moments; then every rank refreshes its bf16 parameters from them."""
        owned = self._owned(slot)
        targets = self._slots.piece_parameter(slot, frozenset(owned))
        for (index, offsets), weight, exp_avg, exp_avg_sq in zip(
            targets, state.weights, state.exp_avgs, state.exp_avg_sqs, strict=True
        ):
            master, adam, _ = owned[index]
            if not adam:
                adam["exp_avg"], adam["exp_avg_sq"] = torch.zeros_like(master), torch.zeros_like(master)
            for destination, piece in ((master, weight), (adam["exp_avg"], exp_avg), (adam["exp_avg_sq"], exp_avg_sq)):
                destination.view(-1)[offsets.flatten().to(destination.device)] = piece.flatten().to(destination)
        optimizer = self._optimizers[slot]
        for child in optimizer.chained_optimizers:
            for group in child.optimizer.param_groups:
                if state.step:
                    group["step"] = state.step
                else:
                    group.pop("step", None)
        for child in optimizer.chained_optimizers:
            child._copy_main_params_to_model_params()
        optimizer.allgather_params()


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
    mismatch = None
    try:
        kind = check_adapter_source(path, run, targets, load_optimizer=load_optimizer)
        if kind == AdapterSourceKind.LEGACY_TINKER_CHECKPOINT:
            _read_legacy_meta(path, run)
            # TE's _extra_state objects are stored per shard (<key>/shard_i_j); the tensors decide the layout
            saved = {
                k
                for k in dcp.FileSystemReader(path).read_metadata().state_dict_metadata
                if ".adapter." in k and "_extra_state" not in k and not _LEGACY_OPTIMIZER_KEY.match(k)
            }
            expected = {
                sharded.key
                for sharded in nested_values(
                    _canonicalize_slot_keys(_slot_weights_sharded_state_dict(model, slot), slot)
                )
                if "_extra_state" not in sharded.key
            }
            if expected != saved:
                mismatch = (
                    f"{path} was saved with another adapter layout than this gateway trains "
                    f"(differing keys e.g. {sorted(expected ^ saved)[:3]}); "
                    "load it on a gateway with the layout it was saved with"
                )
        elif kind.is_read_into_model():
            # load_adapter_dir refuses it only after the slot reset
            mismatch = f"{path} is a {kind.value} source, which this backend cannot load"
    # any failure to read the user's source is theirs; raised after the slot reset, it kills the trainer cell
    except Exception as error:
        mismatch = f"{path} cannot load into this model: {error}"
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, mismatch, group=gloo_group())
    return next((m for m in gathered if m is not None), None)


class LegacyTinkerCheckpointLoader:
    """A slot saved before the adapter format (torch_dist + META.json), read straight into the run's slot.

    Its parametrization is the run's; the old META.json only has to agree with it.
    """

    def __init__(self, model, slot_optimizer: SlotOptimizer, run: AdapterParametrization, load_optimizer: bool):
        self._model, self._optimizer, self._run, self._load_optimizer = model, slot_optimizer, run, load_optimizer

    def __call__(self, path: Path) -> AdapterParametrization:
        meta = _read_legacy_meta(path, self._run)
        _load_legacy_slot(self._model, self._optimizer, str(path), self._load_optimizer)
        return replace(self._run, lora_alpha=meta["lora_alpha"])


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


def _load_legacy_slot(model: Sequence[DDP], slot_optimizer: SlotOptimizer, path: str, load_optimizer: bool) -> None:
    weights = _slot_weights_sharded_state_dict(model, slot_optimizer.slot)
    shells = {_WEIGHTS_KEY: weights, _OPTIM_KEY: slot_optimizer.sharded_state(weights, is_loading=True)}
    _canonicalize_slot_keys(shells, slot_optimizer.slot)

    loaded = dist_checkpointing.load(shells, path)
    slot_optimizer.load_sharded_state(loaded[_OPTIM_KEY])
    if not load_optimizer:
        slot_optimizer.reset_adam_state()
    # the masters are authoritative, also for weights only: main's layout leaves grouped-expert adapter weights
    # unloaded in the model
    for child in slot_optimizer.chained_optimizers:
        child._copy_main_params_to_model_params()
    slot_optimizer.allgather_params()
