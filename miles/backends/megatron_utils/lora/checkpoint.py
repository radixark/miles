"""Megatron's adapter optimizer state behind the library's checkpoint members, and the legacy Tinker loader.

A multi-LoRA slot's LayerWiseDistributedOptimizer keeps each parameter whole on one DP rank: that rank holds the
fp32 master (a Float16 child's ``fp32_from_float16_groups``) and its Adam state; ``step`` lives per param group.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import apply_factories
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.core.utils import unwrap_model
from safetensors import SafetensorError

from miles.backends.megatron_utils.lora.bridge_checkpoint import BridgeAdapterSlots
from miles.backends.megatron_utils.lora.bridge_peft_checkpoint import MegatronBridgeCheckpointLoader
from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer
from miles.backends.megatron_utils.lora.utils import (
    OPTIMIZER_KEY,
    _distributed_optimizers,
    _load_training_state,
    _native_adapter_shard,
    _read_native_adapter_shard,
    _zero_init_empty_hdo_states,
)
from miles.backends.megatron_utils.rematerialize_utils import _build_cast_main_to_params_fn
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import (
    HeadRows,
    check_adapter_source,
    load_adapter_dir,
    read_training_state,
    save_adapter_dir,
    save_and_commit_adapter_dir,
)
from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    AdapterState,
    gloo_group,
    raise_if_any_rank_failed,
)
from miles.backends.training_utils.checkpoint.adapter.commit import UncommittedAdapterDir
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    AdapterParametrization,
    base_model_fingerprint,
)
from miles.backends.training_utils.checkpoint.adapter.probed_slots import ProbedAdapterSlots
from miles.backends.training_utils.checkpoint.adapter.readers import AdapterSourceKind, read_source_lora_alpha
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.hf_utils.config import load_hf_config

_WEIGHTS_KEY = "adapter_weights"
_OPTIM_KEY = "adapter_optimizer"


class LayerWiseSlotMembers:
    """Checkpoint members of Bridge multi-LoRA slots trained by per-slot LayerWise optimizers."""

    bf16_adapter_key_style = "peft"

    def __init__(self, adapter_slots: ProbedAdapterSlots, slot_optimizers: dict[int, SlotOptimizer]):
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

    def get_local_shard_specs(self, slot: int):
        return self._slots.local_shard_specs(slot, frozenset(self._owned(slot)))

    def export_adapter_state(self, slot: int) -> AdapterState:
        error, owned = None, {}
        try:
            owned = self._owned(slot)
        except Exception as exc:
            error = exc
        # the specs below gather over Gloo: a rank that cannot list its ownership must not leave the others there
        raise_if_any_rank_failed(error)
        holds = frozenset(owned)
        params = self._slots.slot_parameters(slot)
        specs = self._slots.local_shard_specs(slot, holds)
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


class DistributedOptimizerMembers:
    """Checkpoint members of single LoRA (slot 0, Bridge or native) trained by Megatron's DistributedOptimizer.

    Each DP group's rank 0 gathers whole parameters' fp32 masters and moments over Gloo and writes them; on import
    it scatters them back, and every rank re-casts its bf16 shard and all-gathers the parameters.
    """

    def __init__(
        self,
        adapter_slots: ProbedAdapterSlots,
        optimizer,
        model: Sequence[DDP],
        *,
        precision_aware: bool,
        save_moments: bool = True,
    ):
        self._slots, self._optimizer, self._model = adapter_slots, optimizer, model
        self._precision_aware, self._save_moments = precision_aware, save_moments
        # the export is the slots'; native's names its experts like sglang
        self.bf16_adapter_key_style = adapter_slots.bf16_adapter_key_style

    @property
    def head_rows(self) -> HeadRows | None:
        return self._slots.head_rows

    def export_bf16_adapter(self, slot: int) -> Iterable[tuple[str, torch.Tensor]]:
        return self._slots.export_bf16_adapter(slot)

    def _holds(self) -> frozenset[int]:
        parallel = get_parallel_state()
        # the DistributedOptimizer gathers dense parameters over intra-DP x CP and expert ones over expert DP
        dense_lead, expert_lead = parallel.intra_dp_cp.rank == 0, parallel.edp.rank == 0
        return frozenset(
            i for i, p in enumerate(self._slots.slot_parameters(0)) if (expert_lead if _is_expert(p) else dense_lead)
        )

    def get_local_shard_specs(self, slot: int):
        assert slot == 0, "single LoRA has only slot 0"
        return self._slots.local_shard_specs(0, self._holds())

    def _distributed_optimizers(self) -> list:
        return _distributed_optimizers(self._optimizer)

    def _whole_state(self) -> tuple[dict[int, dict[str, torch.Tensor]], int | None]:
        """Collective: on each DP group's rank 0, its parameters' whole fp32 master and moments; and the step."""
        index_of = {id(p): i for i, p in enumerate(self._slots.slot_parameters(0))}
        whole, steps = {}, set()
        for sub in self._distributed_optimizers():
            # a never-stepped optimizer has no Adam state yet (zero moments); this fills only empty states
            if sub.init_state_fn is not None:
                sub.init_state_fn(sub.optimizer, sub.config)
            else:
                _zero_init_empty_hdo_states([sub])
            gathered = sub.get_parameter_state_dp_zero(use_gloo_comm=True)
            if (step := _distributed_optimizer_step(sub)) is not None:
                steps.add(step)
            if gathered is None:
                continue
            for buffer_index, buffer in enumerate(sub.buffers):
                for param in buffer.param_index_map:
                    world, start, end = _world_slice(gathered, buffer_index, buffer, param)
                    whole[index_of[id(param)]] = {k: world[k][start:end].view(param.shape) for k in _STATE_KEYS}
        assert len(steps) <= 1, f"the DistributedOptimizers disagree on the Adam step: {steps}"
        return whole, steps.pop() if steps else None

    def export_adapter_state(self, slot: int) -> AdapterState:
        whole, step = self._whole_state()
        holds = self._holds()
        params = self._slots.slot_parameters(0)
        specs = self._slots.local_shard_specs(0, holds)

        def pieces(key: str) -> list[torch.Tensor]:
            if not holds:
                # a rank that leads no DP group writes no pieces
                return []
            aligned = [whole[i][key] if i in whole else p.detach().float().cpu() for i, p in enumerate(params)]
            return [
                piece.reshape(spec.local_shape)
                for piece, spec in zip(self._slots.local_pieces(0, aligned, holds), specs, strict=True)
            ]

        assert holds <= whole.keys(), "a led adapter parameter is outside every DistributedOptimizer"
        weights = pieces("param")
        for weight, bf16 in zip(weights, self._slots.local_pieces(0, params, holds), strict=True):
            assert torch.equal(
                weight.to(bf16), bf16.reshape(weight.shape)
            ), "a bf16 adapter parameter is not its master"
        if not self._save_moments:
            return AdapterState(specs, weights, [], [], step=step, has_optimizer_state=False)
        return AdapterState(specs, weights, pieces("exp_avg"), pieces("exp_avg_sq"), step=step)

    def import_adapter_state(self, slot: int, state: AdapterState) -> None:
        """Rank-local: groups the pieces by the parameter this rank leads; the scatter waits for finish."""
        holds = self._holds()
        # the pieces of each parameter this rank leads; elements no piece holds (vocab padding) keep their state
        pieces_by_param = {i: [] for i in holds}
        for (index, offsets), *pieces in zip(
            self._slots.piece_parameter(0, holds), state.weights, state.exp_avgs, state.exp_avg_sqs, strict=True
        ):
            pieces_by_param[index].append((offsets.flatten(), pieces))
        self._imported = pieces_by_param, state.step

    def finish_import_adapter_state(self, slot: int) -> None:
        """Every rank scatters the leaders' pieces over DP, then re-casts and syncs the bf16 parameters."""
        pieces_by_param, step = self._imported
        del self._imported
        index_of = {id(p): i for i, p in enumerate(self._slots.slot_parameters(0))}
        for sub in self._distributed_optimizers():
            if sub.init_state_fn is not None:
                sub.init_state_fn(sub.optimizer, sub.config)
            else:
                _zero_init_empty_hdo_states([sub])
            # elements no piece holds (vocab padding) must keep their master and moments, so they are gathered
            gathered = sub.get_parameter_state_dp_zero(
                use_gloo_comm=True, empty_data=self._slots.load_overwrites_every_element
            )
            if gathered is not None:
                for buffer_index, buffer in enumerate(sub.buffers):
                    for param in buffer.param_index_map:
                        world, start, end = _world_slice(gathered, buffer_index, buffer, param)
                        for offsets, pieces in pieces_by_param[index_of[id(param)]]:
                            for key, piece in zip(_STATE_KEYS, pieces, strict=True):
                                world[key][start:end][offsets] = piece.flatten().cpu()
                del gathered["buckets_coalesced"]
            sub.load_parameter_state_from_dp_zero(gathered)
            _set_distributed_optimizer_step(sub, step)
        _build_cast_main_to_params_fn(self._optimizer, precision_aware=self._precision_aware)()
        for chunk in self._model:
            chunk.start_param_sync(force_sync=True)


_STATE_KEYS = ("param", "exp_avg", "exp_avg_sq")


def _is_expert(param: torch.nn.Parameter) -> bool:
    # Megatron marks parameters reduced over expert DP instead of DP
    return not getattr(param, "allreduce", True)  # config-access-exempt: Megatron sets allreduce on expert params only


def _world_slice(gathered: dict, buffer_index: int, buffer, param) -> tuple[dict, int, int]:
    """``param``'s range in the dp-zero world tensors, which drop each bucket's end padding."""
    start, end, bucket_index = buffer.param_index_map[param]
    offset = start - buffer.bucket_indices[bucket_index][0]
    offset += sum(buffer.buckets[j].numel_unpadded for j in range(bucket_index))
    return gathered[buffer_index][(buffer.param_dtype, buffer.grad_dtype)], offset, offset + (end - start)


def _adam_optimizers(sub) -> list[torch.optim.Optimizer]:
    """The optimizers that step: an offloaded optimizer runs torch Adam on the host and TE FusedAdam on the GPU."""
    if isinstance(sub.optimizer, HybridDeviceOptimizer):
        return sub.optimizer.sub_optimizers
    return [sub.optimizer]


def _steps_per_parameter(optimizer: torch.optim.Optimizer) -> bool:
    # torch Adam keeps the step in each parameter's state; TE FusedAdam keeps it in each param group
    return isinstance(optimizer, torch.optim.Adam | torch.optim.AdamW)


def _distributed_optimizer_step(sub) -> int | None:
    """None when this rank's shard of the optimizer holds no parameters (its groups are never stepped)."""
    steps = set()
    for optimizer in _adam_optimizers(sub):
        if not any(group["params"] for group in optimizer.param_groups):
            continue
        if _steps_per_parameter(optimizer):
            # a never-stepped torch Adam has no state yet
            steps |= {int(state["step"]) for state in optimizer.state.values()} or {0}
        else:
            steps |= {int(group.get("step", 0)) for group in optimizer.param_groups if group["params"]}
    assert len(steps) <= 1, f"the Adam optimizers of one DistributedOptimizer disagree on the step: {steps}"
    return steps.pop() if steps else None


def _set_distributed_optimizer_step(sub, step: int) -> None:
    if isinstance(sub.optimizer, HybridDeviceOptimizer):
        # the sub-optimizers' states become the HDO's state dicts, so the step written below reaches both
        sub.optimizer._sync_hdo_state_to_sub_optimizers()
    for optimizer in _adam_optimizers(sub):
        if _steps_per_parameter(optimizer):
            for state in optimizer.state.values():
                state["step"] = torch.tensor(float(step))
            continue
        for group in optimizer.param_groups:
            if group["params"]:
                group["step"] = step


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
            # TE's _extra_state objects are stored per shard (<key>/shard_i_j); the tensors decide the layout
            saved = {
                k: tuple(metadata.size)
                for k, metadata in dcp.FileSystemReader(path).read_metadata().state_dict_metadata.items()
                if ".adapter." in k and "_extra_state" not in k and not OPTIMIZER_KEY.match(k)
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

    def parametrization(self, path: Path) -> AdapterParametrization:
        return replace(self._run, lora_alpha=_read_legacy_meta(path, self._run)["lora_alpha"])

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
        # dist_checkpointing.load leaves factory-built (grouped-expert) adapter weights unwritten (root cause open); the
        # loaded masters are authoritative
        for child in slot_optimizer.chained_optimizers:
            child._copy_main_params_to_model_params()
    except Exception as exc:
        error = exc
    raise_if_any_rank_failed(error)
    slot_optimizer.allgather_params()


def build_adapter_slots(args, model, iterator, targets) -> ProbedAdapterSlots:
    """Collective: the run's adapter checkpoint view, Bridge's (single or multi-LoRA) or native's."""
    if args.megatron_to_hf_mode == "raw":
        from miles.backends.megatron_utils.lora.native.checkpoint import NativeAdapterSlots

        return NativeAdapterSlots(model, iterator, targets)
    return BridgeAdapterSlots(model, iterator, targets, multi_lora=args.multi_lora)


def single_lora_parametrization(adapter_slots: ProbedAdapterSlots, args) -> AdapterParametrization:
    """The run's parametrization; under multi-LoRA, slot 0's rank and alpha, which its exported tensors carry."""
    spec = adapter_slots.slot_spec(0)
    return adapter_slots.parametrization(
        base_model=args.hf_checkpoint,
        lora_rank=args.lora_rank if spec is None else spec.rank,
        lora_alpha=args.lora_alpha if spec is None else spec.alpha,
        lora_type=args.lora_type,
        experts_shared_outer_loras=args.experts_shared_outer_loras,
        base_model_fingerprint=base_model_fingerprint(load_hf_config(args.hf_checkpoint).to_dict()),
    )


def save_single_lora(
    model, optimizer, opt_param_scheduler, adapter_slots: ProbedAdapterSlots, args, iteration: int, first_rollout: int
):
    """``{save}/iter_N/adapter``: adapter state, training state and the PEFT export, committed by rank 0."""
    members = DistributedOptimizerMembers(
        adapter_slots,
        optimizer,
        model,
        precision_aware=args.backend.use_precision_aware_optimizer,
        save_moments=not args.backend.no_save_optim,
    )
    training_state = {
        "iteration": iteration,
        # another backend resumes the schedule from it: Megatron's scheduler counts samples, which a skipped step or
        # a dynamic global batch decouples from rollouts
        "first_rollout": first_rollout,
        "opt_param_scheduler": opt_param_scheduler.state_dict() if opt_param_scheduler is not None else None,
    }
    save_and_commit_adapter_dir(
        Path(args.backend.save) / f"iter_{iteration:07d}",
        "adapter",
        members,
        0,
        single_lora_parametrization(adapter_slots, args),
        adapter_slots.targets,
        training_state=training_state,
    )


class AdapterRestore(NamedTuple):
    # the saved iteration; None when the source records no training state
    iteration: int | None
    scheduler_restored: bool
    # the rollout the saving run's schedule started at
    first_rollout: int


def restore_single_lora(
    model, args, *, adapter_slots: ProbedAdapterSlots, optimizer, opt_param_scheduler
) -> AdapterRestore:
    """``--lora-adapter-path`` from any recognized source; returns the saved iteration when the source has one,
    whether it restored the scheduler, and the rollout the saving run's schedule started at.

    ``--lora-adapter-convert-save`` then commits the loaded state there in the adapter format.
    """
    assert adapter_slots is not None, "the actor builds the adapter slots before restoring --lora-adapter-path"
    slots, targets = adapter_slots, adapter_slots.targets
    load_optimizer = not args.backend.no_load_optim
    # a convert-save writes moments only when they were loaded
    members = DistributedOptimizerMembers(
        slots,
        optimizer,
        model,
        precision_aware=args.backend.use_precision_aware_optimizer,
        save_moments=load_optimizer and not args.backend.no_save_optim,
    )
    run = single_lora_parametrization(slots, args)
    path = Path(args.lora_adapter_path)
    in_model_loaders_by_kind = {
        AdapterSourceKind.LEGACY_RANK_SHARD: LegacyRankShardLoader(
            model, optimizer, run, load_optimizer, precision_aware=args.backend.use_precision_aware_optimizer
        )
    }
    # a Bridge checkpoint's sharded keys exist only in a Bridge model; elsewhere load_adapter_dir refuses it as a kind
    # this backend cannot load
    if args.megatron_to_hf_mode == "bridge":
        in_model_loaders_by_kind[AdapterSourceKind.MEGATRON_BRIDGE_CHECKPOINT] = MegatronBridgeCheckpointLoader(
            model, optimizer, members, run, load_optimizer
        )
    source = load_adapter_dir(
        path,
        members,
        0,
        run,
        targets,
        load_optimizer=load_optimizer,
        in_model_loaders_by_kind=in_model_loaders_by_kind,
    )
    # the driver resolved --lora-alpha from its own view of the path, which a trainer may see differently
    assert (
        source.lora_alpha == run.lora_alpha
    ), f"{path} was trained with lora_alpha={source.lora_alpha}; this run uses {run.lora_alpha}"
    # an old-format save holds its training state beside the rank shards, whichever reader loaded the weights
    training_state = read_training_state(path) or _read_legacy_training_state(path)
    if args.lora_adapter_convert_save is not None:
        destination = Path(args.lora_adapter_convert_save)
        save_and_commit_adapter_dir(
            destination.parent,
            destination.name,
            members,
            0,
            run,
            targets,
            training_state=training_state,
        )
    if training_state is None:
        return AdapterRestore(None, False, 0)
    # a torchtitan run's training state has its own scheduler state, not Megatron's
    scheduler_state = training_state.get("opt_param_scheduler")
    if opt_param_scheduler is not None and scheduler_state is not None:
        opt_param_scheduler.load_state_dict(scheduler_state)
    return AdapterRestore(training_state["iteration"], scheduler_state is not None, training_state["first_rollout"])


class LegacyRankShardLoader:
    """A single-LoRA ``--save`` written before the adapter format (``adapter_megatron_rank*.pt``), read into the
    model under the parallel layout it was saved with."""

    def __init__(self, model, optimizer, run: AdapterParametrization, load_optimizer: bool, *, precision_aware: bool):
        self._model, self._optimizer, self._run, self._load_optimizer = model, optimizer, run, load_optimizer
        self._precision_aware = precision_aware

    def parametrization(self, path: Path) -> AdapterParametrization:
        # an old save records alpha in its PEFT pair; load_adapter_dir takes the run's when it records none
        return replace(self._run, lora_alpha=read_source_lora_alpha(path))

    def prepare(self, path: Path) -> tuple[list, dict | None]:
        # every rank-local read and check, so a rank missing its shard fails before any rank enters a collective
        shard = _native_adapter_shard(path)
        assert shard is not None, f"{path} has no adapter shard for this rank"
        tensors = _read_native_adapter_shard(self._model, shard)
        state = None
        if self._load_optimizer:
            state = _load_training_state(path)
            assert (
                state is not None and state["optimizer"] is not None and "parameter_state" in state
            ), f"{path} has no optimizer state to resume; pass --no-load-optim"
            assert len(state["parameter_state"]) == len(
                _distributed_optimizers(self._optimizer)
            ), f"{path} holds parameter state for another number of distributed optimizers"
        return tensors, state

    def apply(self, prepared: tuple[list, dict | None]) -> None:
        tensors, state = prepared
        for param, tensor in tensors:
            param.data.copy_(tensor.to(device=param.device))
        if state is not None:
            subs = _distributed_optimizers(self._optimizer)
            _zero_init_empty_hdo_states(subs)
            self._optimizer.load_state_dict(state["optimizer"])
            for sub, parameter_state in zip(subs, state["parameter_state"], strict=True):
                sub.load_parameter_state_from_dp_reshardable(parameter_state)
                if isinstance(sub.optimizer, HybridDeviceOptimizer):
                    # Megatron's load_parameter_state_from_dp_reshardable leaves HDO sub-optimizers stale;
                    # drop once miles-main syncs them there
                    sub.optimizer._sync_hdo_state_to_sub_optimizers()
            _build_cast_main_to_params_fn(self._optimizer, precision_aware=self._precision_aware)()
            for chunk in self._model:
                chunk.start_param_sync(force_sync=True)
        else:
            # the fp32 masters were built before the adapter load
            self._optimizer.reload_model_params()


def _read_legacy_training_state(path: Path) -> dict | None:
    """The iteration and scheduler state of an old-format save (``training_state_rank*.pt``), if it has them.

    Every rank reads rank 0's copy: a run on more ranks than saved has no file of its own, yet must agree.
    """
    state_file = path / "training_state_rank0.pt"
    if not state_file.exists():
        return None
    # optimizer state dicts hold non-tensor objects (step counts, param group metadata)
    state = torch.load(state_file, map_location="cpu", weights_only=False)
    # it predates first_rollout: its schedule is taken to count from rollout 0
    return {"iteration": state["iteration"], "first_rollout": 0, "opt_param_scheduler": state["opt_param_scheduler"]}
