"""Megatron's adapter optimizer state behind the library's checkpoint members, and the legacy Tinker loader.

A multi-LoRA slot's LayerWiseDistributedOptimizer keeps each parameter whole on one DP rank: that rank holds the
fp32 master (a Float16 child's ``fp32_from_float16_groups``) and its Adam state; ``step`` lives per param group.
"""

import json
import os
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
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size, unwrap_model

from miles.backends.megatron_utils.lora.bridge_checkpoint import BridgeAdapterSlots
from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer
from miles.backends.megatron_utils.lora.utils import (
    _distributed_optimizers,
    _load_native_adapter_shard,
    _load_training_state,
    _native_adapter_shard,
)
from miles.backends.megatron_utils.rematerialize_utils import _build_cast_main_to_params_fn
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import (
    load_adapter_dir,
    read_training_state,
    save_adapter_dir,
    save_and_commit_adapter_dir,
)
from miles.backends.training_utils.checkpoint.adapter.adapter_state import AdapterState, gloo_group
from miles.backends.training_utils.checkpoint.adapter.commit import UncommittedAdapterDir
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.checkpoint.adapter.parametrization_conversion import (
    assert_parametrization_matches_run,
    plan_parametrization_conversion,
)
from miles.backends.training_utils.checkpoint.adapter.readers import (
    AdapterSourceKind,
    MilesAdapterDirReader,
    classify_adapter_source,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.hf_utils.config import load_hf_config
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

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


class DistributedOptimizerMembers:
    """Checkpoint members of single LoRA (slot 0, Bridge or native) trained by Megatron's DistributedOptimizer.

    Each DP group's rank 0 gathers whole parameters' fp32 masters and moments over Gloo and writes them; on import
    it scatters them back, and every rank re-casts its bf16 shard and all-gathers the parameters.
    """

    def __init__(
        self,
        adapter_slots: BridgeAdapterSlots,
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

    def export_bf16_adapter(self, slot: int) -> Iterable[tuple[str, torch.Tensor]]:
        return self._slots.export_bf16_adapter(slot)

    def _holds(self) -> frozenset[int]:
        """Parameters whose DP group this rank leads: expert parameters reduce over expert DP."""
        if not self._distributed_optimizers():
            # a plain optimizer keeps every master on every rank; the lowest replica writes each piece
            return frozenset(range(len(self._slots.slot_parameters(0))))
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
        subs = self._optimizer.chained_optimizers
        # another optimizer's state would be silently left out of the checkpoint
        assert all(
            isinstance(sub, DistributedOptimizer) for sub in subs
        ), f"the adapter checkpoint needs --use-distributed-optimizer; got {[type(sub).__name__ for sub in subs]}"
        return subs

    def _replicated_masters(self) -> dict[int, torch.Tensor]:
        """Index in ``slot_parameters(0)`` -> fp32 master, of the parameters a plain Float16 optimizer steps: miles
        runs a non-Adam optimizer (SGD) without the DistributedOptimizer, and it keeps no moments."""
        index_of = {id(p): i for i, p in enumerate(self._slots.slot_parameters(0))}
        masters = {}
        for child in self._optimizer.chained_optimizers:
            if isinstance(child, DistributedOptimizer):
                continue
            for g in range(len(child.optimizer.param_groups)):
                for param, master in zip(child.float16_groups[g], child.fp32_from_float16_groups[g], strict=True):
                    masters[index_of[id(param)]] = master
                for param in child.fp32_from_fp32_groups[g]:
                    masters[index_of[id(param)]] = param
        return masters

    def _whole_state(self) -> tuple[dict[int, dict[str, torch.Tensor]], int | None]:
        """Collective: on each DP group's rank 0, its parameters' whole fp32 master and moments; and the step."""
        index_of = {id(p): i for i, p in enumerate(self._slots.slot_parameters(0))}
        whole, steps = {}, set()
        for sub in self._distributed_optimizers():
            # a never-stepped optimizer has no Adam state yet (zero moments); this fills only empty states
            if sub.init_state_fn is not None:
                sub.init_state_fn(sub.optimizer, sub.config)
            elif isinstance(sub.optimizer, HybridDeviceOptimizer) and not sub.optimizer.state:
                _init_zero_hdo_state(sub.optimizer)
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
        whole |= {index: {"param": master.detach().cpu()} for index, master in self._replicated_masters().items()}
        return whole, steps.pop() if steps else None

    def export_adapter_state(self, slot: int) -> AdapterState:
        assert not (
            self._save_moments and self._replicated_masters()
        ), "a non-Adam optimizer keeps no moments; save its adapter with --no-save-optim"
        whole, step = self._whole_state()
        holds = self._holds()
        params = self._slots.slot_parameters(0)
        specs = self._slots.local_shard_specs(0, holds)

        def pieces(key: str) -> list[torch.Tensor]:
            if not holds:
                # no piece is cut here; the aligned copies would only pad the index space
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
        holds = self._holds()
        params = self._slots.slot_parameters(0)
        # the pieces of each parameter this rank leads; elements no piece holds (vocab padding) keep their state
        pieces_by_param = {i: [] for i in holds}
        for (index, offsets), *pieces in zip(
            self._slots.piece_parameter(0, holds), state.weights, state.exp_avgs, state.exp_avg_sqs, strict=True
        ):
            pieces_by_param[index].append((offsets.flatten(), pieces))
        for index, master in self._replicated_masters().items():
            for offsets, (weight, *_moments) in pieces_by_param[index]:
                master.view(-1)[offsets.to(master.device)] = weight.flatten().to(master)
        index_of = {id(p): i for i, p in enumerate(params)}
        for sub in self._distributed_optimizers():
            if sub.init_state_fn is not None:
                sub.init_state_fn(sub.optimizer, sub.config)
            # every element is overwritten below, so nothing is gathered
            gathered = sub.get_parameter_state_dp_zero(use_gloo_comm=True, empty_data=True)
            if gathered is not None:
                for buffer_index, buffer in enumerate(sub.buffers):
                    for param in buffer.param_index_map:
                        world, start, end = _world_slice(gathered, buffer_index, buffer, param)
                        for offsets, pieces in pieces_by_param[index_of[id(param)]]:
                            for key, piece in zip(_STATE_KEYS, pieces, strict=True):
                                world[key][start:end][offsets] = piece.flatten().cpu()
                del gathered["buckets_coalesced"]
            sub.load_parameter_state_from_dp_zero(gathered)
            _set_distributed_optimizer_step(sub, state.step)
        _build_cast_main_to_params_fn(self._optimizer, precision_aware=self._precision_aware)()
        if self._distributed_optimizers():
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


def _distributed_optimizer_step(sub) -> int | None:
    """None when this rank's shard of the optimizer holds no parameters (its groups are never stepped)."""
    if isinstance(sub.optimizer, HybridDeviceOptimizer):
        steps = {int(state["step"]) for state in sub.optimizer.state.values() if "step" in state}
        assert len(steps) <= 1, f"offloaded Adam states disagree on the step: {steps}"
        return steps.pop() if steps else (0 if sub.optimizer.state else None)
    groups = [group for group in sub.optimizer.param_groups if group["params"]]
    return groups[0].get("step", 0) if groups else None


def _init_zero_hdo_state(hdo: HybridDeviceOptimizer) -> None:
    """The state a never-stepped HDO is equivalent to: its fp32 master, zero moments, step 0.

    Megatron's own dummy_step would take a real Adam step on random gradients.
    """
    for group in hdo.param_groups:
        for param in group["params"]:
            master = hdo.param_to_inner_param[param]
            hdo.state[param] = {
                "step": torch.tensor(0.0),
                "exp_avg": torch.zeros_like(master),
                "exp_avg_sq": torch.zeros_like(master),
                "master_param": master,
            }


def _set_distributed_optimizer_step(sub, step: int) -> None:
    if isinstance(sub.optimizer, HybridDeviceOptimizer):
        for shard in sub.optimizer.state:
            sub.optimizer.state[shard]["step"] = torch.tensor(float(step))
        sub.optimizer._sync_hdo_state_to_sub_optimizers()
        return
    for group in sub.optimizer.param_groups:
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
    mismatch = None
    try:
        kind = classify_adapter_source(path, targets, load_optimizer=load_optimizer)
        if kind == AdapterSourceKind.LEGACY_TINKER_CHECKPOINT:
            _read_legacy_meta(path, run)
            if load_optimizer:
                # SlotOptimizer's LayerWise shards state over these groups, owner per param differing by slot;
                # META.json does not record the slot that saved it
                groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["dp_cp", "expt_dp"])
                if max(get_pg_size(groups.dp_cp), get_pg_size(groups.expt_dp)) > 1:
                    raise ValueError(
                        "a checkpoint saved before adapter checkpoints cannot restore optimizer state under data "
                        "parallelism; load it with optimizer=False (weights only)"
                    )
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
        elif kind == AdapterSourceKind.ADAPTER_DIR:
            source = MilesAdapterDirReader(path, targets).source
            steps = plan_parametrization_conversion(source.parametrization, run, source_dtype=source.weight_dtype)
            assert_parametrization_matches_run(
                source.parametrization,
                run,
                steps,
                load_optimizer=load_optimizer,
                has_optimizer_state=source.has_optimizer_state,
            )
    except (AssertionError, ValueError) as error:
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


def _read_legacy_meta(path: Path, run: AdapterParametrization) -> dict:
    """META.json, asserted to agree with ``run`` on what fixes the slot's tensor shapes."""
    # checkpoints written before the layout was recorded are all fused, per-expert
    meta = {
        "lora_type": "lora",
        "experts_shared_outer_loras": False,
        **json.loads((path / "META.json").read_text()),
    }
    run = asdict(run)
    for field in ("base_model", "lora_rank", "lora_type", "experts_shared_outer_loras"):
        assert meta[field] == run[field], f"{path} was saved with {field}={meta[field]!r}; this run has {run[field]!r}"
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
    shells = {_WEIGHTS_KEY: weights}
    if load_optimizer:
        shells[_OPTIM_KEY] = slot_optimizer.sharded_state(weights, is_loading=True)
    _canonicalize_slot_keys(shells, slot_optimizer.slot)

    loaded = dist_checkpointing.load(shells, path)
    if load_optimizer:
        slot_optimizer.load_sharded_state(loaded[_OPTIM_KEY])
        # the masters are authoritative: main's layout leaves grouped-expert adapter weights unloaded in the model
        for child in slot_optimizer.chained_optimizers:
            child._copy_main_params_to_model_params()
        slot_optimizer.allgather_params()
    else:
        slot_optimizer.reload_masters()


def build_adapter_slots(args, model, iterator, targets) -> BridgeAdapterSlots:
    """Collective: the run's adapter checkpoint view, Bridge's (single or multi-LoRA) or native's."""
    if args.megatron_to_hf_mode == "raw":
        from miles.backends.megatron_utils.lora.native.checkpoint import NativeAdapterSlots

        return NativeAdapterSlots(model, iterator, targets)
    return BridgeAdapterSlots(model, iterator, targets, multi_lora=args.multi_lora)


def single_lora_parametrization(adapter_slots: BridgeAdapterSlots, args) -> AdapterParametrization:
    base_model = args.hf_checkpoint
    # one local model under another path string (a trailing slash, a symlinked mount) is the same base; a hub id
    # is no local path
    if os.path.exists(base_model):
        base_model = os.path.realpath(base_model)
    return adapter_slots.parametrization(
        base_model=base_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_type=args.lora_type,
        experts_shared_outer_loras=args.experts_shared_outer_loras,
    )


def save_single_lora(model, optimizer, opt_param_scheduler, adapter_slots: BridgeAdapterSlots, args, iteration: int):
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


def restore_single_lora(model, args, *, optimizer, opt_param_scheduler) -> int | None:
    """``--lora-adapter-path`` from any recognized source; returns the saved iteration when the source has one.

    ``--lora-adapter-convert-save`` then commits the loaded state there in the adapter format.
    """
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator import get_hf_weight_iterator
    from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement

    iterator = get_hf_weight_iterator(
        args,
        model,
        required_placement=WeightUpdatePlacement(gather_pp=True),
        model_name=None,
        quantization_config=None,
    )
    targets = get_hf_lora_targets(load_hf_config(args.hf_checkpoint).to_dict())
    slots = build_adapter_slots(args, model, iterator, targets)
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
    legacy = LegacyRankShardLoader(model, optimizer, run, load_optimizer)
    load_adapter_dir(
        path,
        members,
        0,
        run,
        targets,
        load_optimizer=load_optimizer,
        in_model_loaders_by_kind={AdapterSourceKind.LEGACY_RANK_SHARD: legacy},
    )
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
        return None
    if opt_param_scheduler is not None and training_state["opt_param_scheduler"] is not None:
        opt_param_scheduler.load_state_dict(training_state["opt_param_scheduler"])
    return training_state["iteration"]


class LegacyRankShardLoader:
    """A single-LoRA ``--save`` written before the adapter format (``adapter_megatron_rank*.pt``), read into the
    model under the parallel layout it was saved with."""

    def __init__(self, model, optimizer, run: AdapterParametrization, load_optimizer: bool):
        self._model, self._optimizer, self._run, self._load_optimizer = model, optimizer, run, load_optimizer

    def __call__(self, path: Path) -> AdapterParametrization:
        shard = _native_adapter_shard(path)
        assert shard is not None, f"{path} has no adapter shard for this rank"
        _load_native_adapter_shard(self._model, shard)
        if self._load_optimizer:
            state = _load_training_state(path)
            assert (
                state is not None and state["optimizer"] is not None and "parameter_state" in state
            ), f"{path} has no optimizer state to resume; pass --no-load-optim"
            self._optimizer.load_state_dict(state["optimizer"])
            for sub, parameter_state in zip(
                _distributed_optimizers(self._optimizer), state["parameter_state"], strict=True
            ):
                sub.load_parameter_state_from_dp_reshardable(parameter_state)
        else:
            # the fp32 masters were built before the adapter load
            self._optimizer.reload_model_params()
        # _resolve_lora_alpha already made the run's alpha the source's
        return self._run


def _read_legacy_training_state(path: Path) -> dict | None:
    """The iteration and scheduler state of an old-format save (``training_state_rank*.pt``), if it has them.

    Every rank reads rank 0's copy: a run on more ranks than saved has no file of its own, yet must agree.
    """
    state_file = path / "training_state_rank0.pt"
    if not state_file.exists():
        return None
    # optimizer state dicts hold non-tensor objects (step counts, param group metadata)
    state = torch.load(state_file, map_location="cpu", weights_only=False)
    return {"iteration": state["iteration"], "opt_param_scheduler": state["opt_param_scheduler"]}
