"""torchtitan LoRA's adapter checkpoint members, the HF layout of its adapter elements, and the loader for
torchtitan's own DCP checkpoints.

The layout comes from the model's own state-dict adapter: ``sd_adapter.to_hf`` runs on index-valued tensors of each
adapter's global shape, so fused projections (wqkv, w13) split exactly as their base weights do; each rank keeps the
HF positions its DTensor shard holds. Collective: ranks agree on a probe failure, then gather every pipeline stage's
targets and shared-A groups.

The fp32 parameters are their own masters; AdamW keeps ``exp_avg`` / ``exp_avg_sq`` as DTensors shaped like them and
one ``step`` per parameter.
"""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor
from torch.optim.lr_scheduler import _update_param_group_val

from miles.backends.torchtitan_utils.lora_names import is_adapter_key, split_adapter_key
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import (
    HeadRows,
    load_adapter_dir,
    read_training_state,
    save_and_commit_adapter_dir,
)
from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    AdapterState,
    gloo_group,
    raise_if_any_rank_failed,
)
from miles.backends.training_utils.checkpoint.adapter.base_model import (
    assert_same_base_tensors,
    hf_fingerprint_tensors,
    is_fingerprint_tensor,
    resolve_hf_dir,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    AdapterParametrization,
    base_model_fingerprint,
)
from miles.backends.training_utils.checkpoint.adapter.probed_slots import (
    HfPiece,
    ProbedAdapterSlots,
    RowMapping,
    agreed_row_mapping,
    finish_local_pieces,
    rectangles,
)
from miles.backends.training_utils.checkpoint.adapter.readers import AdapterSourceKind
from miles.utils.lora.hf_lora_targets import HfLoraTargets


# a placed code is rank * _RANK_STRIDE + the element's offset in the rank's concatenated local shards
_RANK_STRIDE = 2**40


def _local_codes(param: torch.Tensor, start: int) -> torch.Tensor:
    """Collective for a DTensor. A global-shaped tensor: this rank's elements numbered from ``start`` in local
    row-major order, others -1."""
    if not isinstance(param, DTensor):
        return torch.arange(start, start + param.numel()).view(param.shape)
    local = param.to_local()
    rank = dist.get_rank()
    codes = torch.arange(start, start + local.numel(), device=local.device).view(local.shape) + rank * _RANK_STRIDE
    # DTensor places every rank's shard, so any placement (strided, uneven, multi-dim) lands on its global rows
    placed = DTensor.from_local(codes, param.device_mesh, param.placements, shape=param.shape, stride=param.stride())
    placed = placed.full_tensor().cpu()
    return torch.where(placed // _RANK_STRIDE == rank, placed % _RANK_STRIDE, -1)


def probe_torchtitan_row_mapping(params: Mapping[str, torch.Tensor], sd_adapter) -> RowMapping:
    """Collective. ``params``: the adapter parameters by titan key, in the order their local shards are
    concatenated."""
    error, pieces, groups, modules = None, [], [], set()
    try:
        pieces, groups, modules = _probe_local(params, sd_adapter)
    except Exception as exc:
        error = exc
    return agreed_row_mapping(error, pieces, groups, modules)


def _probe_local(
    params: Mapping[str, torch.Tensor], sd_adapter
) -> tuple[list[HfPiece], list[tuple[str, ...]], set[str]]:
    starts, start = {}, 0
    for key, param in params.items():
        starts[key] = start
        start += (param.to_local() if isinstance(param, DTensor) else param).numel()
    pieces, groups, modules = [], [], set()
    for key in params:
        projection, factor = split_adapter_key(key)
        if factor != "lora_b":
            continue
        lora_a_key = f"{projection}.lora_a.weight"
        lora_b = sd_adapter.to_hf({f"{projection}.weight": _local_codes(params[key], starts[key])})
        assert lora_b, f"{projection}: to_hf exports nothing"
        lora_a = _local_codes(params[lora_a_key], starts[lora_a_key])
        hf_modules = [name.removesuffix(".weight") for name in lora_b]
        modules.update(hf_modules)
        group = tuple(sorted(hf_modules))
        if len(group) > 1:
            # one titan lora_a serves every HF projection of a fused weight; it is stored once, under the group's
            # first member
            groups.append(group)
        named = [(f"{group[0]}.lora_A", lora_a)]
        named += [(f"{module}.lora_B", part) for module, part in zip(hf_modules, lora_b.values(), strict=True)]
        for hf_key, codes in named:
            if (mine := codes >= 0).any():
                pieces += [
                    HfPiece(hf_key, tuple(codes.shape), tuple(s.start for s in box), codes[box].clone())
                    for box in rectangles(hf_key, mine)
                ]
    return finish_local_pieces(pieces, start), groups, modules


class TorchtitanAdapterSlots(ProbedAdapterSlots):
    """Single-LoRA checkpoint members of the torchtitan backend; ``slot_parameters`` are the local shards."""

    bf16_adapter_key_style = "peft"

    def __init__(self, model_parts, iterator, sd_adapter, targets: HfLoraTargets):
        """Collective: probes the row mapping."""
        self._params = {
            key: param for part in model_parts for key, param in part.named_parameters() if is_adapter_key(key)
        }
        super().__init__(probe_torchtitan_row_mapping(self._params, sd_adapter), targets, iterator)

    @property
    def parameters(self) -> dict[str, torch.nn.Parameter]:
        """The adapter parameters (DTensors under FSDP) by titan key, in ``slot_parameters`` order."""
        return self._params

    def slot_parameters(self, slot: int) -> list[torch.Tensor]:
        assert slot == 0, "the torchtitan backend trains a single LoRA adapter"
        return [_local(p) for p in self._params.values()]


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class TorchtitanOptimizerMembers:
    """Checkpoint members of torchtitan single LoRA trained by its OptimizersContainer (one AdamW per model part)."""

    bf16_adapter_key_style = "peft"

    def __init__(
        self, adapter_slots: TorchtitanAdapterSlots, optimizers: list[torch.optim.Optimizer], *, save_moments: bool
    ):
        self._slots, self._optimizers, self._save_moments = adapter_slots, optimizers, save_moments

    @property
    def head_rows(self) -> HeadRows | None:
        return self._slots.head_rows

    def export_bf16_adapter(self, slot: int):
        return self._slots.export_bf16_adapter(slot)

    def get_local_shard_specs(self, slot: int):
        return self._slots.get_local_shard_specs(slot)

    def adam_states(self) -> list[dict]:
        """Each adapter parameter's AdamW state, created (zero, step 0) where no step has run yet."""
        # lazy: actor.py imports this module, and its CPU tests run without torchtitan
        from torchtitan.components.optim.utils import init_optim_state

        state_of = {}
        for optimizer in self._optimizers:
            init_optim_state(optimizer)
            state_of |= {id(p): optimizer.state[p] for group in optimizer.param_groups for p in group["params"]}
        return [state_of[id(p)] for p in self._slots.parameters.values()]

    def export_adapter_state(self, slot: int) -> AdapterState:
        states = self.adam_states()
        steps = {int(state["step"]) for state in states}
        assert len(steps) == 1, f"the adapter parameters disagree on the AdamW step: {steps}"
        specs = self._slots.get_local_shard_specs(slot)

        def pieces(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
            return [
                piece.reshape(spec.local_shape).cpu()
                for piece, spec in zip(self._slots.local_pieces(slot, tensors), specs, strict=True)
            ]

        weights = pieces(self._slots.slot_parameters(slot))
        if not self._save_moments:
            return AdapterState(specs, weights, [], [], step=steps.pop(), has_optimizer_state=False)
        return AdapterState(
            specs,
            weights,
            pieces([_local(state["exp_avg"]) for state in states]),
            pieces([_local(state["exp_avg_sq"]) for state in states]),
            step=steps.pop(),
        )

    def import_adapter_state(self, slot: int, state: AdapterState) -> None:
        """In place, into the AdamW state; restore() runs before --stream-optimizer-state-to-disk moves it to files."""
        states = self.adam_states()
        params = self._slots.slot_parameters(slot)
        with torch.no_grad():
            for (index, offsets), *pieces in zip(
                self._slots.piece_parameter(slot, None), state.weights, state.exp_avgs, state.exp_avg_sqs, strict=True
            ):
                destinations = (params[index], _local(states[index]["exp_avg"]), _local(states[index]["exp_avg_sq"]))
                for destination, piece in zip(destinations, pieces, strict=True):
                    destination.view(-1)[offsets.flatten().to(destination.device)] = piece.flatten().to(destination)
            for adam in states:
                adam["step"].fill_(state.step)


_TRAINING_STATE_KEYS = ("train_state.step", "train_state.ntokens_seen", "lr_scheduler.last_epoch")


class TorchtitanCheckpointLoader:
    """A torchtitan DCP checkpoint (``step-N/`` of torchtitan v0.4.0, or of miles before the adapter format), read
    straight into this run's adapter parameters and AdamW state under any parallel layout.

    Only lora_a / lora_b, their ``optimizer.state.<fqn>.*``, ``train_state`` and ``lr_scheduler`` are read; the base
    must match the run's HF files. The DCP records no alpha: the run's (``--lora-alpha``) is used. A model-only DCP
    leaves ``train_state`` None.
    """

    def __init__(
        self,
        adapter_slots: TorchtitanAdapterSlots,
        members: TorchtitanOptimizerMembers,
        model_parts,
        sd_adapter,
        run: AdapterParametrization,
        load_optimizer: bool,
    ):
        self._slots, self._members, self._model_parts, self._sd_adapter = (
            adapter_slots,
            members,
            model_parts,
            sd_adapter,
        )
        self._run, self._load_optimizer = run, load_optimizer
        self.train_state: dict | None = None
        self.lr_scheduler_state: dict | None = None

    def _base_fingerprint_tensors(self, path: Path, metadata) -> dict[str, torch.Tensor]:
        """The DCP's fingerprint tensors under their HF names (only what the titan model holds, e.g. no MTP)."""
        fqns, error = set(), None
        try:
            for part in self._model_parts:
                for fqn, param in part.named_parameters():
                    candidate = "norm" in fqn or fqn.startswith("layers.0.attention.")
                    if candidate and split_adapter_key(fqn) is None and fqn in metadata:
                        hf_names = self._sd_adapter.to_hf({fqn: torch.empty(param.shape, device="meta")})
                        if any(is_fingerprint_tensor(name) for name in hf_names):
                            fqns.add(fqn)
        except Exception as exc:
            error = exc
        raise_if_any_rank_failed(error)
        # pipeline stages hold different layers; every rank reads them all
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, fqns, group=gloo_group())
        tensors = {
            fqn: torch.empty(metadata[fqn].size, dtype=metadata[fqn].properties.dtype)
            for fqn in set().union(*gathered)
        }
        dcp.load(tensors, checkpoint_id=path, no_dist=True)
        return {name: t for name, t in self._sd_adapter.to_hf(tensors).items() if is_fingerprint_tensor(name)}

    def parametrization(self, path: Path) -> AdapterParametrization:
        """Collective: every check against the source, reading it only; ``apply`` then loads it."""
        metadata, error = None, None
        try:
            metadata = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
        except Exception as exc:
            error = exc
        raise_if_any_rank_failed(error)
        params = self._slots.parameters
        try:
            saved = {k for k in metadata if is_adapter_key(k)}
            expected = [None] * dist.get_world_size()
            dist.all_gather_object(expected, {k: tuple(p.shape) for k, p in params.items()}, group=gloo_group())
            expected = {k: shape for rank_params in expected for k, shape in rank_params.items()}
            assert (
                saved == expected.keys()
            ), f"{path} holds other adapters than this run's model, e.g. {sorted(saved ^ expected.keys())[:3]}"
            for key, shape in expected.items():
                assert (
                    tuple(metadata[key].size) == shape
                ), f"{path}: {key} is {tuple(metadata[key].size)}, this run's is {shape} (another --lora-rank?)"
            present = [key in metadata for key in _TRAINING_STATE_KEYS]
            assert all(present) or not any(present), f"{path} holds part of {_TRAINING_STATE_KEYS}"
            if self._load_optimizer:
                for key in params:
                    for name in ("exp_avg", "exp_avg_sq", "step"):
                        assert (
                            f"optimizer.state.{key}.{name}" in metadata
                        ), f"{path} has no optimizer state for {key}; pass --no-load-optim"
                    for name in ("exp_avg", "exp_avg_sq"):
                        dtype = metadata[f"optimizer.state.{key}.{name}"].properties.dtype
                        assert (
                            dtype == torch.float32
                        ), f"{path}: {key}'s {name} is {dtype}; only fp32 moments resume exactly"
        except Exception as exc:
            error = exc
        raise_if_any_rank_failed(error)
        # agreed before the collective dcp.load in apply: e.g. one node's HF cache may lack --hf-checkpoint
        try:
            source_tensors = self._base_fingerprint_tensors(path, metadata)
            hf_dir = resolve_hf_dir(self._run.base_model)
            assert (
                hf_dir is not None
            ), f"--hf-checkpoint {self._run.base_model} has no local HF weights to compare {path} with"
            assert_same_base_tensors(str(path), source_tensors, hf_fingerprint_tensors(hf_dir), self._run.base_model)
        except Exception as exc:
            error = exc
        raise_if_any_rank_failed(error)
        self._checked = (path, metadata)
        return replace(self._run, lora_alpha=None)

    def prepare(self, path: Path) -> tuple:
        checked_path, metadata = self._checked
        assert checked_path == path, f"prepare({path}) after checking {checked_path}"
        params = self._slots.parameters
        states = self._members.adam_states()
        loaded = dict(params)
        if self._load_optimizer:
            for key, state in zip(params, states, strict=True):
                loaded |= {f"optimizer.state.{key}.{name}": state[name] for name in ("exp_avg", "exp_avg_sq", "step")}
        # torchtitan's default final checkpoint (last_save_model_only) holds the model alone: no training state
        has_training_state = all(key in metadata for key in _TRAINING_STATE_KEYS)
        if has_training_state:
            # non-tensor entries: DCP replaces these placeholders with the saved values
            loaded |= dict.fromkeys(_TRAINING_STATE_KEYS, 0)
        return path, loaded, states, has_training_state

    def apply(self, prepared: tuple) -> None:
        path, loaded, states, has_training_state = prepared
        with torch.no_grad():
            dcp.load(loaded, checkpoint_id=path)
        if has_training_state:
            self.train_state = {
                "step": loaded["train_state.step"],
                "ntokens_seen": loaded["train_state.ntokens_seen"],
            }
            self.lr_scheduler_state = {"last_epoch": loaded["lr_scheduler.last_epoch"]}
        steps = [None] * dist.get_world_size()
        dist.all_gather_object(steps, {int(state["step"]) for state in states}, group=gloo_group())
        steps = set().union(*steps)
        assert len(steps) == 1, f"{path}'s adapter parameters disagree on the AdamW step: {steps}"


def load_lr_schedulers(lr_schedulers, state: dict) -> None:
    lr_schedulers.load_state_dict(state)
    # load_state_dict sets only the schedulers' own lr; the optimizers would take their next step at the fresh one
    for scheduler in lr_schedulers:
        for group, lr in zip(scheduler.optimizer.param_groups, scheduler.get_last_lr(), strict=True):
            _update_param_group_val(group, "lr", lr)


def _torchtitan_adapter_slots(trainer, args) -> TorchtitanAdapterSlots:
    """Collective: probes the adapter's HF layout."""
    from miles.backends.torchtitan_utils.hf_weight_iterator import TitanHfWeightIterator
    from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
    from miles.utils.hf_utils.config import load_hf_config
    from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

    iterator = TitanHfWeightIterator.build(
        args,
        trainer,
        required_placement=WeightUpdatePlacement(gather_pp=True),
        model_name=None,
        quantization_config=None,
    )
    targets = get_hf_lora_targets(load_hf_config(args.hf_checkpoint).to_dict())
    return TorchtitanAdapterSlots(trainer.model_parts, iterator, trainer.engine.state_dict_adapter, targets)


class TorchtitanLoraCheckpoint:
    """Single-LoRA RL checkpoints of the torchtitan backend: ``{save}/iter_N/adapter`` and ``--lora-adapter-path``."""

    def __init__(self, trainer, args, steps_per_rollout: int):
        from miles.utils.hf_utils.config import load_hf_config

        self._trainer, self._args, self._steps_per_rollout = trainer, args, steps_per_rollout
        # the rollout at this schedule's step 0: --start-rollout-id starts the steps there, not at rollout 0
        self._first_rollout = 0 if args.start_rollout_id is None else args.start_rollout_id
        self.slots = _torchtitan_adapter_slots(trainer, args)
        self.members = TorchtitanOptimizerMembers(
            self.slots, trainer.optimizers.optimizers, save_moments=not args.backend.no_save_optim
        )
        self.run = self.slots.parametrization(
            base_model=args.hf_checkpoint,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_type=args.lora_type,
            experts_shared_outer_loras=False,
            base_model_fingerprint=base_model_fingerprint(load_hf_config(args.hf_checkpoint).to_dict()),
        )

    def _training_state(self, iteration: int | None) -> dict:
        return {
            "iteration": iteration,
            "step": self._trainer.step,
            "first_rollout": self._first_rollout,
            # a run with these steps per rollout resumes the recorded step exactly, even off the grid
            "steps_per_rollout": self._steps_per_rollout,
            "lr_scheduler": self._trainer.engine.optim.lr_schedulers.state_dict(),
        }

    def save(self, rollout_id: int) -> None:
        save_and_commit_adapter_dir(
            Path(self._args.backend.save) / f"iter_{rollout_id:07d}",
            "adapter",
            self.members,
            0,
            self.run,
            self.slots.targets,
            training_state=self._training_state(rollout_id),
        )

    def restore(self) -> int | None:
        """``--lora-adapter-path``; returns the last completed rollout the source maps to, or None.

        ``--lora-adapter-convert-save`` then commits the loaded state there in the adapter format.
        """
        path = Path(self._args.lora_adapter_path)
        load_optimizer = not self._args.backend.no_load_optim
        loader = TorchtitanCheckpointLoader(
            self.slots,
            self.members,
            self._trainer.model_parts,
            self._trainer.engine.state_dict_adapter,
            self.run,
            load_optimizer,
        )
        source = load_adapter_dir(
            path,
            self.members,
            0,
            self.run,
            self.slots.targets,
            load_optimizer=load_optimizer,
            in_model_loaders_by_kind={AdapterSourceKind.TORCHTITAN_CHECKPOINT: loader},
        )
        # the driver resolved --lora-alpha from its own view of the path, which a trainer may see differently
        assert (
            source.lora_alpha == self.run.lora_alpha
        ), f"{path} was trained with lora_alpha={source.lora_alpha}; this run uses {self.run.lora_alpha}"
        training_state = read_training_state(path)
        if loader.train_state is not None and self._dcp_on_rollout_grid(path, loader.train_state, load_optimizer):
            training_state = self._adopt_dcp_schedule(loader)
        elif training_state is not None and "lr_scheduler" in training_state:
            training_state = self._adopt_torchtitan_dir_schedule(path, training_state, load_optimizer)
        elif training_state is not None and training_state["iteration"] is not None:
            training_state = self._adopt_megatron_dir_schedule(training_state)
        if self._args.start_rollout_id is not None:
            # the actor starts at --start-rollout-id, not after the source's rollout: count later saves' steps from it
            self._first_rollout = self._args.start_rollout_id - self._trainer.step // self._steps_per_rollout
        if self._args.lora_adapter_convert_save is not None:
            self._convert_save(training_state, load_optimizer)
        return None if training_state is None else training_state["iteration"]

    def _dcp_on_rollout_grid(self, path: Path, train_state: dict, load_optimizer: bool) -> bool:
        # a torchtitan DCP records steps, not the rollout; only one miles saved on this run's rollout grid maps
        on_grid = train_state["step"] % self._steps_per_rollout == 0
        # loading weights alone needs no schedule: a DCP off the grid (e.g. a standalone torchtitan run's) starts fresh
        assert on_grid or not load_optimizer, (
            f"{path} is at step {train_state['step']}, not a whole rollout of this run's "
            f"{self._steps_per_rollout} steps; pass --no-load-optim to load its weights and start the schedule "
            "fresh"
        )
        return on_grid

    def _adopt_dcp_schedule(self, loader: TorchtitanCheckpointLoader) -> dict:
        completed_rollouts = loader.train_state["step"] // self._steps_per_rollout
        self._first_rollout = 0
        self._trainer.engine.load_state_dict(loader.train_state)
        load_lr_schedulers(self._trainer.engine.optim.lr_schedulers, loader.lr_scheduler_state)
        return self._training_state(completed_rollouts - 1 if completed_rollouts else None)

    def _adopt_torchtitan_dir_schedule(self, path: Path, training_state: dict, load_optimizer: bool) -> dict:
        iteration = training_state["iteration"]
        self._first_rollout = training_state["first_rollout"]
        # compacted rollouts train a data-dependent number of steps each, so a run's own dir may sit off the grid
        same_steps_per_rollout = training_state.get("steps_per_rollout") == self._steps_per_rollout
        # the schedule is restored by step: on another --global-batch-size grid it would not match the rollout
        grid_step = 0 if iteration is None else (iteration + 1 - self._first_rollout) * self._steps_per_rollout
        if same_steps_per_rollout:
            grid_step = training_state["step"]
        assert training_state["step"] == grid_step or not load_optimizer, (
            f"{path} is at step {training_state['step']} after iteration {iteration}, but this run takes "
            f"{self._steps_per_rollout} steps per rollout (step {grid_step}): keep its rollout and batch sizes, "
            "or pass --no-load-optim to load its weights and start the schedule fresh"
        )
        if training_state["step"] != grid_step:
            # weights only: a fresh schedule after the source's rollout, as for a Megatron dir off the grid
            self._first_rollout = 0 if iteration is None else iteration + 1
            return self._training_state(iteration)
        load_lr_schedulers(self._trainer.engine.optim.lr_schedulers, training_state["lr_scheduler"])
        self._trainer.engine.num_completed_steps = training_state["step"]
        return training_state

    def _adopt_megatron_dir_schedule(self, training_state: dict) -> dict:
        """Megatron's scheduler state does not apply here: resume this run's schedule after the rollouts the source
        trained from its first rollout on."""
        iteration = training_state["iteration"]
        self._first_rollout = training_state["first_rollout"]
        step = (iteration + 1 - self._first_rollout) * self._steps_per_rollout
        load_lr_schedulers(self._trainer.engine.optim.lr_schedulers, {"last_epoch": step})
        self._trainer.engine.num_completed_steps = step
        return training_state

    def _convert_save(self, training_state: dict | None, load_optimizer: bool) -> None:
        destination = Path(self._args.lora_adapter_convert_save)
        # a convert-save writes moments only when they were loaded
        convert_members = TorchtitanOptimizerMembers(
            self.slots,
            self._trainer.optimizers.optimizers,
            save_moments=load_optimizer and not self._args.backend.no_save_optim,
        )
        save_and_commit_adapter_dir(
            destination.parent,
            destination.name,
            convert_members,
            0,
            self.run,
            self.slots.targets,
            training_state=training_state,
        )
