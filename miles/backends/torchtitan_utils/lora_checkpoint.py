"""torchtitan LoRA's adapter checkpoint members, the HF layout of its adapter elements, and the loader for
torchtitan's own DCP checkpoints.

The layout comes from the model's own state-dict adapter: ``sd_adapter.to_hf`` runs on index-valued tensors of each
adapter's global shape, so fused projections (wqkv, w13) split exactly as their base weights do; each rank keeps the
HF positions its DTensor shard holds. Needs no collective except agreeing who writes replicated pieces.

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
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.optim.lr_scheduler import _update_param_group_val
from torchtitan.components.optim.utils import init_optim_state

from miles.backends.torchtitan_utils.lora_names import is_adapter_key, split_adapter_key
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import (
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
    fingerprint_named_tensors,
    hf_base_fingerprint,
    is_fingerprint_tensor,
)
from miles.backends.training_utils.checkpoint.adapter.key_mapping import map_peft_keys_to_hf_keys
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.checkpoint.adapter.probed_slots import (
    HfPiece,
    ProbedAdapterSlots,
    RowMapping,
    rectangle,
)
from miles.backends.training_utils.checkpoint.adapter.readers import AdapterSourceKind
from miles.utils.lora.hf_lora_targets import HfLoraTargets

_PEFT_FACTOR = {"lora_a": "lora_A", "lora_b": "lora_B"}


def hf_export_names(sd_adapter, projection: str, weight_shape: tuple[int, ...]) -> list[str]:
    """The HF modules a titan projection's weight exports as; several for a fused projection."""
    probe = torch.empty(weight_shape, device="meta")
    return [name.removesuffix(".weight") for name in sd_adapter.to_hf({f"{projection}.weight": probe})]


def _local_box(param: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(param, DTensor):
        shape, offset = compute_local_shape_and_global_offset(param.shape, param.device_mesh, param.placements)
        return tuple(shape), tuple(offset)
    return tuple(param.shape), (0,) * param.dim()


def _local_codes(param: torch.Tensor, start: int) -> torch.Tensor:
    """A global-shaped tensor: this rank's elements numbered from ``start`` in local row-major order, others -1."""
    shape, offset = _local_box(param)
    codes = torch.full(tuple(param.shape), -1, dtype=torch.long)
    box = tuple(slice(o, o + n) for o, n in zip(offset, shape, strict=True))
    codes[box] = torch.arange(start, start + int(torch.Size(shape).numel())).view(shape)
    return codes


def probe_torchtitan_row_mapping(params: Mapping[str, torch.Tensor], sd_adapter, targets: HfLoraTargets) -> RowMapping:
    """``params``: the adapter parameters by titan key, in the order their local shards are concatenated."""
    starts, start = {}, 0
    for key, param in params.items():
        starts[key] = start
        start += int(torch.Size(_local_box(param)[0]).numel())
    pieces, groups, modules = [], [], set()
    for key in params:
        projection, factor = split_adapter_key(key)
        if factor != "lora_b":
            continue
        lora_a_key = f"{projection}.lora_a.weight"
        lora_b = sd_adapter.to_hf({f"{projection}.weight": _local_codes(params[key], starts[key])})
        lora_a = _local_codes(params[lora_a_key], starts[lora_a_key])
        hf_modules = [map_peft_keys_to_hf_keys(f"{name.removesuffix('.weight')}.lora_A.weight").key for name in lora_b]
        hf_modules = [k.removesuffix(".lora_A") for k in hf_modules]
        modules.update(hf_modules)
        if len(hf_modules) > 1:
            # one titan lora_a serves every HF projection of a fused weight; it is stored once, under the first
            groups.append(tuple(hf_modules))
        named = [(f"{hf_modules[0]}.lora_A", lora_a)]
        named += [(f"{module}.lora_B", part) for module, part in zip(hf_modules, lora_b.values(), strict=True)]
        for hf_key, codes in named:
            if (mine := codes >= 0).any():
                box = rectangle(hf_key, mine)
                pieces.append(HfPiece(hf_key, tuple(codes.shape), tuple(s.start for s in box), codes[box]))
    covered = torch.zeros(start, dtype=torch.bool)
    for piece in pieces:
        covered[piece.local_elements.flatten()] = True
    assert covered.all(), f"to_hf does not reach {int((~covered).sum())} of this rank's adapter elements"
    return RowMapping(pieces, tuple(sorted(groups)), tuple(sorted(modules)))


class TorchtitanAdapterSlots(ProbedAdapterSlots):
    """Single-LoRA checkpoint members of the torchtitan backend; ``slot_parameters`` are the local shards."""

    bf16_adapter_key_style = "peft"

    def __init__(self, model_parts, iterator, sd_adapter, targets: HfLoraTargets):
        self._iterator, self.targets = iterator, targets
        self._specs = {}
        self._params = {
            key: param
            for part in model_parts
            for key, param in part.named_parameters()
            if split_adapter_key(key) is not None
        }
        self.row_mapping = probe_torchtitan_row_mapping(self._params, sd_adapter, targets)

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

    def __init__(self, adapter_slots: TorchtitanAdapterSlots, optimizers: list[torch.optim.Optimizer]):
        self._slots, self._optimizers = adapter_slots, optimizers

    def export_bf16_adapter(self, slot: int):
        return self._slots.export_bf16_adapter(slot)

    def get_local_shard_specs(self, slot: int):
        return self._slots.get_local_shard_specs(slot)

    def _states(self) -> list[dict]:
        """Each adapter parameter's AdamW state, created (zero, step 0) where no step has run yet."""
        state_of = {}
        for optimizer in self._optimizers:
            init_optim_state(optimizer)
            state_of |= {id(p): optimizer.state[p] for group in optimizer.param_groups for p in group["params"]}
        return [state_of[id(p)] for p in self._slots.parameters.values()]

    def export_adapter_state(self, slot: int) -> AdapterState:
        states = self._states()
        steps = {int(state["step"]) for state in states}
        assert len(steps) == 1, f"the adapter parameters disagree on the AdamW step: {steps}"
        specs = self._slots.get_local_shard_specs(slot)

        def pieces(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
            return [
                piece.reshape(spec.local_shape).cpu()
                for piece, spec in zip(self._slots.local_pieces(slot, tensors), specs, strict=True)
            ]

        return AdapterState(
            specs,
            pieces(self._slots.slot_parameters(slot)),
            pieces([_local(state["exp_avg"]) for state in states]),
            pieces([_local(state["exp_avg_sq"]) for state in states]),
            step=steps.pop(),
        )

    def import_adapter_state(self, slot: int, state: AdapterState) -> None:
        """In place: the moments may be file-backed (--stream-optimizer-state-to-disk)."""
        states = self._states()
        params = self._slots.slot_parameters(slot)
        with torch.no_grad():
            for (index, offsets), *pieces in zip(
                self._slots.piece_parameter(slot, None), state.weights, state.exp_avgs, state.exp_avg_sqs, strict=True
            ):
                destinations = (params[index], _local(states[index]["exp_avg"]), _local(states[index]["exp_avg_sq"]))
                for destination, piece in zip(destinations, pieces, strict=True):
                    destination.view(-1)[offsets.flatten().to(destination.device)] = piece.flatten().to(destination)
            for adam in states:
                adam["step"].fill_(state.step or 0)


class TorchtitanCheckpointLoader:
    """A torchtitan DCP checkpoint (``step-N/`` of torchtitan v0.4.0, or of miles before the adapter format), read
    straight into this run's adapter parameters and AdamW state under any parallel layout.

    Only lora_a / lora_b, their ``optimizer.state.<fqn>.*``, ``train_state`` and ``lr_scheduler`` are read; the base
    must match the run's HF files. The DCP records no alpha: the run's (``--lora-alpha``) is used.
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

    def _base_fingerprint(self, path: Path, metadata) -> str:
        names = {}
        for part in self._model_parts:
            for fqn, param in part.named_parameters():
                if "norm" in fqn and fqn in metadata:
                    (hf_name,) = self._sd_adapter.to_hf({fqn: torch.empty(param.shape, device="meta")})
                    if is_fingerprint_tensor(hf_name):
                        names[fqn] = hf_name
        # pipeline stages hold different norms; every rank reads them all
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, names, group=gloo_group())
        names = {fqn: hf for rank_names in gathered for fqn, hf in rank_names.items()}
        tensors = {fqn: torch.empty(metadata[fqn].size, dtype=metadata[fqn].properties.dtype) for fqn in names}
        dcp.load(tensors, checkpoint_id=path, no_dist=True)
        return fingerprint_named_tensors((names[fqn], tensor) for fqn, tensor in tensors.items())

    def __call__(self, path: Path) -> AdapterParametrization:
        metadata = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
        params = self._slots.parameters
        error = None
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
        assert self._base_fingerprint(path, metadata) == hf_base_fingerprint(
            Path(self._run.base_model)
        ), f"{path} was trained on another base model than --hf-checkpoint {self._run.base_model}"

        states = self._members._states()
        loaded = dict(params)
        if self._load_optimizer:
            for key, state in zip(params, states, strict=True):
                loaded |= {f"optimizer.state.{key}.{name}": state[name] for name in ("exp_avg", "exp_avg_sq", "step")}
        # non-tensor entries: DCP replaces these placeholders with the saved values
        loaded |= {"train_state.step": 0, "train_state.ntokens_seen": 0, "lr_scheduler.last_epoch": 0}
        with torch.no_grad():
            dcp.load(loaded, checkpoint_id=path)
        self.train_state = {"step": loaded["train_state.step"], "ntokens_seen": loaded["train_state.ntokens_seen"]}
        self.lr_scheduler_state = {"last_epoch": loaded["lr_scheduler.last_epoch"]}
        steps = [None] * dist.get_world_size()
        dist.all_gather_object(steps, {int(state["step"]) for state in states}, group=gloo_group())
        steps = set().union(*steps)
        assert len(steps) == 1, f"{path}'s adapter parameters disagree on the AdamW step: {steps}"
        return replace(self._run, lora_alpha=None)


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
        self._trainer, self._args, self._steps_per_rollout = trainer, args, steps_per_rollout
        self.slots = _torchtitan_adapter_slots(trainer, args)
        self.members = TorchtitanOptimizerMembers(self.slots, trainer.optimizers.optimizers)
        self.run = self.slots.parametrization(
            base_model=args.hf_checkpoint,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_type=args.lora_type,
            experts_shared_outer_loras=False,
        )

    def _training_state(self, iteration: int | None) -> dict:
        return {
            "iteration": iteration,
            "step": self._trainer.step,
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
        """``--lora-adapter-path``; returns the saved iteration when the source has one.

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
        load_adapter_dir(
            path,
            self.members,
            0,
            self.run,
            self.slots.targets,
            load_optimizer=load_optimizer,
            in_model_loaders_by_kind={AdapterSourceKind.TORCHTITAN_CHECKPOINT: loader},
        )
        training_state = read_training_state(path)
        if loader.train_state is not None:
            self._trainer.engine.load_state_dict(loader.train_state)
            load_lr_schedulers(self._trainer.engine.optim.lr_schedulers, loader.lr_scheduler_state)
            # a torchtitan DCP records steps, not the rollout; miles saved it after a whole rollout's steps
            completed_rollouts = self._trainer.step // self._steps_per_rollout
            training_state = self._training_state(completed_rollouts - 1 if completed_rollouts else None)
        elif training_state is not None and "lr_scheduler" in training_state:
            load_lr_schedulers(self._trainer.engine.optim.lr_schedulers, training_state["lr_scheduler"])
            self._trainer.engine.num_completed_steps = training_state["step"]
        elif training_state is not None and training_state["iteration"] is not None:
            # written by a Megatron run, whose scheduler state does not apply here: resume the schedule at its rollout
            step = (training_state["iteration"] + 1) * self._steps_per_rollout
            load_lr_schedulers(self._trainer.engine.optim.lr_schedulers, {"last_epoch": step})
            self._trainer.engine.num_completed_steps = step
        if self._args.lora_adapter_convert_save is not None:
            destination = Path(self._args.lora_adapter_convert_save)
            save_and_commit_adapter_dir(
                destination.parent,
                destination.name,
                self.members,
                0,
                self.run,
                self.slots.targets,
                training_state=training_state or self._training_state(None),
            )
        return None if training_state is None else training_state["iteration"]
