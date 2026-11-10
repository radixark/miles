"""The torchtitan multi-LoRA actor: the Tinker gateway's slot RPCs over torchtitan's multi-slot model."""

import json
from functools import partial
from pathlib import Path

import torch
from torch.distributed.fsdp import FSDPModule
import torch.distributed as dist
from safetensors import SafetensorError

from miles.backends.torchtitan_utils.actor import TorchtitanTrainRayActor
from miles.backends.torchtitan_utils.hf_weight_iterator import TitanHfWeightIterator
from miles.backends.torchtitan_utils.lora_checkpoint import TorchtitanAdapterSlots, TorchtitanSlotOptimizerMembers
from miles.backends.torchtitan_utils.slot_optimizer import TorchtitanSlotOptimizer, step_slot_optimizers
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import (
    check_adapter_source,
    load_adapter_source,
    save_adapter_dir,
    save_external_export,
    save_sampler_export,
)
from miles.backends.training_utils.checkpoint.adapter.adapter_state import gloo_group, raise_if_any_rank_failed
from miles.backends.training_utils.checkpoint.adapter.commit import UncommittedAdapterDir
from miles.backends.training_utils.checkpoint.adapter.source_error import AdapterSourceError
from miles.backends.training_utils.data.rollout import get_batch, get_data_iterator, get_rollout_data
from miles.backends.training_utils.loss.hub.response_layout import response_layout_type
from miles.backends.training_utils.loss.objective import loss_function
from miles.backends.training_utils.metrics.log_utils import aggregate_train_losses
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.utils.hf_utils.config import load_hf_config
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets
from miles.utils.object_store import StoreObjectRef
from miles.utils.tracking_utils.structured_log import with_logs

# the keys a Tinker loss reads, beside what get_batch adds
_SLOT_PASS_KEYS = [
    "tokens",
    "total_lengths",
    "response_lengths",
    "loss_masks",
    "log_probs",
    "advantages",
    "rollout_log_probs",
    "max_seq_lens",
    "loss_weights",
    "target_tokens",
    "sample_indices",
]


class TorchtitanMultiLoRATrainRayActor(TorchtitanTrainRayActor):
    # the gateway serves no rollouts, and each slot's optimizer steps per request at the request's lr: the engine's
    # own schedule never steps
    def _steps_per_rollout(self) -> int:
        return 1

    def _lr_total_steps(self) -> int:
        return 1

    def _init_training_state(self) -> None:
        # adapters reach the engines as sampler exports the gateway publishes, not through a weight update
        self.weight_updater = None
        self.slot_optimizers: dict[int, TorchtitanSlotOptimizer] = {}
        iterator = TitanHfWeightIterator.build(
            self.args,
            self.trainer,
            required_placement=WeightUpdatePlacement(gather_pp=True),
            model_name=None,
            quantization_config=None,
        )
        self.adapter_slots = TorchtitanAdapterSlots(
            self.trainer.model_parts,
            iterator,
            self.trainer.engine.state_dict_adapter,
            get_hf_lora_targets(load_hf_config(self.args.hf_checkpoint).to_dict()),
            multi_lora=True,
        )

    @with_logs
    def forward_backward(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        self._heartbeat.bump()
        return self._slot_pass(batch_id, rollout_data_ref, forward_only=False)

    @with_logs
    def forward_only(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        """Same loss pass as forward_backward, without the backward: the Tinker forward() contract returns the
        requested loss per datum."""
        self._heartbeat.bump()
        return self._slot_pass(batch_id, rollout_data_ref, forward_only=True)

    def _slot_pass(self, batch_id: int, rollout_data_ref: StoreObjectRef, *, forward_only: bool) -> dict:
        """One work unit; gradients add to what the slots already accumulated, until their optim_step."""
        rollout_data, store_get_result = get_rollout_data(
            args=self.args,
            rollout_data_ref=rollout_data_ref,
            output_store_read=self.output_store_read,
            train_parallel_config=self.train_parallel_config,
            sample_device=response_layout_type(self.args).sample_device,
        )
        with store_get_result:
            data_iterators, num_microbatches = get_data_iterator(self.args, self.model_parts, rollout_data)
            assert len(num_microbatches) == 1, "a work unit is a single forward/backward pass"
            data_iterator = data_iterators[0]
            data_iterator.reset()
            # the trainer tallies each pass's loss here; zero_grad, which starts the tally in RL, would drop the
            # slots' accumulated gradients
            self.trainer.engine.loss = torch.zeros((), device=self.trainer.device)
            loss = partial(_slot_loss, self.args, num_microbatches[0])
            logs = []
            for _ in range(num_microbatches[0]):
                batch = get_batch(
                    data_iterator,
                    _SLOT_PASS_KEYS,
                    self.args.data_pad_size_multiplier,
                    self.args.qkv_format,
                    get_position_ids=True,
                )
                self.adapter_slots.route_tokens(batch["adapter_token_counts"])
                if forward_only:
                    with torch.no_grad():
                        logs += self.trainer.run_forward([batch], lambda logits, b: loss(logits, b)[1])
                else:
                    logs += self.trainer.run_forward_backward([batch], loss)
            if forward_only:
                # FSDP keeps the last block (norm, lm_head) gathered after a forward until a backward reshards it;
                # the slot RPCs read every parameter's local shard
                reshard_parameters(self.model_parts)
        self.prof.step(rollout_id=batch_id)
        per_datum = [output for log in logs for output in log["per_datum"]]
        if get_parallel_state().is_pp_last_stage:
            return {"metrics": aggregate_train_losses(logs), "per_datum": per_datum}
        return {"metrics": {}, "per_datum": per_datum}

    @with_logs
    def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        self._heartbeat.bump()
        return step_slot_optimizers(self.slot_optimizers, adam_params_by_slot)

    @with_logs
    def load_slot(self, slot: int, rank: int, alpha: float, source: dict | None = None) -> dict | None:
        """``source``: ``path``, ``load_optimizer`` and the source's ``parametrization_fields``; None starts the slot
        fresh. A source the slot cannot take is refused before the slot changes, as a user error."""
        if source is not None:
            parametrization = self.adapter_slots.parametrization(**source["parametrization_fields"])
            mismatch, checked = self._check_load_source(
                source["path"], parametrization, source["load_optimizer"], rank
            )
            if mismatch is not None:
                return {"error": mismatch, "error_category": "user"}
            # load_state lands in the slot create_model claimed; it is claimed again at the source's alpha
            self._unload(slot)
        # the gateway takes no per-adapter seed
        self.adapter_slots.init(slot, rank, alpha, seed=None)
        self.slot_optimizers[slot] = TorchtitanSlotOptimizer(slot, self.adapter_slots.slot_params(slot))
        if source is not None:
            load_adapter_source(
                Path(source["path"]),
                self._checkpoint_members(),
                slot,
                parametrization,
                self.adapter_slots.targets,
                load_optimizer=source["load_optimizer"],
                in_model_loaders_by_kind={},
                checked=checked,
            )
        return None

    def _check_load_source(self, path: str, run, load_optimizer: bool, rank: int):
        """Collective: why ``path`` cannot load into a slot of rank ``rank`` (every rank gets the same answer) or
        None, and on rank 0 the checked source."""
        shapes_by_rank = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(self.adapter_slots.global_shapes(rank), shapes_by_rank, dst=0, group=gloo_group())
        checked, mismatch, server_error = None, None, None
        if dist.get_rank() == 0:
            try:
                checked = check_adapter_source(
                    Path(path),
                    run,
                    self.adapter_slots.targets,
                    load_optimizer=load_optimizer,
                    run_shapes={key: shape for shapes in shapes_by_rank for key, shape in shapes.items()},
                    head_rows=self.adapter_slots.head_rows,
                )
                if checked.kind.is_read_into_model():
                    mismatch = f"{path} is a {checked.kind.value} source, which this backend cannot load"
            except _SOURCE_ERRORS as error:
                mismatch = f"{path} cannot load into this model: {error}"
            except Exception as error:
                # storage faults (EIO, ESTALE) and bugs are the gateway's
                server_error = error
        raise_if_any_rank_failed(server_error)
        verdict = [mismatch]
        dist.broadcast_object_list(verdict, src=0, group=gloo_group())
        return verdict[0], (None if verdict[0] is not None else checked)

    def _checkpoint_members(self) -> TorchtitanSlotOptimizerMembers:
        return TorchtitanSlotOptimizerMembers(self.adapter_slots, self.slot_optimizers)

    @with_logs
    def save_slot(
        self, slot: int, root: str, name: str, tenant_digest: str, parametrization_fields: dict
    ) -> UncommittedAdapterDir:
        """Write the slot's adapter state under ``root``, uncommitted; the gateway commits it."""
        return save_adapter_dir(
            Path(root),
            name,
            self._checkpoint_members(),
            slot,
            self.adapter_slots.parametrization(**parametrization_fields),
            tenant_digest=tenant_digest,
        )

    @with_logs
    def export_slot(
        self, slot: int, root: str, name: str, tenant_digest: str, parametrization_fields: dict
    ) -> UncommittedAdapterDir:
        """Write the slot's sampler export under ``root``, uncommitted; the gateway commits it."""
        self._heartbeat.bump()
        parametrization = self.adapter_slots.parametrization(**parametrization_fields)
        assert parametrization.lora_rank == self.adapter_slots.lora_rank(
            slot
        ), "the gateway and the slot disagree on rank"
        return save_sampler_export(
            root=Path(root),
            name=name,
            members=self.adapter_slots,
            slot=slot,
            parametrization=parametrization,
            targets=self.adapter_slots.targets,
            tenant_digest=tenant_digest,
            lora_dropout=self.args.lora_dropout,
        )

    @with_logs
    def export_external_slot(
        self,
        slot: int,
        root: str,
        name: str,
        tenant_digest: str,
        parametrization_fields: dict,
        age_encryption_recipients: list[str],
    ) -> UncommittedAdapterDir:
        """Write the slot's official Tinker export under ``root``, uncommitted; the gateway commits it."""
        self._heartbeat.bump()
        parametrization = self.adapter_slots.parametrization(**parametrization_fields)
        assert parametrization.lora_rank == self.adapter_slots.lora_rank(
            slot
        ), "the gateway and the slot disagree on rank"
        return save_external_export(
            Path(root),
            name,
            self._checkpoint_members(),
            slot,
            parametrization,
            self.adapter_slots.targets,
            tenant_digest=tenant_digest,
            age_encryption_recipients=age_encryption_recipients,
            lora_dropout=self.args.lora_dropout,
        )

    @with_logs
    def unload_slot(self, slot: int) -> dict | None:
        self._unload(slot)
        return None

    def _unload(self, slot: int) -> None:
        """The slot's optimizer and gradients die with it."""
        self.slot_optimizers.pop(slot).zero_grads()
        self.adapter_slots.clear(slot)


def reshard_parameters(model_parts) -> None:
    for part in model_parts:
        for module in part.modules():
            if isinstance(module, FSDPModule):
                module.reshard()


def _slot_loss(args, num_microbatches: int, logits: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
    loss, _normalizer, log_dict = loss_function(
        args=args, batch=batch, num_microbatches=num_microbatches, logits=logits, apply_megatron_loss_scaling=False
    )
    return loss, log_dict


# a source the user named that is absent, unreadable, malformed or refused; any other failure is the gateway's
_SOURCE_ERRORS = (
    AdapterSourceError,
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
    PermissionError,
    json.JSONDecodeError,
    SafetensorError,
)
