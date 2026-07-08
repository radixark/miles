import logging
import random
import socket
from argparse import Namespace
from contextlib import nullcontext
from typing import TYPE_CHECKING

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from torch_memory_saver import torch_memory_saver

from miles.ray.train_actor import TrainRayActor
from miles.utils import train_dump_utils
from miles.utils.argparse_utils import inplace_modify_args
from miles.utils.audit_utils.event_logger.logger import event_logger_context
from miles.utils.audit_utils.witness.allocator import WitnessInfo
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group, init_process_group
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.hf_config import load_hf_config
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.processing_utils import load_tokenizer
from miles.utils.ray_utils import Box
from miles.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from miles.utils.replay_base import all_replay_managers, routing_replay_manager
from miles.utils.test_utils.ft_test_actions import FTTestActionActorExecutor
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils.structured_log import with_logs
from miles.utils.tracking_utils.tracking import init_tracking
from miles.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from ..training_utils.data import DataIterator, get_data_iterator, get_rollout_data, sync_actor_critic_data
from ..training_utils.log_utils import log_cpu_memory, log_perf_data, log_rollout_data
from ..training_utils.loss import (
    compute_advantages_and_returns,
    get_log_probs_and_entropy,
    get_values,
    log_train_advantage_computation_event,
)
from ..training_utils.parallel import get_parallel_state
from ..training_utils.replay_data import fill_replay_data, register_replay_list_sequential
from .checkpoint import load_checkpoint
from .ft.checkpoint_transfer import recv_ckpt
from .ft.checkpoint_transfer import send_ckpt as _send_ckpt
from .ft.in_memory_checkpoint import InMemoryCheckpointManager
from .ft.indep_dp import reconfigure_indep_dp_group
from .initialize import init, is_first_replica_megatron_main_rank
from .lora_utils import is_lora_enabled
from .model import TrainStepOutcome, forward_only, initialize_model_and_optimizer, save, train
from .parallel import verify_megatron_parallel_state
from .replay_utils import register_replay_list_moe
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_distributed.broadcast import UpdateWeightFromDistributed
from .update_weight.update_weight_from_distributed.p2p import UpdateWeightP2P
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

if TYPE_CHECKING:
    from miles.ray.rollout.rollout_manager import EnginesAndLock

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class MegatronTrainRayActor(TrainRayActor):
    @with_logs
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        *,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
        recv_ckpt_src_rank: int | None = None,
        indep_dp_info: IndepDPInfo,
    ) -> int | None:
        monkey_patch_torch_dist()

        super().init(args, role, with_ref, with_opd_teacher=with_opd_teacher)

        for m in all_replay_managers:
            m.register_replay_list_func = register_replay_list_sequential
        routing_replay_manager.register_replay_list_func = register_replay_list_moe

        init(
            args,
            indep_dp_store_addr=self._indep_dp_store_addr,
            indep_dp_info=indep_dp_info,
        )

        self._ft_test_action_executor = FTTestActionActorExecutor.from_args(
            args,
            cell_index=indep_dp_info.cell_index,
            num_cells=indep_dp_info.num_cells,
            rank=self._rank,
        )

        if args.dumper_enable:
            from sglang.srt.debug_utils.dumper import dumper

            dumper.apply_source_patches()

        self._is_first_replica_megatron_main_rank = is_first_replica_megatron_main_rank()

        if self._is_first_replica_megatron_main_rank:
            init_tracking(args, primary=False)

        unsupported = {"train_actor", "train_log_probs"} & set(args.profile_target)
        if unsupported and args.use_pytorch_profiler:
            raise NotImplementedError(
                f"--profile-target {' '.join(sorted(unsupported))} is not supported for Megatron backend"
            )
        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = load_hf_config(args.hf_checkpoint)
                self.tokenizer = load_tokenizer(
                    self.args.hf_checkpoint, chat_template_path=self.args.chat_template_path, trust_remote_code=True
                )
            dist.barrier(group=get_gloo_group())

        self.train_parallel_config = {} if args.indep_dp else {"dp_size": get_parallel_state().intra_dp.size}
        dist.barrier(group=get_gloo_group())

        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                # --train-memory-margin-bytes can tune this
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        if self.args.debug_rollout_only:
            return 0

        if role == "critic":
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters
        else:
            for m in all_replay_managers:
                m.enabled = getattr(self.args, f"use_{m.name}_replay", False)
                m.enable_check_replay_result = m.enabled and self.args.ci_test

        checkpointing_context = None
        if recv_ckpt_src_rank is not None:
            ckpt_manager = recv_ckpt(
                indep_dp=get_parallel_state().indep_dp,
                src_rank=recv_ckpt_src_rank,
            )
            checkpointing_context = {"local_checkpoint_manager": ckpt_manager}
        elif args.non_persistent_ckpt_type == "local":
            checkpointing_context = {"local_checkpoint_manager": InMemoryCheckpointManager()}

        heal_load_overrides: dict[str, object] = (
            dict(no_load_optim=False, no_load_rng=False, finetune=False) if recv_ckpt_src_rank is not None else {}
        )
        with inplace_modify_args(args, heal_load_overrides):
            (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
                args, role, checkpointing_context=checkpointing_context
            )

        parallel_state = get_parallel_state()
        if parallel_state.cp.size > 1:
            from miles_plugins.models.cp_utils import detect_and_setup_hybrid_cp

            for model_chunk in self.model:
                detect_and_setup_hybrid_cp(
                    model_chunk, parallel_state.cp.group, parallel_state.cp.rank, parallel_state.cp.size
                )

        verify_megatron_parallel_state(self.model)

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return

        start_rollout_id = loaded_rollout_id + 1

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=args.megatron_to_hf_mode == "raw",
                translate_gpu_to_cpu=not self.args.enable_weights_backuper,
            ),
            single_tag=None if args.enable_weights_backuper else "actor",
        )
        self._active_model_tag: str | None = "actor"
        if self._enable_weight_backup:
            self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        # Load teacher model for Megatron-based on-policy distillation
        if with_opd_teacher:
            self.load_other_checkpoint("teacher", args.opd_teacher_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            self.args.vocab_size = self.tokenizer.vocab_size

        if self.args.colocate:
            update_weight_cls = UpdateWeightFromTensor
        else:
            if self.args.update_weight_transfer_mode == "broadcast":
                update_weight_cls = UpdateWeightFromDistributed
            elif self.args.update_weight_transfer_mode == "disk-delta":
                # Lazy import: keeps the delta deps (numpy/zstandard/xxhash) off the other paths.
                from .update_weight.update_weight_from_distributed.delta import UpdateWeightFromDiskDelta

                update_weight_cls = UpdateWeightFromDiskDelta
            else:
                update_weight_cls = UpdateWeightP2P
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            is_lora=is_lora_enabled(args),
        )

        # empty cache after initialization
        clear_memory()

        self._switch_model("actor")
        if self.args.offload_train:
            self.sleep()

        self.rollout_data_postprocess = None
        if (x := self.args.rollout_data_postprocess_path) is not None:
            from miles.utils.misc import load_function

            self.rollout_data_postprocess = load_function(x)

        self.prof.on_init_end()

        return start_rollout_id

    @with_logs
    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        should_log_cpu_memory = is_first_replica_megatron_main_rank() and hasattr(self, "_last_rollout_id")

        destroy_process_groups()

        tag = "default" if is_lora_enabled(self.args) else None
        torch_memory_saver.pause(tag=tag)

        print_memory("after offload model")

        if should_log_cpu_memory:
            log_cpu_memory(self._last_rollout_id, self.args, "after_offload_train")

    @with_logs
    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        tag = "default" if is_lora_enabled(self.args) else None
        torch_memory_saver.resume(tag=tag)

        clear_memory()
        reload_process_groups()
        print_memory("after wake_up model")

    @property
    def _enable_weight_backup(self) -> bool:
        """Weight backup is only needed for CPU-side model switching or colocated tensor weight sync."""
        return self.with_ref or self.with_opd_teacher or self.args.keep_old_actor or self.args.colocate

    def _switch_model(self, target_tag: str) -> None:
        if not self._enable_weight_backup:
            return
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def _set_replay_stage(self, stage: str) -> None:
        for m in all_replay_managers:
            m.stage = stage

    @with_logs
    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        rollout_id: int,
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                rollout_id=rollout_id,
                store_prefix=store_prefix,
            )

    @with_logs
    @event_logger_context(
        lambda _self, rollout_id, rollout_data_ref, witness_info, attempt: dict(rollout_id=rollout_id, attempt=attempt)
    )
    def train(
        self,
        rollout_id: int,
        rollout_data_ref: Box,
        witness_info: WitnessInfo | None,
        attempt: int,
    ) -> TrainStepOutcome:
        self._heartbeat.bump()
        self._last_rollout_id = rollout_id
        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = get_rollout_data(self.args, rollout_data_ref, witness_info=witness_info)
            if self.args.debug_rollout_only:
                log_rollout_data(rollout_id, self.args, rollout_data)
                return TrainStepOutcome.NORMAL

        if self.role == "critic":
            return self.train_critic(rollout_id, rollout_data)
        else:
            return self.train_actor(rollout_id, rollout_data, witness_info=witness_info, attempt=attempt)

    @with_logs
    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> TrainStepOutcome:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                rollout_id=rollout_id,
            )
        )

        if rollout_id >= self.args.num_critic_only_steps:
            sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)

        compute_advantages_and_returns(self.args, rollout_data)

        self.args.loss_type = "value_loss"
        train_step_outcome: TrainStepOutcome = train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
            witness_info=None,
            attempt=0,
        )

        self._heartbeat.bump()
        return train_step_outcome

    def _use_rollout_replay(self, m) -> bool:
        return getattr(self.args, f"use_rollout_{m.name}_replay", False)

    @with_logs
    def train_actor(
        self, rollout_id: int, rollout_data: RolloutBatch, *, witness_info: WitnessInfo | None, attempt: int
    ) -> TrainStepOutcome:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

        for m in all_replay_managers:
            if self._use_rollout_replay(m):
                fill_replay_data(
                    args=self.args,
                    models=self.model,
                    data_iterator=data_iterator,
                    num_microbatches=num_microbatches,
                    rollout_data=rollout_data,
                    data_key=m.data_key,
                    replay_list=m.replays,
                    register_replay_list_func=m.register_replay_list_func,
                    if_sp_region=m.if_sp_region,
                    indices_are_token_positions=m.replay_indices_are_token_positions,
                )

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    self._set_replay_stage("fallthrough")
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            rollout_id=rollout_id,
                            store_prefix="ref_",
                        )
                    )
                # Forward teacher model to get teacher_log_probs for Megatron-based OPD
                if "teacher" in self.weights_backuper.backup_tags:
                    self._set_replay_stage("fallthrough")
                    self._switch_model("teacher")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            rollout_id=rollout_id,
                            store_prefix="teacher_",
                        )
                    )
                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                    for m in all_replay_managers:
                        if m.enabled:
                            if self._use_rollout_replay(m):
                                m.stage = "replay_forward"
                            else:
                                m.stage = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            rollout_id=rollout_id,
                            store_prefix="",
                        )
                    )
                    for m in all_replay_managers:
                        if self._use_rollout_replay(m):
                            m.clear_all_forward()

                if self.args.use_critic:
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)
                log_train_advantage_computation_event(rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data)

            # Train
            self._set_replay_stage("replay_backward")
            with timer("actor_train"):
                train_step_outcome = train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                    witness_info=witness_info,
                    attempt=attempt,
                    ft_test_action_executor=self._ft_test_action_executor,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        for m in all_replay_managers:
            if m.enabled:
                m.clear_all()

        if train_step_outcome == TrainStepOutcome.NORMAL:
            # update the cpu actor weight to the latest model
            if self._enable_weight_backup:
                self.weights_backuper.backup("actor")
            else:
                torch.cuda.synchronize()

            # Update ref model if needed
            if (
                self.args.ref_update_interval is not None
                and (rollout_id + 1) % self.args.ref_update_interval == 0
                and "ref" in self.weights_backuper.backup_tags
            ):
                with timer("ref_model_update"):
                    if is_first_replica_megatron_main_rank():
                        logger.info(f"Updating ref model at rollout_id {rollout_id}")
                    self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args, extra_metrics=self.weight_updater.pop_metrics())

        self._heartbeat.bump()
        return train_step_outcome

    @with_logs
    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        self._heartbeat.bump()
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving.
        if self.args.offload_train:
            reload_process_groups()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from miles.backends.megatron_utils.model import save_hf_model

            save_hf_model(self.args, rollout_id, self.model)

        if self.args.offload_train:
            destroy_process_groups()

    @with_logs
    @timer
    def update_weights(self, info: "EnginesAndLock") -> None:
        self._heartbeat.bump()
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        rollout_engines = info.rollout_engines
        rollout_engine_lock = info.rollout_engine_lock
        has_new_engines = info.has_new_engines
        engine_gpu_counts = info.engine_gpu_counts
        engine_gpu_offsets = info.engine_gpu_offsets
        del info

        if self.args.offload_train:
            reload_process_groups()

        if has_new_engines or not self.weight_updater.is_rollout_engines_fresh():
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_has_new_engines.remote())

        if self.args.debug_skip_weight_update:
            if dist.get_rank() == 0:
                logger.warning("Skipping actor-to-rollout weight update because " "--debug-skip-weight-update is set.")
            if self.args.offload_train:
                destroy_process_groups()
            return

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0 and not is_lora_enabled(self.args):
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if self.args.offload_train:
            destroy_process_groups()

    @with_logs
    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        # load_checkpoint reads self.args.ckpt_step to pick which iteration to load.
        # Temporarily override it for ref/teacher loads, then restore after the load below.
        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step

        if model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        if model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    @with_logs
    def connect_actor_critic(
        self,
        actor_handle: ActorHandle | None = None,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        if self.role == "actor":
            master_address = ray.util.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            actor_handle.connect_actor_critic.remote(master_address=master_address, master_port=master_port)

        group_name = "actor_critic"
        world_size = 2
        self._actor_critic_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0 if self.role == "actor" else 1,
            group_name=group_name,
        )

    @with_logs
    def send_ckpt(self, dst_rank: int) -> None:
        # These states are not handled
        assert not self.args.keep_old_actor

        _send_ckpt(
            indep_dp=get_parallel_state().indep_dp,
            model=self.model,
            optimizer=self.optimizer,
            opt_param_scheduler=self.opt_param_scheduler,
            iteration=self._last_rollout_id,
            dst_rank=dst_rank,
        )

    @with_logs
    def reconfigure_indep_dp(self, indep_dp_info: IndepDPInfo) -> None:
        reconfigure_indep_dp_group(
            parallel_state=get_parallel_state(),
            store_addr=self._indep_dp_store_addr,
            indep_dp_info=indep_dp_info,
            megatron_rank=dist.get_rank(),
            megatron_world_size=dist.get_world_size(),
        )
        self.weight_updater.mark_engine_connection_stale()
