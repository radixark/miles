"""TrainRayActor over torchtitan's Trainer.

miles builds the config tree, instantiates the trainer, drives it through the
shared RL loop, and streams weights to the rollout engines. Nothing here
touches torchtitan internals.
"""

import logging
import random
from argparse import Namespace

import ray
import torch.distributed as dist

from miles.backends.torchtitan_utils import routing_replay
from miles.backends.torchtitan_utils.config import build_trainer_config
from miles.backends.torchtitan_utils.parallel import create_titan_parallel_state, parallel_dims_from_config
from miles.backends.torchtitan_utils.trainer import TitanTrainer
from miles.backends.torchtitan_utils.weight_bridge import get_hf_weight_iterator
from miles.backends.training_utils.data import get_data_iterator, get_rollout_data
from miles.backends.training_utils.log_utils import log_rollout_data
from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.model_assets import load_model_assets
from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.backends.training_utils.torch_native_loop import run_log_probs, run_optimizer_steps
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.ray.train_actor import TrainRayActor
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, move_optimizer_state, print_memory
from miles.utils.profile_utils import TrainProfiler
from miles.utils.ray_utils import Box
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils.tracking import init_tracking

logger = logging.getLogger(__name__)


def _steps_per_rollout(args: Namespace) -> int:
    return max(args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size, 1)


class TorchtitanTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        *,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
        recv_ckpt_src_rank: int | None = None,
        indep_dp_info=None,
    ) -> int | None:  # type: ignore[override]
        super().init(args, role, with_ref, with_opd_teacher=with_opd_teacher)

        assert recv_ckpt_src_rank is None, "torchtitan backend does not support checkpoint healing"
        assert not with_opd_teacher, "torchtitan backend does not support on-policy distillation yet"

        config = build_trainer_config(
            args,
            hf_assets_path=args.hf_checkpoint,
            lr_total_steps=args.num_rollout * _steps_per_rollout(args),
            dump_subdir="actor",
        )

        routing_replay.enable(args)

        self.ref_runner = None
        if args.debug_rollout_only:
            set_parallel_state(create_titan_parallel_state(parallel_dims_from_config(config.parallelism)))
            self.train_parallel_config = {"dp_size": get_parallel_state().intra_dp.size}
            return 0

        self.prof = TrainProfiler(args)

        assets = load_model_assets(args)
        self.hf_config = assets.hf_config
        self.tokenizer = assets.tokenizer

        self.trainer = TitanTrainer(config)
        self.trainer.enable_context_parallel_gather()
        set_parallel_state(
            create_titan_parallel_state(self.trainer.parallel_dims, is_pp_last_stage=self.trainer.has_last_stage())
        )
        self.train_parallel_config = {"dp_size": get_parallel_state().intra_dp.size}
        routing_replay.install(self.trainer.model_parts)

        state = get_parallel_state()
        cp_mesh = self.trainer.parallel_dims.get_optional_mesh("cp")
        reports = cp_mesh is None or dist.get_rank(cp_mesh.get_group()) == 0
        if reports and state.effective_dp_cp.rank == 0 and state.tp.rank == 0 and state.is_pp_last_stage:
            init_tracking(args, primary=False)

        self.trainer.checkpointer.load()
        start_rollout_id = self.trainer.step // _steps_per_rollout(args)

        if with_ref:
            self.ref_runner = self._build_ref_runner(args)

        self.weight_updater = WeightUpdater(
            args,
            self.trainer,
            weights_getter=lambda: None,
            model_name=type(self.hf_config).__name__.lower() if args.model_name is None else args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            iterator_factory=get_hf_weight_iterator,
            parallel_state=get_parallel_state(),
            is_lora=False,
        )

        clear_memory()
        if args.offload_train:
            self.sleep()
        self.prof.on_init_end()
        return int(getattr(args, "start_rollout_id", None) or start_rollout_id)

    def _build_ref_runner(self, args: Namespace):
        """A frozen, CPU-offloaded second trainer for reference log probs."""
        if not args.ref_load:
            raise ValueError("--ref-load is required to build a torchtitan reference model")
        ref_config = build_trainer_config(args, hf_assets_path=args.ref_load, lr_total_steps=1, dump_subdir="ref")
        ref_config.training.enable_cpu_offload = True
        ref_trainer = TitanTrainer(ref_config)
        ref_trainer.checkpointer.load()
        for part in ref_trainer.model_parts:
            part.eval()
            part.requires_grad_(False)
        logger.info(f"Built a CPU-offloaded torchtitan reference trainer from {args.ref_load}")
        return ref_trainer.step_runner()

    @timer
    def sleep(self) -> None:
        if not self.args.offload_train:
            return
        print_memory("before offload model")
        for part in self.trainer.model_parts:
            part.cpu()
        move_optimizer_state(self.trainer.optimizers.optimizers, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        if not self.args.offload_train:
            return
        for part in self.trainer.model_parts:
            part.cuda()
        move_optimizer_state(self.trainer.optimizers.optimizers, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up model")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only or self.args.save is None:
            return
        assert not self.args.async_save, "TorchtitanTrainRayActor does not support async_save yet."
        self.trainer.checkpointer.save(self.trainer.step, last_step=True)

    def train(
        self,
        rollout_id: int,
        rollout_data_ref: Box,
        witness_info=None,
        attempt: int = 0,
    ) -> None:
        assert witness_info is None and attempt == 0
        self._heartbeat.bump()
        if self.args.offload_train:
            self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            rollout_data, store_get_result = get_rollout_data(self.args, rollout_data_ref, witness_info=None)
            with store_get_result:
                if self.args.debug_rollout_only:
                    return
                self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

        self._heartbeat.bump()

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        data_iterators, num_microbatches = get_data_iterator(self.args, self.trainer.model_parts, rollout_data)
        routing_replay.fill(
            self.args,
            self.trainer.model_parts,
            data_iterators,
            num_microbatches,
            rollout_data,
            align=self.trainer.align_token_side_channel,
        )
        data_iterator = data_iterators[0]
        assert num_microbatches, f"empty microbatch schedule for micro_batch_size={self.args.micro_batch_size}"

        runner = self.trainer.step_runner()

        if self.ref_runner is not None:
            with routing_replay.stage(routing_replay.FALLTHROUGH):
                rollout_data.update(
                    run_log_probs(
                        self.args,
                        data_iterator,
                        num_microbatches,
                        self.ref_runner,
                        profiler=self.prof,
                        store_prefix="ref_",
                    )
                )

        with routing_replay.stage(routing_replay.log_prob_stage(self.args)):
            rollout_data.update(
                run_log_probs(
                    self.args,
                    data_iterator,
                    num_microbatches,
                    runner,
                    profiler=self.prof,
                )
            )
        routing_replay.rewind()

        compute_advantages_and_returns(self.args, rollout_data)
        log_rollout_data(rollout_id, self.args, rollout_data)

        with routing_replay.stage(routing_replay.REPLAY_BACKWARD), timer("actor_train"):
            run_optimizer_steps(
                self.args,
                rollout_id,
                data_iterator,
                num_microbatches,
                runner,
                profiler=self.prof,
            )
        routing_replay.reset()

        self.prof.step(rollout_id=rollout_id)

    @timer
    def update_weights(self, info) -> None:  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if info.has_new_engines or not self.weight_updater.is_rollout_engines_fresh():
            self.weight_updater.connect_rollout_engines(
                info.rollout_engines,
                info.rollout_engine_lock,
                engine_gpu_counts=info.engine_gpu_counts,
                engine_gpu_offsets=info.engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_has_new_engines.remote())

        print_memory("before update_weights")
        self.weight_updater.update_weights()
        print_memory("after update_weights")
        if dist.get_rank() == 0:
            ray.get(self.rollout_manager.set_weight_version.remote(self.weight_updater.weight_version))

        if self.args.ci_test and info.rollout_engines and self.weight_updater.weight_version > 0:
            engine = random.choice(info.rollout_engines)
            engine_version = ray.get(engine.get_weight_version.remote())
            if str(engine_version) != str(self.weight_updater.weight_version):
                raise RuntimeError(
                    f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                )
        clear_memory()

    def _get_parallel_config(self):
        return self.train_parallel_config
