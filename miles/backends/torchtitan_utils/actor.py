"""TrainRayActor over torchtitan's Trainer.

miles' responsibilities end at three things: build a ``Trainer.Config`` from
miles args and instantiate the trainer (the black box -- model construction,
parallelisms, the PP schedule, HF checkpoint load, optimizers all live in
torchtitan), run the shared RL flow (rollout data in -> reference/actor log
probs -> advantages -> optimizer steps), and stream weights to the rollout
engines. The shared loop drives the trainer through its step-runner adapter;
this class never touches torchtitan internals.
"""

import logging
import random
from argparse import Namespace

import ray
import torch
import torch.distributed as dist

from miles.backends.torchtitan_utils import routing_replay
from miles.backends.torchtitan_utils.parallel import create_titan_parallel_state, parallel_dims_from_config
from miles.backends.torchtitan_utils.trainer import TitanTrainer, build_trainer_config
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

        # The LR schedule's horizon is optimizer steps over the whole run.
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
        # CP is internal to the trainer: it gathers logits back to full length
        # before miles' loss hub sees them (see parallel.py on why the shared
        # helpers are told cp is 1).
        self.trainer.enable_context_parallel_gather()
        set_parallel_state(
            create_titan_parallel_state(
                self.trainer.parallel_dims, is_pp_last_stage=self.trainer.has_last_stage()
            )
        )
        self.train_parallel_config = {"dp_size": get_parallel_state().intra_dp.size}
        # Actor only: registering twice would double the manager's stream list.
        routing_replay.install(self.trainer.model_parts)

        # Tracking must live on the rank that produces the training metrics:
        # the loss (and with it ppo_kl and the train-rollout mismatch numbers)
        # exists only on the last pipeline stage, and the shared log helpers
        # gate on exactly this predicate. Global rank 0 is a FIRST-stage rank
        # under PP, so keying tracking off it loses every train metric.
        state = get_parallel_state()
        # ParallelState reports cp=1 on purpose (see parallel.py), so the shared
        # predicate cannot tell two context-parallel peers apart and both would
        # write every metric. Their values are identical -- the loss is taken on
        # the gathered full sequence -- so one of them reports and the other
        # never initializes tracking, which makes its log calls no-ops.
        cp_mesh = self.trainer.parallel_dims.get_optional_mesh("cp")
        reports = cp_mesh is None or dist.get_rank(cp_mesh.get_group()) == 0
        if reports and state.effective_dp_cp.rank == 0 and state.tp.rank == 0 and state.is_pp_last_stage:
            init_tracking(args, primary=False)

        # Fresh runs fall through to the HF assets load (from_hf via the
        # family's adapter); a resumed run finds the native checkpoint under
        # the trainer's dump folder and restores everything, trainer.step
        # included.
        self.trainer.checkpointer.load()
        start_rollout_id = self.trainer.step // _steps_per_rollout(args)

        # Built after the actor so the two never race for HBM during init; it
        # is CPU-offloaded, so it costs host memory rather than device memory.
        if with_ref:
            self.ref_runner = self._build_ref_runner(args)

        # The shared updater owns transport, session and bucketing; torchtitan
        # supplies only the iterator that turns its shards into HF tensors.
        # weights_getter is None: the iterator reads the trainer's live parts.
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
        """A frozen second trainer for reference log probs.

        The same black box, pointed at the reference checkpoint and
        CPU-offloaded so the two models never both hold HBM. Its optimizer
        never steps, so no optimizer state is ever allocated.
        """
        if not args.ref_load:
            raise ValueError("--ref-load is required to build a torchtitan reference model")
        ref_config = build_trainer_config(
            args, hf_assets_path=args.ref_load, lr_total_steps=1, dump_subdir="ref"
        )
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
        # last_step forces the save regardless of titan's own interval; miles
        # decides the cadence.
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
        # Before unwrapping: fill_replay_data resets every iterator in the list.
        routing_replay.fill(
            self.args,
            self.trainer.model_parts,
            data_iterators,
            num_microbatches,
            rollout_data,
            # The queues describe the sequence the routers actually see, which
            # is the trainer's, not the rollout's: padded under PP, sharded
            # under CP.
            align=self.trainer.align_token_side_channel,
        )
        data_iterator = data_iterators[0]
        assert num_microbatches, f"empty microbatch schedule for micro_batch_size={self.args.micro_batch_size}"

        runner = self.trainer.step_runner()

        if self.ref_runner is not None:
            # The reference model routes on its own weights: replaying the
            # actor's routing into it would make the KL term measure routing
            # rather than the policy.
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

        if self.args.ci_test and info.rollout_engines:
            engine = random.choice(info.rollout_engines)
            engine_version = ray.get(engine.get_weight_version.remote())
            if str(engine_version) != str(self.weight_updater.weight_version):
                raise RuntimeError(
                    f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                )
        clear_memory()

    def _get_parallel_config(self):
        return self.train_parallel_config
