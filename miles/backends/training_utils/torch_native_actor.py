"""The RL loop for backends whose model is a set of torch modules (FSDP, torchtitan).

A subclass builds the model in ``init`` and exposes it through the provider
surface below; the rollout step, the weight push and the host offload are
implemented here once.
"""

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch.distributed as dist

from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput
from miles.backends.training_utils.data import get_data_iterator, get_rollout_data
from miles.backends.training_utils.log_utils import log_rollout_data
from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.offload import offload_to_host, reload_to_device
from miles.backends.training_utils.torch_native_loop import run_log_probs, run_optimizer_steps
from miles.ray.train_actor import TrainRayActor
from miles.utils import train_metric_utils
from miles.utils.flops_utils import flops_args_from_hf_config, fwd_tflops_per_gpu
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.ray_utils import Box
from miles.utils.timer import inverse_timer, timer

if TYPE_CHECKING:
    from miles.ray.rollout.inference_controller import UpdatableEngines

logger = logging.getLogger(__name__)


class TorchNativeTrainRayActor(TrainRayActor):
    """Provider surface, set by the subclass in ``init``:

    ``model_parts`` (modules to offload and to size the data iterator),
    ``optimizers`` (their optimizers), ``weight_updater`` (a WeightUpdater),
    ``prof`` (TrainProfiler), ``hf_config``, ``routing_replay`` (the backend's
    routing-replay module: ``fill``/``stage``/``log_prob_stage``/``rewind``/``reset``
    and the stage constants), ``step_runner()``, and optionally ``ref_runner``,
    ``ref_context()``, ``align_token_side_channel`` and ``after_rollout``.
    """

    routing_replay = None
    ref_runner = None
    align_token_side_channel = None
    _flops_args = None

    def step_runner(self):
        raise NotImplementedError

    def ref_context(self):
        return nullcontext()

    def after_rollout(self, rollout_id: int, rollout_data) -> None:
        pass

    def _init_flops(self, hf_config) -> None:
        try:
            self._flops_args = flops_args_from_hf_config(hf_config)
        except Exception as e:
            self._flops_args = None
            logger.warning(f"MFU will not be reported, {type(hf_config).__name__} could not be sized: {e}")

    @timer
    def sleep(self) -> None:
        if self.args.offload_train:
            offload_to_host(self.model_parts, self.optimizers)

    @timer
    def wake_up(self) -> None:
        if self.args.offload_train:
            reload_to_device(self.model_parts, self.optimizers)

    def train(
        self,
        rollout_id: int,
        rollout_data_ref: Box,
        witness_info=None,
        attempt: int = 0,
    ) -> TrainStepOutput:
        assert witness_info is None and attempt == 0
        self._heartbeat.bump()
        if self.args.offload_train:
            self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            rollout_data, store_get_result = get_rollout_data(self.args, rollout_data_ref, witness_info=None)
            with store_get_result:
                if self.args.debug_rollout_only:
                    return TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
                self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
            compute_total_fwd_flops=(
                (lambda seq_lens: fwd_tflops_per_gpu(seq_lens, self._flops_args, dist.get_world_size()))
                if self._flops_args is not None
                else None
            ),
        )
        self._heartbeat.bump()
        return TrainStepOutput(outcome=TrainStepOutcome.NORMAL)

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        replay = self.routing_replay
        data_iterators, num_microbatches = get_data_iterator(self.args, self.model_parts, rollout_data)
        replay.fill(
            self.args,
            self.model_parts,
            data_iterators,
            num_microbatches,
            rollout_data,
            align=self.align_token_side_channel,
        )
        data_iterator = data_iterators[0]
        assert num_microbatches, f"empty microbatch schedule for micro_batch_size={self.args.micro_batch_size}"

        runner = self.step_runner()

        if self.ref_runner is not None:
            with replay.stage(replay.FALLTHROUGH), self.ref_context():
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

        with replay.stage(replay.log_prob_stage(self.args)):
            rollout_data.update(run_log_probs(self.args, data_iterator, num_microbatches, runner, profiler=self.prof))
        replay.rewind()

        compute_advantages_and_returns(self.args, rollout_data)
        log_rollout_data(rollout_id, self.args, rollout_data)

        with replay.stage(replay.REPLAY_BACKWARD), timer("actor_train"):
            run_optimizer_steps(self.args, rollout_id, data_iterator, num_microbatches, runner, profiler=self.prof)
        replay.reset()

        self.prof.step(rollout_id=rollout_id)
        self.after_rollout(rollout_id, rollout_data)

    @timer
    def update_weights(self, info: "UpdatableEngines") -> int | None:  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return None
        self.weight_updater.reconnect_if_needed(info)
        print_memory("before update_weights")
        self.weight_updater.update_weights()
        print_memory("after update_weights")
        if self.args.ci_test:
            self.weight_updater.verify_engine_version(info.rollout_engines)
        clear_memory()
        return self.weight_updater.weight_version

    def _get_parallel_config(self):
        return self.train_parallel_config
