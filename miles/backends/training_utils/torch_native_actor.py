"""The RL loop for backends whose model is a set of torch modules (FSDP, torchtitan).

A subclass builds its model in ``init`` and fills in the provider surface
declared on the class; the rollout step, the weight push and the host offload
are implemented here once. The backend's training step itself is reached
through a ``StepRunner``.
"""

import functools
import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import PretrainedConfig, PreTrainedTokenizerBase

from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput
from miles.backends.training_utils.ci_utils import check_grad_norm
from miles.backends.training_utils.data import DataIterator, get_batch, get_data_iterator, get_rollout_data
from miles.backends.training_utils.log_utils import (
    aggregate_forward_results,
    aggregate_train_losses,
    log_rollout_data,
    log_train_step,
)
from miles.backends.training_utils.loss import compute_advantages_and_returns, get_log_probs_and_entropy, loss_function
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.step_runner import StepRunner
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.ray.train_actor import TrainRayActor
from miles.utils import train_metric_utils
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.flops_utils import flops_args_from_hf_config, fwd_tflops_per_gpu
from miles.utils.memory_utils import clear_memory, move_optimizer_state, print_memory
from miles.utils.profile_utils import TrainProfiler
from miles.utils.ray_utils import Box
from miles.utils.timer import inverse_timer, timer

if TYPE_CHECKING:
    from miles.ray.rollout.inference_controller import UpdatableEngines

logger = logging.getLogger(__name__)

FORWARD_ONLY_KEYS = [
    "tokens",
    "loss_masks",
    "multimodal_train_inputs",
    "total_lengths",
    "response_lengths",
    "max_seq_lens",
]
TRAIN_KEYS = FORWARD_ONLY_KEYS + [
    "log_probs",
    "advantages",
    "returns",
    "ref_log_probs",
    "rollout_log_probs",
]


class TorchNativeTrainRayActor(TrainRayActor):
    routing_replay: ModuleType
    model_parts: Sequence[torch.nn.Module]
    optimizers: Sequence[torch.optim.Optimizer]
    weight_updater: WeightUpdater
    prof: TrainProfiler
    hf_config: PretrainedConfig
    tokenizer: PreTrainedTokenizerBase
    ref_runner: StepRunner | None = None
    align_token_side_channel: Callable[[torch.Tensor, int], torch.Tensor] | None = None

    def step_runner(self) -> StepRunner:
        raise NotImplementedError

    def ref_context(self) -> AbstractContextManager:
        return nullcontext()

    def after_rollout(self, rollout_id: int, rollout_data: dict) -> None:
        pass

    @property
    def train_parallel_config(self) -> dict:
        return {"dp_size": get_parallel_state().intra_dp.size}

    def _get_parallel_config(self) -> dict:
        return self.train_parallel_config

    def _build_weight_updater(self, model, iterator_factory: Callable) -> WeightUpdater:
        model_name = self.args.model_name
        if model_name is None:
            model_name = type(self.hf_config).__name__.lower()
        return WeightUpdater(
            self.args,
            model,
            weights_getter=lambda: None,
            model_name=model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            iterator_factory=iterator_factory,
            parallel_state=get_parallel_state(),
            is_lora=False,
        )

    @functools.cached_property
    def _fwd_tflops(self) -> Callable[[list[int]], float] | None:
        try:
            flops_args = flops_args_from_hf_config(self.hf_config)
        except Exception as e:
            logger.warning(f"MFU will not be reported, {type(self.hf_config).__name__} could not be sized: {e}")
            return None
        return lambda seq_lens: fwd_tflops_per_gpu(seq_lens, flops_args, dist.get_world_size())

    @timer
    def sleep(self) -> None:
        if self.args.offload_train:
            self._move_to("cpu")

    @timer
    def wake_up(self) -> None:
        if self.args.offload_train:
            self._move_to("cuda")

    def _move_to(self, device: str) -> None:
        print_memory(f"before moving the model to {device}")
        for module in self.model_parts:
            module.to(device)
        move_optimizer_state(self.optimizers, device)
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory(f"after moving the model to {device}")

    def train(self, rollout_id: int, rollout_data_ref: Box, witness_info=None, attempt: int = 0) -> TrainStepOutput:
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
            compute_total_fwd_flops=self._fwd_tflops,
        )
        self._heartbeat.bump()
        return TrainStepOutput(outcome=TrainStepOutcome.NORMAL)

    def _train_core(self, rollout_id: int, rollout_data: dict) -> None:
        replay = self.routing_replay
        data_iterators, num_microbatches = get_data_iterator(self.args, self.model_parts, rollout_data)
        assert num_microbatches, f"empty microbatch schedule for micro_batch_size={self.args.micro_batch_size}"
        replay.fill(
            self.args,
            self.model_parts,
            data_iterators,
            num_microbatches,
            rollout_data,
            align=self.align_token_side_channel,
        )
        data_iterator = data_iterators[0]
        runner = self.step_runner()

        if self.ref_runner is not None:
            with replay.stage(replay.FALLTHROUGH), self.ref_context():
                rollout_data.update(self._log_probs(self.ref_runner, data_iterator, num_microbatches, "ref_"))
        with replay.stage(replay.log_prob_stage(self.args)):
            rollout_data.update(self._log_probs(runner, data_iterator, num_microbatches))
        replay.rewind()

        compute_advantages_and_returns(self.args, rollout_data)
        log_rollout_data(rollout_id, self.args, rollout_data)

        with replay.stage(replay.REPLAY_BACKWARD), timer("actor_train"):
            self._optimizer_steps(runner, data_iterator, num_microbatches, rollout_id)
        replay.reset()

        self.prof.step(rollout_id=rollout_id)
        self.after_rollout(rollout_id, rollout_data)

    @torch.no_grad()
    def _log_probs(
        self, runner: StepRunner, data_iterator: DataIterator, num_microbatches: list[int], store_prefix: str = ""
    ) -> dict[str, list[torch.Tensor]]:
        """No-grad pass over the rollout collecting token log probs; entropy only for the actor pass."""
        args = self.args
        forward_store: list[dict] = []
        data_iterator.reset()

        def compute(logits: torch.Tensor, batch: dict) -> dict:
            result = get_log_probs_and_entropy(
                logits=logits,
                args=args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=batch["total_lengths"],
                response_lengths=batch["response_lengths"],
                with_entropy=(store_prefix == ""),
                max_seq_lens=batch.get("max_seq_lens"),
            )
            entry = {f"{store_prefix}log_probs": result["log_probs"]}
            if "entropy" in result:
                entry["entropy"] = result["entropy"]
            return entry

        with timer(f"{store_prefix}log_probs"):
            for microbatches in num_microbatches:
                progress = tqdm(range(microbatches), desc=f"{store_prefix}log_probs", disable=dist.get_rank() != 0)
                batches = self._fetch_batches(
                    self.prof.iterate_train_log_probs(progress), data_iterator, FORWARD_ONLY_KEYS
                )
                forward_store.extend(runner.forward_only_step(batches, compute))

        return aggregate_forward_results(forward_store, data_iterator, args, store_prefix)

    def _optimizer_steps(
        self, runner: StepRunner, data_iterator: DataIterator, num_microbatches: list[int], rollout_id: int
    ) -> None:
        """One optimizer step per entry in ``num_microbatches``; the loss is normalized per whole step."""
        args = self.args
        data_iterator.reset()
        state = get_parallel_state()

        for step_id, microbatches in enumerate(num_microbatches):
            runner.zero_grad()
            progress = tqdm(range(microbatches), desc="actor_train", disable=dist.get_rank() != 0)
            batches = self._fetch_batches(self.prof.iterate_train_actor(progress), data_iterator, TRAIN_KEYS)
            losses_reduced = runner.forward_backward_step(batches, partial(_step_loss, args, microbatches))
            metrics = runner.apply_step()

            if args.ci_test:
                check_grad_norm(
                    args=args,
                    grad_norm=metrics.grad_norm,
                    rollout_id=rollout_id,
                    step_id=step_id,
                    role="actor",
                    rank=state.intra_dp_cp.rank,
                )
            log_train_step(
                args=args,
                loss_dict=aggregate_train_losses(losses_reduced),
                grad_norm=metrics.grad_norm,
                rollout_id=rollout_id,
                step_id=step_id,
                num_steps_per_rollout=len(num_microbatches),
                role="actor",
                extra_metrics=metrics.extra_metrics,
                should_log=state.is_metrics_rank,
            )

    def _fetch_batches(self, progress, data_iterator: DataIterator, keys: list[str]) -> Iterator[dict]:
        for _ in progress:
            yield get_batch(
                data_iterator,
                keys,
                self.args.data_pad_size_multiplier,
                self.args.qkv_format,
                get_position_ids=True,
            )

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


def _step_loss(args, num_microbatches: int, logits: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
    loss, _normalizer, log_dict = loss_function(
        args=args,
        batch=batch,
        num_microbatches=num_microbatches,
        logits=logits,
        apply_megatron_loss_scaling=False,
    )
    return loss, log_dict
