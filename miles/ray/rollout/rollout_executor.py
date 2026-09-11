import asyncio
import copy
import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.debug_data import RolloutDataInjectionUtil, load_debug_rollout_data, save_debug_rollout_data
from miles.ray.rollout.eval_fleet import EvalFleetInfo, RolloutExecutorEvalFleet
from miles.ray.rollout.metrics import log_eval_rollout_data, log_eval_skip, log_rollout_data
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.router_manager import resolve_router_addrs, wait_session_server_ready
from miles.ray.rollout.train_data_conversion import (
    ROLLOUT_DATA_VALUE_SPEC,
    convert_samples_to_train_data,
    split_train_data_by_dp,
)
from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
    call_rollout_fn,
)
from miles.rollout.checkpoint_eval import CheckpointEvalFn, EvalSkip
from miles.rollout.fully_async_data_buffer import Group
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.utils import object_store
from miles.utils.async_utils import maybe_await, submit
from miles.utils.audit_utils.event_analyzer import analyzer as event_analyzer
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.event_logger.logger import event_logger_context
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.sample_ownership.checker import SampleOwnershipChecker
from miles.utils.audit_utils.sample_ownership.flow import (
    insert_data_source_issue_recorder,
    log_dropped_groups,
    log_dropped_sample_indices,
    suppress_drop_logging,
)
from miles.utils.data import RolloutDataPack
from miles.utils.environ import use_legacy_rollout_v1
from miles.utils.file_utils import atomic_torch_save
from miles.utils.function_registry import load_function
from miles.utils.hf_config import is_complete_hf_export
from miles.utils.http_utils import init_http_client
from miles.utils.init_once import InitOnce, init_once
from miles.utils.logging_utils import configure_logger
from miles.utils.metric_checker import MetricChecker
from miles.utils.multi_lora import EmptyBatchTimeoutError
from miles.utils.timer import timer
from miles.utils.tracking_utils.tracking import init_tracking
from miles.utils.weight_version import assert_samples_weight_version_sane, assert_weight_version_is_published
from miles.utils.workers.worker_provider.base import BaseWorkerProvider

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


def compute_executor_state_path(directory: str | Path, *, rollout_id: int | None) -> Path:
    return Path(directory) / "rollout" / f"executor_state_{rollout_id}.pt"


@dataclass(frozen=True)
class LastBatch:
    rollout_id: int
    train_data: dict[str, Any]


class RolloutExecutor:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(
        self,
        *,
        args,
        router_providers: Sequence[BaseWorkerProvider],
        session_server_provider: BaseWorkerProvider | None,
        inference_controller_provider: BaseWorkerProvider,
    ):
        self._init_once = InitOnce(type(self).__name__)

        configure_logger(args, source=SimpleProcessIdentity(component="rollout_executor"))

        self.args = args
        # set by the training actor after each weight update, keyed by trainer model id (None for one policy)
        self._weight_versions_of_model_id: dict[str | None, int] = {}
        self._rollouts_since_publish_of_model_id: dict[str | None, int] = defaultdict(int)
        self._train_parallel_configs_of_model_id: dict[str | None, dict[str, Any]] = {}
        self._router_providers = router_providers
        self._session_server_provider = session_server_provider
        self._inference_controller_provider = inference_controller_provider
        self._sample_ownership_checker = SampleOwnershipChecker(args=args)
        self._last_batch: LastBatch | None = None
        self._replay: LastBatch | None = None
        self._get_save_lock = asyncio.Lock()

    @init_once
    async def init(self) -> None:
        args = self.args
        if not args.debug_train_only:
            await resolve_router_addrs(args, router_providers=self._router_providers)
            await wait_session_server_ready(args, provider=self._session_server_provider)

        # TODO make args immutable
        init_tracking(args, primary=False, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
        object_store.init_instance(args, contribute_segment=False)

        if not self.args.debug_train_only:
            init_http_client(args)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)
        insert_data_source_issue_recorder(self.data_source)

        self.use_legacy_rollout_v1 = use_legacy_rollout_v1()
        if not self.use_legacy_rollout_v1:
            if self.args.load_debug_rollout_data is not None:
                self.generate_rollout = None
                self.eval_generate_rollout = None
            else:
                input = RolloutFnConstructorInput(args=args, data_source=self.data_source)
                self.generate_rollout = load_rollout_function(input, self.args.rollout_function_path)
                if self.args.eval_function_path == self.args.rollout_function_path:
                    # Reuse the instance so train and eval share one state (and stateful
                    # rollout fns like FullyAsyncRolloutFn are not constructed twice).
                    self.eval_generate_rollout = self.generate_rollout
                else:
                    self.eval_generate_rollout = load_rollout_function(input, self.args.eval_function_path)
        else:
            self.generate_rollout = load_function(self.args.rollout_function_path)
            self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if (x := self.args.custom_reward_post_process_path) is not None:
            self.custom_reward_post_process_func = load_function(x)
        self.custom_convert_samples_to_train_data_func = None
        if (x := self.args.custom_convert_samples_to_train_data_path) is not None:
            self.custom_convert_samples_to_train_data_func = load_function(x)
        if self.generate_rollout is not None:
            logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
            logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        self.rollout_id = -1
        self._eval_lock = asyncio.Lock()
        self._eval_fleet: RolloutExecutorEvalFleet | None = None

        self._metric_checker = MetricChecker.maybe_create(args)

    async def get_init_state(self) -> str:
        return self._init_once.state.value

    # -------------------------- lifecycle -----------------------------

    async def dispose(self) -> None:
        if not self.use_legacy_rollout_v1 and self.generate_rollout is not None:
            await maybe_await(self.generate_rollout.dispose())
        if (close := getattr(self.data_source, "close", None)) is not None:
            close()
        event_analyzer.run_analysis_from_args(self.args)
        if self._metric_checker is not None:
            self._metric_checker.dispose()
        if isinstance(self.eval_generate_rollout, CheckpointEvalFn):
            await maybe_await(self.eval_generate_rollout.dispose())

    # -------------------------- data generation -----------------------------

    @event_logger_context(lambda _self, rollout_id, trainer_model_id=None: dict(rollout_id=rollout_id))
    async def get(self, rollout_id: int, trainer_model_id: str | None = None) -> RolloutDataPack:
        async with self._get_save_lock if trainer_model_id is None else nullcontext():
            self.rollout_id = rollout_id
            await self._sample_ownership_checker.check(rollout_id=rollout_id)
            if trainer_model_id is None and self._replay is not None:
                assert self._replay.rollout_id == rollout_id
                with suppress_drop_logging():
                    result = self._publish_train_data(
                        rollout_id=rollout_id,
                        trainer_model_id=trainer_model_id,
                        data=copy.deepcopy(self._replay.train_data),
                    )
                self._replay = None
                return result
            return await self._get(rollout_id=rollout_id, trainer_model_id=trainer_model_id)

    async def _get(self, rollout_id: int, trainer_model_id: str | None) -> RolloutDataPack:
        start_time = time.time()
        self.rollout_id = rollout_id
        self._rollouts_since_publish_of_model_id[trainer_model_id] += 1
        assert_weight_version_is_published(
            self.args, rollouts_since_publish=self._rollouts_since_publish_of_model_id[trainer_model_id]
        )
        if (get_buffer_length := getattr(self.data_source, "get_buffer_length", None)) is not None:
            dashboard_hooks.report_data_buffer(get_buffer_length())
        with timer("rollout" if trainer_model_id is None else f"{trainer_model_id}/rollout"):
            try:
                data, metadata, metrics = await asyncio.wrap_future(
                    submit(self._get_rollout_data(rollout_id=rollout_id, trainer_model_id=trainer_model_id))
                )
            except EmptyBatchTimeoutError as e:
                assert self.args.multi_lora, "only the multi-LoRA rollout waits for a non-empty batch"
                logger.warning(f"Rollout {rollout_id} produced no trainable group before the empty-wait timeout: {e}")
                return RolloutDataPack(empty_batch_timeout=True)
        save_debug_rollout_data(
            self.args,
            data,
            rollout_id=rollout_id,
            evaluation=False,
            metadata=metadata,
            trainer_model_id=trainer_model_id,
        )
        log_rollout_data(
            rollout_id, self.args, data, metrics, time.time() - start_time, trainer_model_id=trainer_model_id
        )
        data_pack = await asyncio.wrap_future(
            submit(
                self._prepare_train_data(
                    rollout_id=rollout_id,
                    trainer_model_id=trainer_model_id,
                    data=data,
                    metadata=metadata,
                )
            )
        )
        return data_pack

    async def _prepare_train_data(
        self,
        *,
        rollout_id: int,
        trainer_model_id: str | None,
        data: list[Group],
        metadata: dict[str, Any],
    ) -> RolloutDataPack:
        train_data = convert_samples_to_train_data(
            self.args,
            data,
            metadata=metadata,
            custom_convert_samples_to_train_data_func=self.custom_convert_samples_to_train_data_func,
            custom_reward_post_process_func=self.custom_reward_post_process_func,
        )
        data_pack = self._publish_train_data(rollout_id=rollout_id, trainer_model_id=trainer_model_id, data=train_data)
        if trainer_model_id is None:
            self._last_batch = LastBatch(rollout_id=rollout_id, train_data=copy.deepcopy(train_data))
        return data_pack

    def _publish_train_data(
        self, *, rollout_id: int, trainer_model_id: str | None, data: dict[str, Any]
    ) -> RolloutDataPack:
        terminal_drop_reason = (
            "critic_only_warmup"
            if self.args.enable_sample_ownership_checker
            and self.args.use_critic
            and rollout_id < self.args.num_critic_only_steps
            else None
        )
        sample_indices = data.get("sample_indices")
        if self.args.delay_split_train_data_by_dp:
            if terminal_drop_reason is not None:
                log_dropped_sample_indices(
                    list(dict.fromkeys(data["source_sample_indices"])),
                    reason=terminal_drop_reason,
                    rollout_id=rollout_id,
                )
            data_ref = object_store.get_instance().put(value=data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        else:
            data_ref = split_train_data_by_dp(
                self.args,
                data,
                self._train_parallel_configs_of_model_id[trainer_model_id],
                terminal_drop_reason=terminal_drop_reason,
                rollout_id=rollout_id,
            )
        return RolloutDataPack(sample_indices=sample_indices, data_ref=data_ref)

    async def eval(
        self,
        rollout_id: int,
        hf_dir: str | None = None,
        export_time_seconds: float | None = None,
        require_marker: bool = True,
    ) -> None:
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return

        if self.args.eval_uses_snapshots:
            return await self._eval_checkpoint(rollout_id, hf_dir, export_time_seconds, require_marker)

        with timer("eval_rollout"):
            if not self.use_legacy_rollout_v1:
                result = await asyncio.to_thread(
                    call_rollout_function, self.eval_generate_rollout, RolloutFnEvalInput(rollout_id=rollout_id)
                )
            else:
                result = await asyncio.to_thread(
                    call_rollout_fn,
                    self.eval_generate_rollout,
                    self.args,
                    rollout_id,
                    self.data_source,
                    evaluation=True,
                )
        data = result.data
        save_debug_rollout_data(self.args, data, rollout_id=rollout_id, evaluation=True)
        metrics = log_eval_rollout_data(rollout_id, self.args, data, result.metrics)
        if self._metric_checker is not None:
            self._metric_checker.on_eval(metrics)

    async def _eval_checkpoint(
        self, rollout_id: int, hf_dir: str | None, export_time_seconds: float | None, require_marker: bool
    ):
        """Evaluate a snapshot through the checkpoint eval fn (fleet or external
        backend) and log at ``rollout_id``. Every failure degrades to a skipped
        point; the lock serializes pins against a single backend."""
        assert hf_dir is not None, "checkpoint eval requires an HF snapshot dir"
        start_time = time.time()
        async with self._eval_lock:
            if require_marker and not is_complete_hf_export(hf_dir):
                logger.warning(f"Eval snapshot {hf_dir} missing or incomplete, skipping eval {rollout_id}")
                return self.report_eval_skip(rollout_id, "ckpt_missing")

            version = str(rollout_id)
            try:
                state = await self._eval_fleet.pin(hf_dir, version) if self._eval_fleet else None
                eval_input = RolloutFnEvalInput(
                    rollout_id=rollout_id, weight_version=version, hf_dir=hf_dir, generate_state=state
                )
                result = await asyncio.to_thread(call_rollout_function, self.eval_generate_rollout, eval_input)
            except EvalSkip as e:
                return self.report_eval_skip(rollout_id, e.reason)

            data = result.data
            save_debug_rollout_data(self.args, data, rollout_id=rollout_id, evaluation=True)
            extra_metrics = dict(result.metrics or {})
            extra_metrics["eval/lag_steps"] = max(self.rollout_id - rollout_id, 0)
            extra_metrics["eval/duration_seconds"] = time.time() - start_time
            if export_time_seconds is not None:
                extra_metrics["eval/export_time_seconds"] = export_time_seconds
            metrics = log_eval_rollout_data(rollout_id, self.args, data, extra_metrics)
            if self._metric_checker is not None:
                self._metric_checker.on_eval(metrics)

    def report_eval_skip(self, rollout_id: int, reason: str) -> None:
        log_eval_skip(rollout_id, self.args, reason)
        if self.args.ci_test:
            raise RuntimeError(f"CI eval {rollout_id} skipped: {reason}")

    async def _get_rollout_data(self, rollout_id, trainer_model_id: str | None = None):
        if self.args.load_debug_rollout_data is not None:
            data, metadata = load_debug_rollout_data(self.args, rollout_id=rollout_id)
            metrics = None
        else:
            if not self.use_legacy_rollout_v1:
                input = RolloutFnTrainInput(
                    rollout_id=rollout_id,
                    weight_version=self._weight_versions_of_model_id.get(trainer_model_id),
                    trainer_model_id=trainer_model_id,
                )
                data = await asyncio.to_thread(call_rollout_function, self.generate_rollout, input)
            else:
                data = await asyncio.to_thread(
                    call_rollout_fn, self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False
                )
            metrics = data.metrics
            data = data.samples
            untrimmed_data = list(data)
            data, metadata = postprocess_rollout_data(
                self.args, data, train_parallel_config=self._train_parallel_configs_of_model_id[trainer_model_id]
            )
            log_dropped_groups(untrimmed_data, data, reason="trim", rollout_id=rollout_id)
            assert_samples_weight_version_sane(self.args, samples=data)
            if RolloutDataInjectionUtil.should_inject(self.args, rollout_id):
                generated_data = data
                data, metadata = RolloutDataInjectionUtil.load(self.args, rollout_id=rollout_id)
                RolloutDataInjectionUtil.assert_matches_generated(
                    self.args, generated=generated_data, injected=data, rollout_id=rollout_id
                )
                metrics = None
        return data, metadata, metrics

    # -------------------------- checkpointing -----------------------------

    # TODO the train and eval rollout functions will become one object, so one save/load is enough here
    async def save(self, rollout_id: int) -> None:
        async with self._get_save_lock:
            await asyncio.wrap_future(submit(self._save_state(rollout_id)))

    async def _save_state(self, rollout_id: int) -> None:
        self.data_source.save(rollout_id)
        if not self.use_legacy_rollout_v1:
            if self.generate_rollout is not None:
                await maybe_await(self.generate_rollout.save(rollout_id))
            if (eval_fn := self.eval_generate_rollout) is not None and eval_fn is not self.generate_rollout:
                await maybe_await(eval_fn.save(rollout_id))
        self._save_last_batch(rollout_id)
        event_logger_checkpoint.snapshot(self.args, rollout_id)

    def _save_last_batch(self, rollout_id: int) -> None:
        if (save_dir := self.args.save) is None:
            return

        pending = (
            self._last_batch if self._last_batch is not None and self._last_batch.rollout_id > rollout_id else None
        )
        path = compute_executor_state_path(save_dir, rollout_id=rollout_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(path=path, obj={"last_batch": pending})
        logger.info(f"Saved {int(pending is not None)} untrained rollout batch to {path}")

    async def load(self, rollout_id: int | None = None, *, require_complete: bool = False) -> None:
        await asyncio.wrap_future(submit(self._load_state(rollout_id, require_complete=require_complete)))

    async def _load_state(self, rollout_id: int | None, *, require_complete: bool) -> None:
        self._load_last_batch(rollout_id)
        self.data_source.load(rollout_id)
        if not self.use_legacy_rollout_v1:
            if self.generate_rollout is not None:
                await maybe_await(self.generate_rollout.load(rollout_id))
            if (eval_fn := self.eval_generate_rollout) is not None and eval_fn is not self.generate_rollout:
                await maybe_await(eval_fn.load(rollout_id))
        event_logger_checkpoint.restore(self.args)

    def _load_last_batch(self, rollout_id: int | None) -> None:
        if (load_dir := self.args.load) is None:
            return

        path = compute_executor_state_path(load_dir, rollout_id=rollout_id)
        if not path.exists():
            logger.warning(f"No executor state under {path}; a prefetched rollout batch may be lost")
            return

        state = torch.load(path, weights_only=False)
        self._replay = state["last_batch"]
        self._last_batch = copy.deepcopy(self._replay)
        logger.info(f"Loaded {int(self._replay is not None)} untrained rollout batch from {path}")

    # -------------------------- misc APIs -----------------------------

    def get_num_rollout_per_epoch(self) -> int:
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    def set_weight_version(self, weight_version: int, trainer_model_id: str | None = None) -> None:
        # warning instead of assert when use indep_dp ft
        previous = self._weight_versions_of_model_id.get(trainer_model_id)
        if previous is not None and weight_version < previous:
            message = f"Engine weight version went backwards: {previous} -> {weight_version}"
            assert self.args.indep_dp, message
            logger.warning(message)
        self._weight_versions_of_model_id[trainer_model_id] = weight_version
        self._rollouts_since_publish_of_model_id[trainer_model_id] = 0

    def set_train_parallel_config(self, config: dict[str, Any], trainer_model_id: str | None = None) -> None:
        self._train_parallel_configs_of_model_id[trainer_model_id] = config

    async def set_eval_fleet_info(self, eval_fleet_info: EvalFleetInfo | None) -> None:
        if eval_fleet_info is None:
            self._eval_fleet = None
            return

        self._eval_fleet = RolloutExecutorEvalFleet(
            self.args, info=eval_fleet_info, inference_controller_provider=self._inference_controller_provider
        )
