import asyncio
import copy
import inspect
import json
import logging
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

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
from miles.ray.specs.train import ACTOR_ROLE, compute_trainer_configs, create_trainer_controller_handle
from miles.ray.wiring import get_backend_capability
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
    call_rollout_fn,
)
from miles.rollout.checkpoint_eval import CheckpointEvalFn, EvalSkip
from miles.rollout.data_source import compute_global_dataset_state_path
from miles.rollout.fully_async_data_buffer import Group
from miles.rollout.fully_async_rollout import compute_fully_async_state_path
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.utils import object_store
from miles.utils.async_utils import maybe_await, run, submit
from miles.utils.audit_utils.event_analyzer import analyzer as event_analyzer
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.event_logger.logger import event_logger_context, get_event_logger
from miles.utils.audit_utils.event_logger.models import TrainerWitnessCohortPayload
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.sample_flow import (
    log_dropped_groups,
    log_dropped_samples,
    record_data_source_issues,
    suppress_drop_logging,
)
from miles.utils.audit_utils.sample_ownership_store import SampleOwnershipEventStore
from miles.utils.data import RolloutDataPack
from miles.utils.environ import use_legacy_rollout_v1
from miles.utils.file_utils import atomic_torch_save, atomic_write_text
from miles.utils.function_registry import load_function
from miles.utils.hf_config import is_complete_hf_export
from miles.utils.http_utils import init_http_client
from miles.utils.init_once import InitOnce, init_once
from miles.utils.logging_utils import configure_logger
from miles.utils.metric_checker import MetricChecker
from miles.utils.multi_lora import EmptyBatchTimeoutError
from miles.utils.timer import timer
from miles.utils.tracking_utils.tracking import init_tracking
from miles.utils.types import Sample
from miles.utils.weight_version import assert_samples_weight_version_sane, assert_weight_version_is_published
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerStillBusyError
from miles.utils.workers.worker_provider.base import BaseWorkerProvider

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


def compute_executor_state_path(directory: str | Path, *, rollout_id: int | None) -> Path:
    return Path(directory) / "rollout" / f"executor_state_{rollout_id}.pt"


def compute_checkpoint_complete_marker_path(directory: str | Path, *, rollout_id: int | None) -> Path:
    return Path(directory) / "rollout" / f"complete_{rollout_id}"


@dataclass(frozen=True)
class LastBatch:
    rollout_id: int
    samples: list[Group]
    stage: Literal["raw", "postprocessed", "delivered"] = "raw"
    metadata: dict[str, Any] | None = None
    train_data: dict[str, Any] | None = None


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
        self._sample_ownership_task: asyncio.Task[None] | None = None
        self._sample_ownership_started_at: datetime | None = None
        self._sample_ownership_store: SampleOwnershipEventStore | None = None
        self._actor_controller: BaseWorkerHandle | None = None
        self._last_batch: LastBatch | None = None
        self._replay: LastBatch | None = None
        self._replay_stage: Literal["raw", "postprocessed", "delivered"] | None = None
        self._replay_train_data: dict[str, Any] | None = None

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
        record_data_source_issues(self.data_source)

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
        checker_error = await self._stop_sample_ownership_checker()
        if not self.use_legacy_rollout_v1 and self.generate_rollout is not None:
            await maybe_await(self.generate_rollout.dispose())
        if (close := getattr(self.data_source, "close", None)) is not None:
            close()
        event_analyzer.run_analysis_from_args(self.args)
        if self._metric_checker is not None:
            self._metric_checker.dispose()
        if isinstance(self.eval_generate_rollout, CheckpointEvalFn):
            await maybe_await(self.eval_generate_rollout.dispose())
        if checker_error is not None:
            raise checker_error

    # -------------------------- data generation -----------------------------

    @event_logger_context(lambda _self, rollout_id, trainer_model_id=None: dict(rollout_id=rollout_id))
    async def get(self, rollout_id: int, trainer_model_id: str | None = None) -> RolloutDataPack:
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
                data, metadata, metrics = await self._get_rollout_data_with_ownership_check(
                    rollout_id=rollout_id,
                    trainer_model_id=trainer_model_id,
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
        processed_samples = data
        drop_context = suppress_drop_logging() if self._replay_stage == "delivered" else nullcontext()
        try:
            with drop_context:
                if self.args.use_critic and rollout_id < self.args.num_critic_only_steps:
                    log_dropped_samples(
                        processed_samples,
                        reason="critic_only_warmup",
                        rollout_id=rollout_id,
                    )
                if self._replay_train_data is not None:
                    data = copy.deepcopy(self._replay_train_data)
                else:
                    data = convert_samples_to_train_data(
                        self.args,
                        data,
                        metadata=metadata,
                        custom_convert_samples_to_train_data_func=self.custom_convert_samples_to_train_data_func,
                        custom_reward_post_process_func=self.custom_reward_post_process_func,
                    )
                sample_indices = data.get("sample_indices")
                if self.args.delay_split_train_data_by_dp:
                    data_ref = object_store.get_instance().put(value=data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
                else:
                    data_ref = split_train_data_by_dp(
                        self.args, data, self._train_parallel_configs_of_model_id[trainer_model_id]
                    )
            if trainer_model_id is None:
                self._record_processed_batch(
                    rollout_id=rollout_id,
                    samples=processed_samples,
                    metadata=metadata,
                    stage="delivered",
                    train_data=data,
                )
        finally:
            self._replay_stage = None
            self._replay_train_data = None
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
        elif trainer_model_id is None and self._replay is not None and self._replay.rollout_id == rollout_id:
            replayed = self._replay
            self._replay = None
            logger.info(f"Replaying the {len(replayed.samples)} groups recorded for rollout {rollout_id}")
            data = copy.deepcopy(replayed.samples)
            self._replay_stage = replayed.stage
            self._replay_train_data = copy.deepcopy(replayed.train_data)
            if replayed.stage != "raw":
                metadata = copy.deepcopy(replayed.metadata)
                assert metadata is not None
            else:
                untrimmed_data = list(data)
                data, metadata = postprocess_rollout_data(
                    self.args,
                    data,
                    train_parallel_config=self._train_parallel_configs_of_model_id[trainer_model_id],
                )
                log_dropped_groups(untrimmed_data, data, reason="trim", rollout_id=rollout_id)
                assert_samples_weight_version_sane(self.args, samples=data)
                self._record_processed_batch(
                    rollout_id=rollout_id,
                    samples=data,
                    metadata=metadata,
                    stage="postprocessed",
                )
            metrics = None
        else:
            if not self.use_legacy_rollout_v1:
                input = RolloutFnTrainInput(
                    rollout_id=rollout_id,
                    weight_version=self._weight_versions_of_model_id.get(trainer_model_id),
                    trainer_model_id=trainer_model_id,
                )
                if isinstance(self.generate_rollout, BaseRolloutFn) and inspect.iscoroutinefunction(
                    self.generate_rollout.__call__
                ):
                    data = await asyncio.wrap_future(submit(self._call_and_record(input)))
                else:
                    data = await asyncio.to_thread(call_rollout_function, self.generate_rollout, input)
                    if trainer_model_id is None:
                        self._record_last_batch(rollout_id=rollout_id, samples=data.samples)
            else:
                data = await asyncio.to_thread(
                    call_rollout_fn, self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False
                )
                if trainer_model_id is None:
                    self._record_last_batch(rollout_id=rollout_id, samples=data.samples)
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
            if trainer_model_id is None:
                self._record_processed_batch(
                    rollout_id=rollout_id,
                    samples=data,
                    metadata=metadata,
                    stage="postprocessed",
                )

        return data, metadata, metrics

    async def _get_rollout_data_with_ownership_check(
        self,
        *,
        rollout_id: int,
        trainer_model_id: str | None,
    ) -> tuple[Any, Any, Any]:
        if self._sample_ownership_task is None:
            return await self._get_rollout_data(rollout_id=rollout_id, trainer_model_id=trainer_model_id)

        rollout_task = asyncio.create_task(
            self._get_rollout_data(rollout_id=rollout_id, trainer_model_id=trainer_model_id)
        )
        done, _ = await asyncio.wait(
            (rollout_task, self._sample_ownership_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if self._sample_ownership_task in done:
            rollout_task.cancel()
            await asyncio.gather(rollout_task, return_exceptions=True)
            await self._sample_ownership_task
            raise AssertionError("the sample ownership checker completed without an error")
        return await rollout_task

    def _start_sample_ownership_checker(self) -> None:
        if not self.args.sample_ownership_check or self._sample_ownership_task is not None:
            return

        [actor_config] = [config for config in compute_trainer_configs(self.args) if config.role == ACTOR_ROLE]
        self._actor_controller = create_trainer_controller_handle(
            self.args,
            capability=get_backend_capability(self.args),
            trainer_id=actor_config.trainer_id,
        )
        self._sample_ownership_started_at = datetime.now(timezone.utc)
        self._sample_ownership_store = SampleOwnershipEventStore(get_event_logger())
        self._sample_ownership_task = asyncio.create_task(self._run_sample_ownership_checker())

    async def _run_sample_ownership_checker(self) -> None:
        assert self._actor_controller is not None
        assert self._sample_ownership_started_at is not None
        interval = self.args.sample_ownership_check_interval_seconds
        while True:
            await asyncio.sleep(interval)
            await self._run_one_sample_ownership_check()

    async def _run_one_sample_ownership_check(self) -> None:
        assert self._actor_controller is not None
        assert self._sample_ownership_started_at is not None
        assert self._sample_ownership_store is not None
        timeout = max(
            self.args.sample_ownership_grace_period_seconds,
            self.args.sample_ownership_check_timeout_seconds,
        )
        payload, now = await self._collect_current_cpu_witness(timeout=timeout)
        analysis_timeout = self.args.sample_ownership_check_timeout_seconds
        async with asyncio.timeout(analysis_timeout):
            await asyncio.to_thread(self._sample_ownership_store.replace_current, payload)
            events = await asyncio.to_thread(self._sample_ownership_store.read_events)
            await asyncio.to_thread(
                event_analyzer.run_sample_ownership_analysis,
                events,
                grace_period=timedelta(seconds=self.args.sample_ownership_grace_period_seconds),
                process_started_at=self._sample_ownership_started_at,
                now=now,
                event_source=str(self.args.save_debug_event_data),
            )

    async def _collect_current_cpu_witness(self, *, timeout: float) -> tuple[TrainerWitnessCohortPayload, datetime]:
        assert self._actor_controller is not None
        async with asyncio.timeout(timeout):
            while True:
                now = datetime.now(timezone.utc)
                try:
                    payload = await self._actor_controller.log_current_cpu_witness(rollout_id=self.rollout_id)
                except WorkerStillBusyError:
                    await asyncio.sleep(min(self.args.sample_ownership_check_interval_seconds, 1.0))
                    continue
                return payload, now

    async def _stop_sample_ownership_checker(self) -> BaseException | None:
        if (task := self._sample_ownership_task) is None:
            return None
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return None
        except BaseException as error:
            return error
        return AssertionError("the sample ownership checker completed without an error")

    async def _call_and_record(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output = await self.generate_rollout(input)
        if input.trainer_model_id is None:
            self._record_last_batch(rollout_id=input.rollout_id, samples=output.samples)
        return output

    def _record_last_batch(self, *, rollout_id: int, samples: list[Group]) -> None:
        self._last_batch = LastBatch(rollout_id=rollout_id, samples=copy.deepcopy(samples))

    def _record_processed_batch(
        self,
        *,
        rollout_id: int,
        samples: list[Group],
        metadata: dict[str, Any],
        stage: Literal["postprocessed", "delivered"],
        train_data: dict[str, Any] | None = None,
    ) -> None:
        self._last_batch = LastBatch(
            rollout_id=rollout_id,
            samples=copy.deepcopy(samples),
            stage=stage,
            metadata=copy.deepcopy(metadata),
            train_data=copy.deepcopy(train_data),
        )

    # -------------------------- checkpointing -----------------------------

    # TODO the train and eval rollout functions will become one object, so one save/load is enough here
    def save(self, rollout_id: int) -> None:
        if self.args.save is not None:
            compute_checkpoint_complete_marker_path(self.args.save, rollout_id=rollout_id).unlink(missing_ok=True)
        run(self._save_state(rollout_id))
        if self.args.save is not None:
            self._assert_saved_checkpoint_state(rollout_id)
            marker = compute_checkpoint_complete_marker_path(self.args.save, rollout_id=rollout_id)
            marker.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                marker,
                json.dumps(self._checkpoint_manifest(rollout_id, directory=self.args.save), sort_keys=True),
            )

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

    def load(self, rollout_id: int | None = None, *, require_complete: bool = False) -> None:
        self._assert_checkpoint_complete(rollout_id, require_complete=require_complete)
        self._load_last_batch(rollout_id)
        self.data_source.load(rollout_id)
        if not self.use_legacy_rollout_v1:
            if self.generate_rollout is not None:
                self.generate_rollout.load(rollout_id)
            if (eval_fn := self.eval_generate_rollout) is not None and eval_fn is not self.generate_rollout:
                eval_fn.load(rollout_id)
        event_logger_checkpoint.restore(self.args)
        self._start_sample_ownership_checker()

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

    def _assert_checkpoint_complete(self, rollout_id: int | None, *, require_complete: bool) -> None:
        if require_complete:
            assert (
                self.args.load is not None and rollout_id is not None and rollout_id >= 0
            ), f"Cannot require complete rollout state under {self.args.load} for rollout {rollout_id}"
        if (load_dir := self.args.load) is None or rollout_id is None or rollout_id < 0:
            return

        directory = Path(load_dir) / "rollout"
        marker = compute_checkpoint_complete_marker_path(load_dir, rollout_id=rollout_id)
        if marker.exists():
            self._assert_checkpoint_state(load_dir, rollout_id, marker=marker)
            return

        found = sorted(path.name for path in directory.glob(f"*_{rollout_id}.pt")) if directory.is_dir() else []
        assert not found, (
            f"{directory} holds {found} for rollout {rollout_id} but no complete_{rollout_id} marker; "
            "the checkpoint was interrupted and cannot be restored safely"
        )
        assert (
            not require_complete
        ), f"the trainer restored rollout {rollout_id}, but {directory} has no complete_{rollout_id} marker"
        logger.warning(f"No rollout state under {directory} for rollout {rollout_id}; nothing to restore")

    def _assert_saved_checkpoint_state(self, rollout_id: int) -> None:
        assert self.args.save is not None
        self._assert_checkpoint_state(self.args.save, rollout_id)

    def _assert_checkpoint_state(self, directory: str | Path, rollout_id: int, marker: Path | None = None) -> None:
        expected = self._checkpoint_manifest(rollout_id)
        manifest = self._checkpoint_manifest(rollout_id, directory=directory)
        if marker is not None:
            try:
                restored_manifest = json.loads(marker.read_text())
            except json.JSONDecodeError as error:
                raise AssertionError(f"{marker} does not contain a valid checkpoint manifest") from error
            contract_fields = (
                "version",
                "rollout_id",
                "data_source_path",
                "files",
                "event_snapshot_enabled",
                "directories",
            )
            assert all(
                restored_manifest.get(field) == expected[field] for field in contract_fields
            ), f"{marker} describes {restored_manifest!r}, expected contract {expected!r}; the checkpoint is corrupt"
            manifest = restored_manifest
        assert (manifest.get("event_snapshot") is not None) == manifest[
            "event_snapshot_enabled"
        ], f"checkpoint manifest event snapshot declaration is inconsistent: {manifest!r}"
        required_files = [Path(directory) / relative for relative in manifest["files"]]
        for path in required_files:
            assert path.is_file(), (
                f"{Path(directory) / 'rollout'} has complete_{rollout_id} but no {path.name}; "
                "the checkpoint is corrupt"
            )
        for relative in manifest["directories"]:
            path = Path(directory) / relative
            assert path.is_dir(), (
                f"{Path(directory) / 'rollout'} has complete_{rollout_id} but no mandatory state at {path}; "
                "the checkpoint is corrupt"
            )
        if (event_snapshot := manifest.get("event_snapshot")) is not None:
            snapshot_path = Path(directory) / event_snapshot["directory"]
            event_logger_checkpoint.validate_event_snapshot(
                snapshot_path,
                history_files=event_snapshot["history_files"],
                has_current=event_snapshot["has_current"],
            )

        data_source_path = compute_global_dataset_state_path(directory, rollout_id=rollout_id)
        data_source_state = torch.load(data_source_path, weights_only=False)
        for field in ("sample_group_index", "sample_index"):
            value = data_source_state.get(field)
            assert (
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
            ), f"{data_source_path} has invalid {field}={value!r}; the checkpoint cannot preserve sample identity"

        executor_state = torch.load(compute_executor_state_path(directory, rollout_id=rollout_id), weights_only=False)
        if (last_batch := executor_state["last_batch"]) is not None:
            for sample in _iter_samples(last_batch.samples):
                for field, value in (("group_index", sample.group_index), ("index", sample.index)):
                    assert isinstance(value, int) and not isinstance(value, bool) and value >= 0, (
                        f"executor replay batch has invalid {field}={value!r}; "
                        "the checkpoint cannot preserve sample identity"
                    )

    def _checkpoint_manifest(self, rollout_id: int, directory: str | Path | None = None) -> dict[str, Any]:
        assert self.args.save is not None or self.args.load is not None
        files = [
            str(Path("rollout") / compute_executor_state_path(".", rollout_id=rollout_id).name),
            str(Path("rollout") / compute_global_dataset_state_path(".", rollout_id=rollout_id).name),
        ]
        if self.args.fully_async:
            files.append(str(Path("rollout") / compute_fully_async_state_path(".", rollout_id=rollout_id).name))
        directories = []
        event_snapshot = None
        if self.args.save_debug_event_data is not None:
            relative_event_snapshot = event_logger_checkpoint.compute_event_snapshot_path(
                checkpoint_root=Path("."),
                iteration=rollout_id,
            )
            directories.append(str(relative_event_snapshot))
            if directory is not None:
                snapshot_path = Path(directory) / relative_event_snapshot
                event_snapshot = {
                    "directory": str(relative_event_snapshot),
                    "history_files": sorted(
                        str(path.relative_to(snapshot_path)) for path in snapshot_path.glob("**/*.jsonl")
                    ),
                    "has_current": (snapshot_path / "sample_ownership_current.json").is_file(),
                }
        return {
            "version": 1,
            "rollout_id": rollout_id,
            "data_source_path": self.args.data_source_path,
            "files": files,
            "event_snapshot_enabled": self.args.save_debug_event_data is not None,
            "directories": directories,
            "event_snapshot": event_snapshot,
        }

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


def _iter_samples(node: list[Any]) -> Iterator[Sample]:
    for item in node:
        if isinstance(item, Sample):
            yield item
        else:
            yield from _iter_samples(item)
