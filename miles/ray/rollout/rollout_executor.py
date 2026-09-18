import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, TypeVar

from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.debug_data import RolloutDataInjectionUtil, load_debug_rollout_data, save_debug_rollout_data
from miles.ray.rollout.eval_fleet import EvalFleetInfo, RolloutExecutorEvalFleet
from miles.ray.rollout.metrics import log_eval_rollout_data, log_eval_skip, log_rollout_data
from miles.ray.rollout.output_snapshotter import _RolloutExecutorOutputSnapshotter
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
from miles.rollout.inference_rollout.compatibility import load_rollout_function
from miles.utils import object_store
from miles.utils.async_utils import maybe_await
from miles.utils.audit_utils.event_analyzer import analyzer as event_analyzer
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.check import completed_actor_steps
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.event_logger.logger import (
    event_logger_context,
    get_event_logger,
    is_event_logger_initialized,
    read_events,
)
from miles.utils.audit_utils.event_logger.models import ExplicitlyDroppedSamplesEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.sample_ownership.recorder import SampleOwnershipRecorder
from miles.utils.data import RolloutDataPack
from miles.utils.environ import use_legacy_rollout_v1
from miles.utils.function_registry import load_function
from miles.utils.hf_config import is_complete_hf_export
from miles.utils.http_utils import init_http_client
from miles.utils.init_once import InitOnce, init_once
from miles.utils.logging_utils import configure_logger
from miles.utils.metric_checker import MetricChecker
from miles.utils.multi_lora import EmptyBatchTimeoutError
from miles.utils.simple_checkpointer import atomic_save_folder
from miles.utils.timer import timer
from miles.utils.tracking_utils.tracking import init_tracking
from miles.utils.weight_version import assert_samples_weight_version_sane, assert_weight_version_is_published
from miles.utils.workers.worker_provider.base import BaseWorkerProvider

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

_DATA_SOURCE_DIRNAME = "data_source"
_GENERATE_ROLLOUT_DIRNAME = "generate_rollout"
_EVAL_GENERATE_ROLLOUT_DIRNAME = "eval_generate_rollout"
_EXECUTOR_DIRNAME = "executor"


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
        self.last_get_rollout_id_of_model_id: dict[str | None, int] = {}
        self._rollout_id_being_served: int | None = None
        self._rollouts_since_publish_of_model_id: dict[str | None, int] = defaultdict(int)
        self._train_parallel_configs_of_model_id: dict[str | None, dict[str, Any]] = {}
        self._router_providers = router_providers
        self._session_server_provider = session_server_provider
        self._inference_controller_provider = inference_controller_provider
        self._output_snapshotter = _RolloutExecutorOutputSnapshotter(args=args)

    @init_once
    async def init(self) -> None:
        args = self.args
        if not args.debug_train_only:
            await resolve_router_addrs(args, router_providers=self._router_providers)
            await wait_session_server_ready(args, provider=self._session_server_provider)

        # TODO make args immutable
        init_tracking(args, primary=False, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
        object_store.init_instance(args, contribute_segment=False)

        init_http_client(args)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)
        SampleOwnershipRecorder.install(
            args=args, data_source=self.data_source, current_rollout_id=self._current_rollout_id
        )

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
        self._log_untrained_snapshots_as_dropped()
        event_analyzer.run_sample_ownership_analysis(args=self.args)
        event_analyzer.run_analysis_from_args(self.args)
        if self._metric_checker is not None:
            self._metric_checker.dispose()
        if isinstance(self.eval_generate_rollout, CheckpointEvalFn):
            await maybe_await(self.eval_generate_rollout.dispose())

    def _log_untrained_snapshots_as_dropped(self) -> None:
        if not self.args.enable_sample_ownership_checker or not is_event_logger_initialized():
            return

        events = read_events(get_event_logger().log_dir, strict=True)
        latest_trained_rollout_id = max((step.rollout_id for step in completed_actor_steps(events)), default=-1)
        dropped_sources = {
            index
            for event in events
            if isinstance(event, ExplicitlyDroppedSamplesEvent)
            for index in event.source_sample_indices
        }
        held_samples = self._output_snapshotter.take_held_samples(after_rollout_id=latest_trained_rollout_id)
        SampleOwnershipRecorder.log_dropped_samples(
            args=self.args,
            samples=[
                sample
                for sample in held_samples
                if (sample.lineage.source_sample_index if sample.lineage is not None else sample.index)
                not in dropped_sources
            ],
            reason="shutdown_prefetched",
        )

    # -------------------------- data generation -----------------------------

    @event_logger_context(lambda _self, rollout_id, trainer_model_id=None: dict(rollout_id=rollout_id))
    async def get(self, rollout_id: int, trainer_model_id: str | None = None) -> RolloutDataPack:
        self.last_get_rollout_id_of_model_id[trainer_model_id] = rollout_id
        self._rollout_id_being_served = rollout_id
        event_analyzer.run_sample_ownership_analysis(args=self.args)
        replay = self._output_snapshotter.get(trainer_model_id=trainer_model_id, rollout_id=rollout_id)
        if replay is None:
            generated = await self._generate_rollout_data(rollout_id=rollout_id, trainer_model_id=trainer_model_id)
            if generated is None:
                return RolloutDataPack(empty_batch_timeout=True)
            data, metadata = generated
            # No await may sit between the generated batch arriving and its capture: a save in that window
            self._output_snapshotter.capture(
                trainer_model_id=trainer_model_id, rollout_id=rollout_id, data=data, metadata=metadata
            )
        else:
            data, metadata = replay.data, replay.metadata

        with SampleOwnershipRecorder.suppress_drop_logging() if replay is not None else nullcontext():
            train_data = convert_samples_to_train_data(
                self.args,
                data,
                metadata=metadata,
                custom_convert_samples_to_train_data_func=self.custom_convert_samples_to_train_data_func,
                custom_reward_post_process_func=self.custom_reward_post_process_func,
            )
            sample_indices = train_data.get("sample_indices")
            if self.args.delay_split_train_data_by_dp:
                data_ref = object_store.get_instance().put(value=train_data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
            else:
                data_ref = split_train_data_by_dp(
                    self.args, train_data, self._train_parallel_configs_of_model_id[trainer_model_id]
                )
            return RolloutDataPack(sample_indices=sample_indices, data_ref=data_ref)

    def _current_rollout_id(self) -> int:
        assert (
            self._rollout_id_being_served is not None
        ), "the data source issued samples before any rollout was requested"
        return self._rollout_id_being_served

    async def _generate_rollout_data(
        self, *, rollout_id: int, trainer_model_id: str | None
    ) -> tuple[list[Group], dict[str, Any]] | None:
        start_time = time.time()
        self._rollouts_since_publish_of_model_id[trainer_model_id] += 1
        assert_weight_version_is_published(
            self.args, rollouts_since_publish=self._rollouts_since_publish_of_model_id[trainer_model_id]
        )
        if (get_buffer_length := getattr(self.data_source, "get_buffer_length", None)) is not None:
            dashboard_hooks.report_data_buffer(get_buffer_length())
        with timer("rollout" if trainer_model_id is None else f"{trainer_model_id}/rollout"):
            try:
                data, metadata, metrics = await self._get_rollout_data(
                    rollout_id=rollout_id, trainer_model_id=trainer_model_id
                )
            except EmptyBatchTimeoutError as e:
                assert self.args.multi_lora, "only the multi-LoRA rollout waits for a non-empty batch"
                logger.warning(f"Rollout {rollout_id} produced no trainable group before the empty-wait timeout: {e}")
                return None
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
        return data, metadata

    async def eval(
        self,
        rollout_id: int,
        hf_dir: str | None = None,
        export_time_seconds: float | None = None,
        require_marker: bool = True,
    ) -> None:
        if self.args.debug_train_only and not self.args.eval_uses_snapshots:
            return

        if self.args.eval_uses_snapshots:
            return await self._eval_checkpoint(rollout_id, hf_dir, export_time_seconds, require_marker)

        with timer("eval_rollout"):
            if not self.use_legacy_rollout_v1:
                result = await maybe_await(self.eval_generate_rollout(RolloutFnEvalInput(rollout_id=rollout_id)))
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
                result = await maybe_await(self.eval_generate_rollout(eval_input))
            except EvalSkip as e:
                return self.report_eval_skip(rollout_id, e.reason)

            data = result.data
            save_debug_rollout_data(self.args, data, rollout_id=rollout_id, evaluation=True)
            extra_metrics = dict(result.metrics or {})
            if (last_get_rollout_id := _single_or_none(self.last_get_rollout_id_of_model_id.values())) is not None:
                extra_metrics["eval/lag_steps"] = max(last_get_rollout_id - rollout_id, 0)
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
                data = await maybe_await(self.generate_rollout(input))
            else:
                data = await asyncio.to_thread(
                    call_rollout_fn, self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False
                )
            metrics = data.metrics
            data = data.samples
            data, metadata = postprocess_rollout_data(
                self.args, data, train_parallel_config=self._train_parallel_configs_of_model_id[trainer_model_id]
            )
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
    # async but never awaits: the RPC layer runs sync methods on a thread, and an await-free coroutine is atomic against the rollout coroutines on this loop
    async def save(self, rollout_id: int) -> None:
        assert self.args.save is not None, "the orchestration only saves when --save is set"

        target = compute_rollout_checkpoint_dir(self.args.save, rollout_id=rollout_id)
        with atomic_save_folder(target) as dir_temp:
            self.data_source.save(dir_temp / _DATA_SOURCE_DIRNAME)
            if not self.use_legacy_rollout_v1:
                if self.generate_rollout is not None:
                    self.generate_rollout.save(dir_temp / _GENERATE_ROLLOUT_DIRNAME)
                if (eval_fn := self.eval_generate_rollout) is not None and eval_fn is not self.generate_rollout:
                    eval_fn.save(dir_temp / _EVAL_GENERATE_ROLLOUT_DIRNAME)
            event_logger_checkpoint.snapshot(self.args, directory=dir_temp / event_logger_checkpoint.SNAPSHOT_DIRNAME)
            self._output_snapshotter.save(dir_temp / _EXECUTOR_DIRNAME)

    # async but never awaits, for the same reason as save
    async def load(self, rollout_id: int) -> None:
        if self.args.load is None:
            logger.warning("no --load: the rollout side starts fresh")
            return
        assert rollout_id >= 0, f"rollout {rollout_id} is not a trained step"

        directory = compute_rollout_checkpoint_dir(self.args.load, rollout_id=rollout_id)
        assert directory.is_dir(), (
            f"the trainer restored rollout {rollout_id}, but {directory} does not exist; a run saved before the "
            f"rollout-side state moved into one directory per rollout cannot resume that state"
        )

        self._output_snapshotter.load(directory / _EXECUTOR_DIRNAME)
        self.data_source.load(directory / _DATA_SOURCE_DIRNAME)
        if not self.use_legacy_rollout_v1:
            if self.generate_rollout is not None:
                self.generate_rollout.load(directory / _GENERATE_ROLLOUT_DIRNAME)
            if (eval_fn := self.eval_generate_rollout) is not None and eval_fn is not self.generate_rollout:
                eval_fn.load(directory / _EVAL_GENERATE_ROLLOUT_DIRNAME)

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


def compute_rollout_checkpoint_dir(directory: str | Path, *, rollout_id: int) -> Path:
    return Path(directory) / "rollout" / str(rollout_id)


_T = TypeVar("_T")


def _single_or_none(xs: Iterable[_T]) -> _T | None:
    xs = list(xs)
    assert len(xs) <= 1, xs
    return xs[0] if xs else None
