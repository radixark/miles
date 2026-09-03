import asyncio
import logging
import time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import TypeVar, cast

import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.addr_allocator import PortCursors
from miles.ray.rollout.debug_data import RolloutDataInjectionUtil, load_debug_rollout_data, save_debug_rollout_data
from miles.ray.rollout.eval_fleet import EvalFleet
from miles.ray.rollout.metrics import log_eval_rollout_data, log_eval_skip, log_rollout_data
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.rollout_server import RolloutServer, start_rollout_servers
from miles.ray.rollout.router_manager import start_session_server
from miles.ray.rollout.server_cell import get_cell_indexer_of_id_map
from miles.ray.rollout.train_data_conversion import (
    ROLLOUT_DATA_VALUE_SPEC,
    convert_samples_to_train_data,
    split_train_data_by_dp,
)
from miles.ray.train_batch_admission import (
    TrainBatchPublication,
    TrainerAdmissionReceipt,
    TrainerAdmissionStatus,
    TrainerCellCohort,
    TrainerCohort,
    data_ref_ids,
    required_trainer_roles,
)
from miles.ray.utils import Lock
from miles.rollout.base_types import (
    LeasedRolloutFnTrainOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnLifecycle,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
    TrainAdmissionHold,
    TrainBatchLease,
    TrainBatchRollbackReason,
    call_rollout_fn,
)
from miles.rollout.checkpoint_eval import CheckpointEvalFn, EvalSkip
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.utils import object_store
from miles.utils.async_utils import get_async_loop
from miles.utils.audit_utils.event_analyzer import analyzer as event_analyzer
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.process_identity import RolloutManagerProcessIdentity
from miles.utils.environ import use_legacy_rollout_v1
from miles.utils.health_monitor import RolloutHealthMonitor
from miles.utils.hf_config import is_complete_hf_export
from miles.utils.http_utils import init_http_client
from miles.utils.logging_utils import configure_logger
from miles.utils.metric_checker import MetricChecker
from miles.utils.misc import load_function
from miles.utils.timer import timer
from miles.utils.tracking_utils.tracking import init_tracking

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

_MAX_RETAINED_TERMINAL_ADMISSIONS = 64
_T = TypeVar("_T")


async def _release_train_admission_hold(hold: TrainAdmissionHold) -> None:
    hold.release()


async def _await_task_terminal(task: asyncio.Future[_T]) -> _T:
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


async def _await_task_before_cancellation(task: asyncio.Future[_T]) -> _T:
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        try:
            await _await_task_terminal(task)
        except BaseException as terminal_error:
            raise cancellation from terminal_error
        raise


def _discover_rollout_lifecycles(*rollout_fns: object) -> tuple[RolloutFnLifecycle, ...]:
    lifecycles: list[RolloutFnLifecycle] = []
    for rollout_fn in rollout_fns:
        if isinstance(rollout_fn, RolloutFnLifecycle) and all(rollout_fn is not lifecycle for lifecycle in lifecycles):
            lifecycles.append(rollout_fn)
    return tuple(lifecycles)


@dataclass
class _PendingTrainerAdmission:
    lease: TrainBatchLease | None
    data_ref: object_store.StoreObjectRef | list[object_store.StoreObjectRef] | None
    publication: TrainBatchPublication
    status: TrainerAdmissionStatus = TrainerAdmissionStatus.PENDING


def _remove_train_data_refs(
    data_ref: object_store.StoreObjectRef | list[object_store.StoreObjectRef],
) -> None:
    refs = data_ref if isinstance(data_ref, list) else [data_ref]
    store = object_store.get_instance()
    first_error: BaseException | None = None
    for ref in refs:
        try:
            store.remove(ref)
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        event_logger_checkpoint.restore(args)
        configure_logger(args, source=RolloutManagerProcessIdentity())

        self.pg = pg
        self.args = args
        # set by the training actor after each weight update
        self.weight_version: int | None = None
        # TODO make args immutable
        init_tracking(args, primary=False, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
        object_store.init_instance(args, contribute_segment=False)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

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
        self._train_rollout_lifecycle = (
            self.generate_rollout if isinstance(self.generate_rollout, RolloutFnLifecycle) else None
        )
        self._rollout_lifecycles = _discover_rollout_lifecycles(
            self.generate_rollout,
            self.eval_generate_rollout,
        )
        # Rollout lifecycle methods own an event loop separate from the manager
        # actor loop, so concurrent actor calls share one deterministic frontier.
        self._lifecycle_async_loop = get_async_loop() if self._rollout_lifecycles else None
        self._closed_rollout_lifecycles: list[RolloutFnLifecycle] = []
        self._rollout_lifecycles_closing = False
        self._dispose_lock = asyncio.Lock()
        self._manager_resources_disposed = False
        self._next_train_admission_hold_id = 0
        self._train_admission_holds: dict[int, TrainAdmissionHold] = {}
        self.custom_reward_post_process_func = None
        if (x := self.args.custom_reward_post_process_path) is not None:
            self.custom_reward_post_process_func = load_function(x)
        self.custom_convert_samples_to_train_data_func = None
        if (x := self.args.custom_convert_samples_to_train_data_path) is not None:
            self.custom_convert_samples_to_train_data_func = load_function(x)
        if self.generate_rollout is not None:
            logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
            logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.servers: dict[str, RolloutServer] = {}
        else:
            init_http_client(args)
            self.servers = start_rollout_servers(args, pg)
            start_session_server(args)
            dashboard_hooks.register_router(args)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1
        self._manager_incarnation = uuid.uuid4().hex
        self._next_admission_id = 0
        self._pending_admissions: dict[int, _PendingTrainerAdmission] = {}
        self._eval_lock = asyncio.Lock()
        self._eval_fleet = EvalFleet(args, srv=self.servers["eval"]) if args.eval_num_gpus > 0 else None

        self._metric_checker = MetricChecker.maybe_create(args)

        # TODO will be replaced by full ft, thus temporarily leave it without modifications
        self._health_monitors = []
        self._rollout_ft_enabled = self.args.use_fault_tolerance and "rollout" in self.args.ft_components
        self._ci_fault_injection_pending = False
        if not self.args.debug_train_only and self._rollout_ft_enabled:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
            self._ci_fault_injection_pending = self.args.ci_test

        self._data_source_closed = False
        self._event_analysis_completed = False
        self._metric_checker_disposed = False
        self._checkpoint_eval_disposed = False
        self._stopped_health_monitors: list[RolloutHealthMonitor] = []
        self._active_generations = 0
        self._generations_drained = asyncio.Event()
        self._generations_drained.set()

    # -------------------------- lifecycle -----------------------------
    # TODO: may have a `async def init` here later

    def get_router_address(self) -> tuple[str, int]:
        return self.args.sglang_router_ip, self.args.sglang_router_port

    def _submit_lifecycle_coroutine(self, coroutine: Coroutine[object, object, _T]) -> asyncio.Future[_T]:
        if self._lifecycle_async_loop is None:
            raise RuntimeError("Rollout lifecycle event loop is not initialized.")
        concurrent_future = asyncio.run_coroutine_threadsafe(coroutine, self._lifecycle_async_loop.loop)
        return asyncio.wrap_future(concurrent_future)

    def _raise_if_rollout_lifecycles_closing(self) -> None:
        if self._rollout_lifecycles_closing:
            raise RuntimeError("Rollout manager lifecycle is closing.")

    def _ensure_lifecycle_state(self) -> None:
        if not hasattr(self, "_train_rollout_lifecycle"):
            self._train_rollout_lifecycle = None
        if not hasattr(self, "_rollout_lifecycles"):
            self._rollout_lifecycles = ()
        if self._rollout_lifecycles and getattr(self, "_lifecycle_async_loop", None) is None:
            self._lifecycle_async_loop = get_async_loop()
        if not hasattr(self, "_closed_rollout_lifecycles"):
            self._closed_rollout_lifecycles = []
        if not hasattr(self, "_rollout_lifecycles_closing"):
            self._rollout_lifecycles_closing = False
        if not hasattr(self, "_dispose_lock"):
            self._dispose_lock = asyncio.Lock()
        if not hasattr(self, "_manager_resources_disposed"):
            self._manager_resources_disposed = False
        if not hasattr(self, "_next_train_admission_hold_id"):
            self._next_train_admission_hold_id = 0
        if not hasattr(self, "_train_admission_holds"):
            self._train_admission_holds = {}
        if not hasattr(self, "_active_generations"):
            self._active_generations = 0
        if not hasattr(self, "_generations_drained"):
            self._generations_drained = asyncio.Event()
            self._generations_drained.set()
        if not hasattr(self, "_data_source_closed"):
            self._data_source_closed = False
        if not hasattr(self, "_event_analysis_completed"):
            self._event_analysis_completed = False
        if not hasattr(self, "_metric_checker_disposed"):
            self._metric_checker_disposed = False
        if not hasattr(self, "_checkpoint_eval_disposed"):
            self._checkpoint_eval_disposed = False
        if not hasattr(self, "_stopped_health_monitors"):
            self._stopped_health_monitors = []
        if not hasattr(self, "_health_monitors"):
            self._health_monitors = []

    def _begin_generation(self) -> None:
        self._ensure_lifecycle_state()
        self._raise_if_rollout_lifecycles_closing()
        self._active_generations += 1
        self._generations_drained.clear()

    def _end_generation(self) -> None:
        self._active_generations -= 1
        if self._active_generations < 0:
            self._active_generations = 0
            raise RuntimeError("Rollout manager generation accounting underflowed.")
        if self._active_generations == 0:
            self._generations_drained.set()

    async def acquire_train_admission_hold(self) -> int | None:
        self._ensure_lifecycle_state()
        lifecycle = self._train_rollout_lifecycle
        if lifecycle is None:
            return None
        self._raise_if_rollout_lifecycles_closing()
        acquire_task = self._submit_lifecycle_coroutine(lifecycle.acquire_train_admission_hold())
        try:
            hold = await asyncio.shield(acquire_task)
        except asyncio.CancelledError as cancellation:
            try:
                hold = await _await_task_terminal(acquire_task)
                release_task = self._submit_lifecycle_coroutine(_release_train_admission_hold(hold))
                await _await_task_terminal(release_task)
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            raise

        try:
            self._raise_if_rollout_lifecycles_closing()
        except BaseException as closing_error:
            try:
                release_task = self._submit_lifecycle_coroutine(_release_train_admission_hold(hold))
                await _await_task_terminal(release_task)
            except BaseException as cleanup_error:
                raise closing_error from cleanup_error
            raise

        hold_id = self._next_train_admission_hold_id
        self._next_train_admission_hold_id += 1
        self._train_admission_holds[hold_id] = hold
        return hold_id

    async def wait_train_admission_hold(self, hold_id: int | None) -> None:
        self._ensure_lifecycle_state()
        if hold_id is None:
            return
        try:
            hold = self._train_admission_holds[hold_id]
        except KeyError:
            raise RuntimeError(f"Unknown train admission hold {hold_id}.") from None
        wait_task = self._submit_lifecycle_coroutine(hold.wait_terminal())
        await _await_task_before_cancellation(wait_task)

    async def release_train_admission_hold(self, hold_id: int | None) -> None:
        self._ensure_lifecycle_state()
        if hold_id is None:
            return
        try:
            hold = self._train_admission_holds[hold_id]
        except KeyError:
            raise RuntimeError(f"Unknown train admission hold {hold_id}.") from None
        release_task = self._submit_lifecycle_coroutine(_release_train_admission_hold(hold))
        cancellation: asyncio.CancelledError | None = None
        release_error: BaseException | None = None
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError as error:
            cancellation = error
            try:
                await _await_task_terminal(release_task)
            except BaseException as terminal_error:
                release_error = terminal_error
        except BaseException as error:
            release_error = error

        if release_error is not None:
            if cancellation is not None:
                raise cancellation from release_error
            raise release_error
        self._train_admission_holds.pop(hold_id, None)
        if cancellation is not None:
            raise cancellation

    async def dispose(self) -> None:
        self._ensure_lifecycle_state()
        async with self._dispose_lock:
            await self._dispose()

    async def _dispose(self) -> None:
        self._ensure_lifecycle_state()
        self._rollout_lifecycles_closing = True
        if self._manager_resources_disposed:
            return

        cancellation: asyncio.CancelledError | None = None
        generations_task = asyncio.create_task(self._generations_drained.wait())
        try:
            await asyncio.shield(generations_task)
        except asyncio.CancelledError as error:
            cancellation = error
            try:
                await _await_task_terminal(generations_task)
            except BaseException as terminal_error:
                raise cancellation from terminal_error

        # A generated leased result must first be registered as a PR4
        # publication.  Otherwise closing the lifecycle could strand its lease.
        try:
            self._reject_unresolved_admissions("dispose")
        except BaseException as admission_error:
            if cancellation is not None:
                raise cancellation from admission_error
            raise

        close_error: BaseException | None = None
        for lifecycle in self._rollout_lifecycles:
            if any(lifecycle is closed for closed in self._closed_rollout_lifecycles):
                continue
            close_task = self._submit_lifecycle_coroutine(lifecycle.close())
            closed = False
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
                try:
                    await _await_task_terminal(close_task)
                except BaseException as terminal_error:
                    close_error = close_error or terminal_error
                else:
                    closed = True
            except BaseException as error:
                close_error = close_error or error
            else:
                closed = True
            if closed:
                self._closed_rollout_lifecycles.append(lifecycle)

        if close_error is not None:
            if cancellation is not None:
                raise cancellation from close_error
            raise close_error

        self._train_admission_holds.clear()
        cleanup_error: BaseException | None = None
        try:
            self._dispose_resources()
        except BaseException as error:
            cleanup_error = error
        else:
            self._manager_resources_disposed = True

        if cancellation is not None:
            if cleanup_error is not None:
                raise cancellation from cleanup_error
            raise cancellation
        if cleanup_error is not None:
            raise cleanup_error

    def _dispose_resources(self) -> None:
        cleanup_errors: list[BaseException] = []
        if not self._data_source_closed:
            if (close := getattr(getattr(self, "data_source", None), "close", None)) is None:
                self._data_source_closed = True
            else:
                try:
                    close()
                except BaseException as error:
                    cleanup_errors.append(error)
                else:
                    self._data_source_closed = True
        if not self._event_analysis_completed:
            try:
                event_analyzer.run_analysis_from_args(self.args)
            except BaseException as error:
                cleanup_errors.append(error)
            else:
                self._event_analysis_completed = True
        metric_checker = getattr(self, "_metric_checker", None)
        if metric_checker is not None and not self._metric_checker_disposed:
            try:
                metric_checker.dispose()
            except BaseException as error:
                cleanup_errors.append(error)
            else:
                self._metric_checker_disposed = True
        eval_generate_rollout = getattr(self, "eval_generate_rollout", None)
        if isinstance(eval_generate_rollout, CheckpointEvalFn) and not self._checkpoint_eval_disposed:
            try:
                eval_generate_rollout.dispose()
            except BaseException as error:
                cleanup_errors.append(error)
            else:
                self._checkpoint_eval_disposed = True
        for monitor in self._health_monitors:
            if any(monitor is stopped for stopped in self._stopped_health_monitors):
                continue
            try:
                monitor.stop()
            except BaseException as error:
                cleanup_errors.append(error)
            else:
                self._stopped_health_monitors.append(monitor)
        if cleanup_errors:
            raise cleanup_errors[0]

    # -------------------------- data generation -----------------------------

    async def generate(self, rollout_id):
        self._begin_generation()
        try:
            return await self._generate(rollout_id)
        finally:
            self._end_generation()

    async def _generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self._health_monitoring_resume()
        if self.args.ci_test and self._rollout_ft_enabled and rollout_id >= 2:
            self._try_ci_fault_injection()
        dashboard_hooks.register_engines(self.servers)
        if (get_buffer_length := getattr(self.data_source, "get_buffer_length", None)) is not None:
            dashboard_hooks.report_data_buffer(get_buffer_length())
        lease: TrainBatchLease | None = None
        data_ref = None
        publication = None
        try:
            with timer("rollout"):
                if self.args.load_debug_rollout_data is not None:
                    data, metadata = load_debug_rollout_data(self.args, rollout_id=rollout_id)
                    metrics = None
                else:
                    output = await self._get_rollout_output(rollout_id)
                    if isinstance(output, LeasedRolloutFnTrainOutput):
                        lease = output.lease
                        if lease.rollout_id != rollout_id:
                            raise ValueError(
                                f"Leased train output for rollout {rollout_id} carries a lease "
                                f"for rollout {lease.rollout_id}."
                            )
                    data = output.samples
                    metrics = output.metrics
                    data, metadata = postprocess_rollout_data(
                        self.args, data, train_parallel_config=self.train_parallel_config
                    )
                    if RolloutDataInjectionUtil.should_inject(self.args, rollout_id):
                        generated_data = data
                        data, metadata = RolloutDataInjectionUtil.load(self.args, rollout_id=rollout_id)
                        RolloutDataInjectionUtil.assert_matches_generated(
                            self.args, generated=generated_data, injected=data, rollout_id=rollout_id
                        )
                        metrics = None
            save_debug_rollout_data(self.args, data, rollout_id=rollout_id, evaluation=False, metadata=metadata)
            log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
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
                data_ref = split_train_data_by_dp(self.args, data, self.train_parallel_config)
            result = dict(sample_indices=sample_indices, data_ref=data_ref)

            if lease is not None:
                self._ensure_admission_state()
                publication = TrainBatchPublication(
                    manager_incarnation=self._manager_incarnation,
                    admission_id=self._next_admission_id,
                    rollout_id=rollout_id,
                    data_ref_ids=data_ref_ids(data_ref),
                    required_roles=required_trainer_roles(self.args, rollout_id),
                )
                self._pending_admissions[publication.admission_id] = _PendingTrainerAdmission(
                    lease=lease,
                    data_ref=data_ref,
                    publication=publication,
                )
                self._next_admission_id += 1
                result["trainer_admission"] = publication
        except BaseException as handoff_error:
            if lease is not None:
                try:
                    lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
                except BaseException as rollback_error:
                    raise handoff_error from rollback_error
                cleanup_error = None
                if publication is not None:
                    try:
                        self._pending_admissions.pop(publication.admission_id, None)
                    except BaseException as registration_cleanup_error:
                        cleanup_error = registration_cleanup_error
                if data_ref is not None:
                    try:
                        _remove_train_data_refs(data_ref)
                    except BaseException as data_cleanup_error:
                        if cleanup_error is None:
                            cleanup_error = data_cleanup_error
                if cleanup_error is not None:
                    raise handoff_error from cleanup_error
            raise
        return result

    def commit_trainer_admission(
        self,
        publication: TrainBatchPublication,
        receipts: tuple[TrainerAdmissionReceipt, ...],
    ) -> TrainerAdmissionStatus:
        """Commit the source lease after every required trainer role acknowledges it."""
        pending = self._get_pending_admission(publication)
        if pending.status is not TrainerAdmissionStatus.PENDING:
            return pending.status
        self._validate_receipts(pending.publication, receipts)
        if pending.lease is None:
            raise RuntimeError(f"Trainer admission {publication.admission_id} has no source lease.")
        pending.status = TrainerAdmissionStatus.COMMIT_FAILED
        pending.lease.commit()
        pending.status = TrainerAdmissionStatus.COMMITTED
        pending.lease = None
        pending.data_ref = None
        self._trim_terminal_admissions()
        return pending.status

    def rollback_trainer_admission(self, publication: TrainBatchPublication) -> TrainerAdmissionStatus:
        """Settle the source lease before deleting a failed publication."""
        pending = self._get_pending_admission(publication)
        if pending.status is not TrainerAdmissionStatus.PENDING:
            return pending.status
        if pending.lease is None or pending.data_ref is None:
            raise RuntimeError(f"Trainer admission {publication.admission_id} has no source publication.")
        pending.status = TrainerAdmissionStatus.ROLLBACK_FAILED
        pending.lease.rollback(TrainBatchRollbackReason.TRAINER_ADMISSION_FAILED)
        _remove_train_data_refs(pending.data_ref)
        pending.status = TrainerAdmissionStatus.ROLLED_BACK
        pending.lease = None
        pending.data_ref = None
        self._trim_terminal_admissions()
        return pending.status

    def get_trainer_admission_status(self, publication: TrainBatchPublication) -> TrainerAdmissionStatus:
        """Return the recorded definitive or fail-closed settlement state."""
        return self._get_pending_admission(publication).status

    def _ensure_admission_state(self) -> None:
        if not hasattr(self, "_manager_incarnation"):
            self._manager_incarnation = uuid.uuid4().hex
        if not hasattr(self, "_next_admission_id"):
            self._next_admission_id = 0
        if not hasattr(self, "_pending_admissions"):
            self._pending_admissions = {}

    def _get_pending_admission(self, publication: TrainBatchPublication) -> _PendingTrainerAdmission:
        self._ensure_admission_state()
        try:
            pending = self._pending_admissions[publication.admission_id]
        except KeyError:
            raise ValueError(f"Unknown trainer admission {publication.admission_id}.") from None
        if pending.publication != publication:
            raise ValueError(f"Trainer admission {publication.admission_id} does not match this manager publication.")
        return pending

    @staticmethod
    def _validate_receipts(
        publication: TrainBatchPublication,
        receipts: tuple[TrainerAdmissionReceipt, ...],
    ) -> None:
        if not receipts:
            raise ValueError(f"Trainer admission {publication.admission_id} requires exactly the expected roles.")
        roles: set[str] = set()
        for receipt in receipts:
            if not isinstance(receipt, TrainerAdmissionReceipt):
                raise ValueError(f"Trainer admission {publication.admission_id} received an invalid role receipt.")
            if receipt.publication != publication:
                raise ValueError(f"Trainer admission {publication.admission_id} has a foreign publication receipt.")
            if receipt.role not in publication.required_roles:
                raise ValueError(f"Trainer admission {publication.admission_id} has a foreign role {receipt.role!r}.")
            if receipt.role in roles:
                raise ValueError(f"Trainer admission {publication.admission_id} repeats role {receipt.role!r}.")
            if not isinstance(receipt.cohort, TrainerCohort):
                raise ValueError(f"Trainer admission {publication.admission_id} received an invalid cohort.")
            if not RolloutManager._is_canonical_cohort(receipt.cohort):
                raise ValueError(f"Trainer admission {publication.admission_id} received a non-canonical cohort.")
            roles.add(receipt.role)
        if roles != publication.required_roles:
            raise ValueError(f"Trainer admission {publication.admission_id} requires exactly the expected roles.")

    @staticmethod
    def _is_canonical_cohort(cohort: TrainerCohort) -> bool:
        if type(cohort.cells) is not tuple or not cohort.cells:
            return False
        if cohort.quorum_id is not None and (type(cohort.quorum_id) is not int or cohort.quorum_id < 0):
            return False
        if cohort.quorum_id is None and len(cohort.cells) != 1:
            return False

        cell_indices: list[int] = []
        for cell in cohort.cells:
            if not isinstance(cell, TrainerCellCohort) or type(cell.cell_index) is not int or cell.cell_index < 0:
                return False
            if type(cell.ranks) is not tuple or not cell.ranks:
                return False
            if any(type(rank) is not int or rank < 0 for rank in cell.ranks):
                return False
            if cell.ranks != tuple(sorted(set(cell.ranks))):
                return False
            cell_indices.append(cell.cell_index)

        if cell_indices != sorted(set(cell_indices)):
            return False
        return cohort.quorum_id is not None or cell_indices == [0]

    def _reject_unresolved_admissions(self, operation: str) -> None:
        self._ensure_admission_state()
        unresolved = [
            admission_id
            for admission_id, pending in self._pending_admissions.items()
            if pending.status
            in (
                TrainerAdmissionStatus.PENDING,
                TrainerAdmissionStatus.COMMIT_FAILED,
                TrainerAdmissionStatus.ROLLBACK_FAILED,
            )
        ]
        if unresolved:
            raise RuntimeError(f"Cannot {operation} with unresolved trainer admissions {unresolved}.")

    def _trim_terminal_admissions(self) -> None:
        terminal_ids = [
            admission_id
            for admission_id, pending in self._pending_admissions.items()
            if pending.status in (TrainerAdmissionStatus.COMMITTED, TrainerAdmissionStatus.ROLLED_BACK)
        ]
        for admission_id in terminal_ids[:-_MAX_RETAINED_TERMINAL_ADMISSIONS]:
            self._pending_admissions.pop(admission_id, None)

    async def eval(
        self,
        rollout_id,
        hf_dir: str | None = None,
        export_time_seconds: float | None = None,
        require_marker: bool = True,
    ):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self._health_monitoring_resume()

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

    async def _get_rollout_output(self, rollout_id: int) -> RolloutFnTrainOutput:
        if not self.use_legacy_rollout_v1:
            rollout_task = asyncio.create_task(
                asyncio.to_thread(
                    call_rollout_function,
                    self.generate_rollout,
                    RolloutFnTrainInput(rollout_id=rollout_id, weight_version=self.weight_version),
                )
            )
        else:
            rollout_task = asyncio.create_task(
                asyncio.to_thread(
                    call_rollout_fn,
                    self.generate_rollout,
                    self.args,
                    rollout_id,
                    self.data_source,
                    evaluation=False,
                )
            )

        try:
            return await asyncio.shield(cast(asyncio.Task[RolloutFnTrainOutput], rollout_task))
        except asyncio.CancelledError as cancellation_error:
            if rollout_task.cancelled():
                raise
            try:
                # Cancellation cannot stop the worker thread, so keep its Task shielded until settlement is possible.
                while not rollout_task.done():
                    try:
                        await asyncio.shield(rollout_task)
                    except asyncio.CancelledError:
                        if rollout_task.cancelled():
                            raise
                output = rollout_task.result()
                if isinstance(output, LeasedRolloutFnTrainOutput):
                    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
            except BaseException as cleanup_error:
                raise cancellation_error from cleanup_error
            raise

    # -------------------------- checkpointing -----------------------------

    def save(self, rollout_id):
        self._reject_unresolved_admissions("save")
        if self.args.rollout_global_dataset:
            self.data_source.save(rollout_id)
        event_logger_checkpoint.snapshot(self.args, rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    # -------------------------- offload/onload -----------------------------

    # TODO may parallelly execute offload/onload across services
    async def offload(self, tags: list[str] | None = None):
        self.health_monitoring_pause()
        for srv in self.servers.values():
            await srv.offload(tags=tags)

    async def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            await srv.onload(tags)

    async def onload_weights(self):
        if "weight" not in self.args.offload_rollout_level:
            return
        await self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    async def onload_kv(self):
        await self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    async def offload_kv(self):
        tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
        if "kv_cache" in self.args.offload_rollout_level:
            tags.append(GPU_MEMORY_TYPE_KV_CACHE)
        await self.offload(tags=tags)

    async def offload_weights(self):
        if "weight" not in self.args.offload_rollout_level:
            return
        await self.offload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    # -------------------------- engine management -----------------------------

    async def get_updatable_engines_and_lock(self):
        """Return engines eligible for weight updates."""
        srv = self._get_updatable_server()
        if not srv:
            return EnginesAndLock(
                rollout_engines=[],
                rollout_engine_lock=self.rollout_engine_lock,
                has_new_engines=False,
                engine_gpu_counts=[],
                engine_gpu_offsets=[],
            )

        await srv.wait_all_engines_alive()
        return EnginesAndLock(
            rollout_engines=[e.actor_handle for e in srv.engines],
            rollout_engine_lock=self.rollout_engine_lock,
            has_new_engines=srv.has_new_engines,
            engine_gpu_counts=srv.engine_gpu_counts,
            engine_gpu_offsets=srv.engine_gpu_offsets,
        )

    def clear_updatable_has_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear has_new_engines after update_weights
        srv = self._get_updatable_server()
        if srv:
            srv.clear_has_new_engines()

    async def recover_updatable_engines(self) -> None:
        """Restart any dead rollout engines and update has_new_engines for update_weights detection.

        Recovers the updatable model (the one that receives weight
        updates from training).
        """
        self.health_monitoring_pause()
        srv = self._get_updatable_server()
        if self.rollout_id == -1 or srv is None:
            return

        await srv.recover()

    def _get_updatable_server(self) -> RolloutServer | None:
        updatable = [srv for srv in self.servers.values() if srv.update_weights]
        match updatable:
            case []:
                return None
            case [srv]:
                return srv
            case _:
                raise ValueError(
                    f"Multiple servers have update_weights=True: {[srv.model_name for srv in updatable]}. "
                    f"Only one updatable server is supported."
                )

    # -------------------------- external start/stop -----------------------------

    async def start_cell(self, cell_id: int):
        port_cursors = PortCursors.empty()
        idx = get_cell_indexer_of_id_map(self.servers)[cell_id]
        group = self.servers[idx.srv_key].server_groups[idx.group_index]
        await group.recover(port_cursors=port_cursors, filter_indices=idx.engine_indices)

    async def stop_cell(self, cell_id: int):
        idx = get_cell_indexer_of_id_map(self.servers)[cell_id]
        group = self.servers[idx.srv_key].server_groups[idx.group_index]
        group.stop_engines(engine_indices=idx.engine_indices)

    # -------------------------- misc APIs -----------------------------

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    async def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ):
        # Only the updatable model is re-synced; a frozen model would always mismatch.
        srv = self._get_updatable_server()
        if srv is None:
            return []
        return await srv.check_weights(
            action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
        )

    def set_weight_version(self, weight_version: int):
        # warning instead of assert when use indep_dp ft
        if self.weight_version is not None and weight_version < self.weight_version:
            message = f"Engine weight version went backwards: {self.weight_version} -> {weight_version}"
            assert self.args.indep_dp, message
            logger.warning(message)
        self.weight_version = weight_version

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    # -------------------------- utils -----------------------------

    def health_monitoring_pause(self) -> None:
        for monitor in self._health_monitors:
            monitor.pause()

    def _health_monitoring_resume(self) -> None:
        for monitor in self._health_monitors:
            monitor.resume()

    @property
    def _server(self) -> RolloutServer | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    # TODO will be replaced by full ft, thus temporarily leave it without modifications
    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if (
            self._server
            and self._server.server_groups[0].all_engines
            and self._server.server_groups[0].all_engines[0].is_allocated
        ):
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self._server.server_groups[0].all_engines[0].actor_handle.simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")


@dataclass(frozen=True)
class EnginesAndLock:
    rollout_engines: list[ray.actor.ActorHandle]
    rollout_engine_lock: ray.actor.ActorHandle
    has_new_engines: bool
    engine_gpu_counts: list[int]
    engine_gpu_offsets: list[int]
