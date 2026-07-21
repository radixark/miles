import asyncio
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.rollout.addr_allocator import PortCursors
from miles.ray.rollout.debug_data import RolloutDataInjectionUtil, load_debug_rollout_data, save_debug_rollout_data
from miles.ray.rollout.metrics import log_eval_rollout_data, log_eval_skip, log_rollout_data
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.rollout_server import RolloutServer, start_rollout_servers
from miles.ray.rollout.router_manager import start_session_server
from miles.ray.rollout.server_cell import get_cell_indexer_of_id_map
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp
from miles.ray.utils import Lock
from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
    call_rollout_fn,
)
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.utils.audit_utils.event_analyzer import analyzer as event_analyzer
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.process_identity import RolloutManagerProcessIdentity
from miles.utils.environ import enable_experimental_rollout_refactor
from miles.utils.health_monitor import RolloutHealthMonitor
from miles.utils.hf_config import is_complete_hf_export
from miles.utils.http_utils import init_http_client
from miles.utils.logging_utils import configure_logger
from miles.utils.metric_checker import MetricChecker
from miles.utils.misc import load_function
from miles.utils.ray_utils import Box
from miles.utils.tracking_utils.tracking import init_tracking

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

# Per-attempt bound on eval-fleet weight loads; a zombie engine costs a skipped point.
EVAL_WEIGHT_LOAD_TIMEOUT_SECS = 600.0


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        event_logger_checkpoint.restore(args)
        configure_logger(args, source=RolloutManagerProcessIdentity())

        self.pg = pg
        self.args = args
        # TODO make args immutable
        init_tracking(args, primary=False, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}")

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.use_experimental_refactor = enable_experimental_rollout_refactor()
        if self.use_experimental_refactor:
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
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.servers: dict[str, RolloutServer] = {}
        else:
            init_http_client(args)
            self.servers = start_rollout_servers(args, pg)
            start_session_server(args)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1
        self._eval_lock = asyncio.Lock()
        self._eval_consumed_snapshots: list[str] = []

        self._metric_checker = MetricChecker.maybe_create(args)

        # TODO will be replaced by full ft, thus temporarily leave it without modifications
        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
            self._ci_fault_injection_pending = self.args.ci_test and "rollout" in self.args.ft_components

    # -------------------------- lifecycle -----------------------------
    # TODO: may have a `async def init` here later

    def dispose(self):
        event_analyzer.run_analysis_from_args(self.args)
        if self._metric_checker is not None:
            self._metric_checker.dispose()
        for monitor in self._health_monitors:
            monitor.stop()

    # -------------------------- data generation -----------------------------

    async def generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self._health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        data, metadata, metrics = await self._get_rollout_data(rollout_id=rollout_id)
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
            data_ref = Box(ray.put(data))
        else:
            data_ref = split_train_data_by_dp(self.args, data, self.train_parallel_config["dp_size"])
        return dict(sample_indices=sample_indices, data_ref=data_ref)

    async def eval(self, rollout_id, hf_dir: str | None = None, export_time_seconds: float | None = None):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self._health_monitoring_resume()

        if getattr(self.args, "eval_num_gpus", 0) > 0:
            assert hf_dir is not None, "eval with a dedicated fleet requires an HF snapshot dir"
            return await self._eval_on_dedicated_fleet(rollout_id, hf_dir, export_time_seconds)

        if self.use_experimental_refactor:
            result = await asyncio.to_thread(
                call_rollout_function, self.eval_generate_rollout, RolloutFnEvalInput(rollout_id=rollout_id)
            )
        else:
            result = await asyncio.to_thread(
                call_rollout_fn, self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True
            )
        data = result.data
        save_debug_rollout_data(self.args, data, rollout_id=rollout_id, evaluation=True)
        metrics = log_eval_rollout_data(rollout_id, self.args, data, result.metrics)
        if self._metric_checker is not None:
            self._metric_checker.on_eval(metrics)

    async def _eval_on_dedicated_fleet(self, rollout_id: int, hf_dir: str, export_time_seconds: float | None):
        """Pin the eval fleet to the snapshot for ``rollout_id``, then run eval.

        Every failure mode degrades to a skipped point logged at ``rollout_id``.
        """
        start_time = time.time()
        # The lock holds across load -> verify -> generate; this is the pinning enforcement.
        async with self._eval_lock:
            srv = self.servers["eval"]
            try:
                await self._mark_unreachable_eval_engines(srv)
                await srv.recover()
                await srv.wait_all_engines_alive()
            except Exception as e:
                logger.warning(f"Eval fleet unhealthy at rollout {rollout_id}, skipping eval: {e}")
                self._log_eval_skip(rollout_id, "unhealthy")
                return

            if hf_dir != self.args.hf_checkpoint and not is_complete_hf_export(hf_dir):
                logger.warning(f"Eval snapshot {hf_dir} missing or incomplete, skipping eval {rollout_id}")
                self._log_eval_skip(rollout_id, "ckpt_missing")
                return

            engines = [e.actor_handle for e in srv.engines]
            weight_version = str(rollout_id)
            for _attempt in range(2):
                try:
                    # A zombie engine (backend dead, actor alive) accepts the call and
                    # never answers; unbounded, it would hold the eval lock forever.
                    await asyncio.wait_for(
                        asyncio.gather(
                            *[
                                e.update_weights_from_disk.remote(hf_dir, weight_version=weight_version)
                                for e in engines
                            ]
                        ),
                        timeout=EVAL_WEIGHT_LOAD_TIMEOUT_SECS,
                    )
                    versions = await asyncio.wait_for(
                        asyncio.gather(*[e.get_weight_version.remote() for e in engines]),
                        timeout=EVAL_WEIGHT_LOAD_TIMEOUT_SECS,
                    )
                except Exception as e:
                    logger.warning(f"Eval fleet weight load from {hf_dir} failed: {e}")
                    versions = []
                if versions and all(str(v) == weight_version for v in versions):
                    break
            else:
                logger.warning(
                    f"Eval fleet failed to pin weight_version={weight_version} (got {versions}), skipping eval"
                )
                self._log_eval_skip(rollout_id, "pin_violation")
                return

            try:
                await self._wait_eval_router_ready(srv)
            except Exception as e:
                logger.warning(f"Eval router not ready at rollout {rollout_id}, skipping eval: {e}")
                self._log_eval_skip(rollout_id, "unhealthy")
                return

            result = await asyncio.to_thread(
                call_rollout_function, self.eval_generate_rollout, RolloutFnEvalInput(rollout_id=rollout_id)
            )
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

            self._gc_eval_snapshots(hf_dir)

    async def _wait_eval_router_ready(self, srv, timeout: float = 180.0) -> None:
        """After a revival the router 503s until its health cycle evicts the dead
        worker; a retried one-token probe proves the route is usable before dispatch."""
        import httpx

        url = f"http://{srv.router_ip}:{srv.router_port}/generate"
        payload = {"input_ids": [0], "sampling_params": {"max_new_tokens": 1, "temperature": 0}}
        deadline = time.time() + timeout
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    response = await client.post(url, json=payload, timeout=60)
                    if response.status_code == 200:
                        return
                    last_error = f"HTTP {response.status_code}"
                except httpx.HTTPError as e:
                    last_error = repr(e)
                if time.time() > deadline:
                    raise TimeoutError(f"eval router at {url} not ready after {timeout}s: {last_error}")
                await asyncio.sleep(5)

    async def _mark_unreachable_eval_engines(self, srv) -> None:
        """Without fault tolerance nothing records an engine death (recover() only
        restarts engines already marked stopped), so the controller probes itself."""
        for group in srv.server_groups:
            for engine in group.all_engines:
                if not engine.is_allocated:
                    continue
                try:
                    await asyncio.wait_for(engine.actor_handle.get_weight_version.remote(), timeout=60)
                except Exception as e:
                    logger.warning(f"Eval engine unreachable ({e!r}); marking stopped for recovery")
                    try:
                        ray.kill(engine.actor_handle)
                    except Exception:
                        pass
                    engine.mark_stopped()

    def report_eval_skip(self, rollout_id: int, reason: str) -> None:
        self._log_eval_skip(rollout_id, reason)

    def _log_eval_skip(self, rollout_id: int, reason: str) -> None:
        log_eval_skip(rollout_id, self.args, reason)

    def _gc_eval_snapshots(self, consumed_dir: str) -> None:
        """Delete consumed --eval-hf-dir snapshots beyond the keep ring; nothing else
        is ever deleted (pending evals reference unconsumed dirs)."""
        staging = getattr(self.args, "eval_hf_dir", None)
        if staging is None:
            return
        staging_root = Path(staging).resolve()
        consumed = Path(consumed_dir).resolve()
        if staging_root not in consumed.parents:
            return
        consumed = str(consumed)
        if consumed in self._eval_consumed_snapshots:
            self._eval_consumed_snapshots.remove(consumed)
        self._eval_consumed_snapshots.append(consumed)
        while len(self._eval_consumed_snapshots) > self.args.eval_keep_snapshots:
            victim = self._eval_consumed_snapshots.pop(0)
            shutil.rmtree(victim, ignore_errors=True)
            logger.info(f"GC'd consumed eval snapshot {victim}")

    async def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data, metadata = load_debug_rollout_data(self.args, rollout_id=rollout_id)
            metrics = None
        else:
            if self.use_experimental_refactor:
                data = await asyncio.to_thread(
                    call_rollout_function, self.generate_rollout, RolloutFnTrainInput(rollout_id=rollout_id)
                )
            else:
                data = await asyncio.to_thread(
                    call_rollout_fn, self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False
                )
            metrics = data.metrics
            data = data.samples
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

        return data, metadata, metrics

    # -------------------------- checkpointing -----------------------------

    def save(self, rollout_id):
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
        await self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    async def onload_kv(self):
        await self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

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
