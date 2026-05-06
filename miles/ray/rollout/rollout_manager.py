import asyncio
import logging
import time

import ray

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
from miles.ray.rollout.debug_data import load_debug_rollout_data, save_debug_rollout_data
from miles.ray.rollout.metrics import log_eval_rollout_data, log_rollout_data
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.rollout_server import RolloutServer, start_rollout_servers
from miles.ray.rollout.router_manager import start_session_server
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp
from miles.ray.utils import Lock
from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
    call_rollout_fn,
)
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.utils.environ import enable_experimental_rollout_refactor
from miles.utils.health_monitor import RolloutHealthMonitor
from miles.utils.http_utils import init_http_client
from miles.utils.logging_utils import configure_logger
from miles.utils.metric_checker import MetricChecker
from miles.utils.misc import load_function
from miles.utils.tracking_utils import init_tracking

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        configure_logger()

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

        self._metric_checker = MetricChecker.maybe_create(args)

        # TODO will be replaced by full ft, thus temporarily leave it without modifications
        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
            self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection

    # -------------------------- lifecycle -----------------------------
    # TODO: may have a `async def init` here later

    def dispose(self):
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
        save_debug_rollout_data(self.args, data, rollout_id=rollout_id, evaluation=False)
        log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        data = convert_samples_to_train_data(
            self.args,
            data,
            metadata=metadata,
            custom_convert_samples_to_train_data_func=self.custom_convert_samples_to_train_data_func,
            custom_reward_post_process_func=self.custom_reward_post_process_func,
        )
        return split_train_data_by_dp(self.args, data, self.train_parallel_config["dp_size"])

    async def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self._health_monitoring_resume()

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

    async def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = load_debug_rollout_data(self.args, rollout_id=rollout_id)
            metadata = {}  # save/load metadata into debug rollout data as well
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

        return data, metadata, metrics

    # -------------------------- checkpointing -----------------------------

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    # -------------------------- offload/onload -----------------------------

    # TODO may parallelly execute offload/onload across services
    async def offload(self, tags: list[str] | None = None):
        self._health_monitoring_pause()
        for srv in self.servers.values():
            await srv.offload(tags)

    async def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            await srv.onload(tags)

    async def onload_weights(self):
        await self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    async def onload_kv(self):
        await self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    # -------------------------- engine management -----------------------------

    def get_updatable_engines_and_lock(self):
        """Return engines eligible for weight updates."""
        srv = self._get_updatable_server()
        engines = srv.engines if srv else []
        gpu_counts = srv.engine_gpu_counts if srv else []
        gpu_offsets = srv.engine_gpu_offsets if srv else []
        has_new = srv.has_new_engines if srv else False
        return engines, self.rollout_engine_lock, has_new, gpu_counts, gpu_offsets

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
        self._health_monitoring_pause()
        srv = self._get_updatable_server()
        if self.rollout_id == -1 or srv is None:
            return

        await srv.recover()

    def _get_updatable_server(self) -> RolloutServer | None:
        updatable = [srv for srv in self.servers.values() if srv.update_weights]
        assert len(updatable) <= 1, (
            f"Multiple servers have update_weights=True: {[srv.model_name for srv in updatable]}. "
            f"Only one updatable server is supported."
        )
        return updatable[0] if updatable else None

    # -------------------------- misc APIs -----------------------------

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    async def check_weights(self, action: str):
        return await asyncio.gather(*[srv.check_weights(action=action) for srv in self.servers.values()])

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    # -------------------------- utils -----------------------------

    def _health_monitoring_pause(self) -> None:
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

        if self._server and self._server.server_groups[0].all_engines and self._server.server_groups[0].all_engines[0]:
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self._server.server_groups[0].all_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")
