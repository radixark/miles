import asyncio
import logging
import time

import ray

from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.debug_data import RolloutDataInjectionUtil, load_debug_rollout_data, save_debug_rollout_data
from miles.ray.rollout.metrics import log_eval_rollout_data, log_rollout_data
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
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
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.utils import object_store
from miles.utils.audit_utils.event_analyzer import analyzer as event_analyzer
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.process_identity import RolloutExecutorProcessIdentity
from miles.utils.environ import enable_experimental_rollout_refactor
from miles.utils.http_utils import init_http_client
from miles.utils.logging_utils import configure_logger
from miles.utils.metric_checker import MetricChecker
from miles.utils.misc import load_function
from miles.utils.timer import timer
from miles.utils.tracking_utils.tracking import init_tracking

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


@ray.remote
class RolloutExecutor:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args):
        event_logger_checkpoint.restore(args)
        configure_logger(args, source=RolloutExecutorProcessIdentity())

        self.args = args
        # TODO make args immutable
        init_tracking(args, primary=False, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
        object_store.init_instance(args, contribute_segment=False)

        if not self.args.debug_train_only:
            init_http_client(args)

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

        self._metric_checker = MetricChecker.maybe_create(args)

    # -------------------------- lifecycle -----------------------------
    # TODO: may have a `async def init` here later

    def dispose(self):
        if (close := getattr(self.data_source, "close", None)) is not None:
            close()
        event_analyzer.run_analysis_from_args(self.args)
        if self._metric_checker is not None:
            self._metric_checker.dispose()

    # -------------------------- data generation -----------------------------

    async def get(self, rollout_id):
        start_time = time.time()
        if (get_buffer_length := getattr(self.data_source, "get_buffer_length", None)) is not None:
            dashboard_hooks.report_data_buffer(get_buffer_length())
        with timer("rollout"):
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
            data_ref = object_store.get_instance().put(value=data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        else:
            data_ref = split_train_data_by_dp(self.args, data, self.train_parallel_config["dp_size"])
        return dict(sample_indices=sample_indices, data_ref=data_ref)

    async def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return

        with timer("eval_rollout"):
            if self.use_experimental_refactor:
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

    # TODO the train and eval rollout functions will become one object, so one save/load is enough here
    def save(self, rollout_id):
        if self.args.rollout_global_dataset:
            self.data_source.save(rollout_id)
        if self.use_experimental_refactor:
            self.generate_rollout.save(rollout_id)
        event_logger_checkpoint.snapshot(self.args, rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)
        if self.use_experimental_refactor:
            self.generate_rollout.load(rollout_id)

    # -------------------------- misc APIs -----------------------------

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config
