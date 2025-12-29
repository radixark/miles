import asyncio
import logging
import multiprocessing
import os
import random
import time
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.backends.sglang_utils.sglang_engine import SGLangEngine
from miles.rollout.base_types import call_rollout_fn
from miles.rollout.streaming_rollout_manager import StreamingRolloutManager, derive_streaming_start_params
from miles.utils import tracking_utils
from miles.utils.health_monitor import RolloutHealthMonitor
from miles.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, init_http_client
from miles.utils.iter_utils import group_by
from miles.utils.logging_utils import configure_logger
from miles.utils.metric_checker import MetricChecker
from miles.utils.metric_utils import compute_pass_rate, compute_rollout_step, compute_statistics, dict_add_prefix
from miles.utils.misc import load_function
from miles.utils.ray_utils import Box
from miles.utils.reward_normalization import group_normalize_rewards
from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions
from miles.utils.tracking_utils import init_tracking
from miles.utils.types import Sample

from ..utils.metric_utils import has_repetition
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        configure_logger()

        self.args = args
        self.pg = pg
        _start_router(args)
        # TODO make args immutable
        init_tracking(args, primary=False, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
        init_http_client(args)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.all_rollout_engines = []
        else:
            num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
            num_engines = args.rollout_num_gpus // num_gpu_per_engine
            self.all_rollout_engines = [None] * num_engines
        self.num_new_engines = init_rollout_engines(args, pg, self.all_rollout_engines)
        self.nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // args.num_gpus_per_node)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()

        self._metric_checker = MetricChecker.maybe_create(args)
        if self.args.use_fault_tolerance:
            self._health_monitor = RolloutHealthMonitor(self, args)

        self._streaming: StreamingRolloutManager | None = None
        self._trainer_version: int = 0
        self._streaming_rollout_id: int | None = None

    def dispose(self):
        if self._metric_checker is not None:
            self._metric_checker.dispose()

    # TODO maybe rename "rollout_engines" and "all_rollout_engines" later
    @property
    def rollout_engines(self):
        # when doing multi-node serving, we will only send request to node-0 for each engine.
        return self.all_rollout_engines[:: self.nodes_per_engine]

    def get_rollout_engines_and_lock(self):
        return self.rollout_engines, self.rollout_engine_lock, self.num_new_engines

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        monitor_started = self.args.use_fault_tolerance and self._health_monitor.start()
        start_time = time.time()
        try:
            data, metrics = self._get_rollout_data(rollout_id=rollout_id)
            self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
            _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
            data = self._convert_samples_to_train_data(data)
            return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])
        finally:
            if monitor_started:
                self._health_monitor.stop()
                self.num_new_engines = init_rollout_engines(self.args, self.pg, self.all_rollout_engines)
            else:
                self.num_new_engines = 0

    async def start_streaming(self, start_rollout_id: int):
        if self.args.debug_train_only or self.args.debug_rollout_only:
            raise RuntimeError("--streaming-async is not supported in debug-only modes")

        if self._streaming is not None:
            return self._streaming.supports_subset_engine_updates()

        num_engines = len(self.rollout_engines)
        if num_engines == 0:
            raise RuntimeError("No rollout engines available for --streaming-async")

        engine_urls = ray.get([engine.get_http_base_url.remote() for engine in self.rollout_engines])

        params = derive_streaming_start_params(self.args, num_engines=num_engines)

        # Capability hint to the trainer: if subset updates aren't supported, fall back to global update.
        if not params.supports_subset_engine_updates:
            logger.warning(
                "Only one rollout engine detected; subset engine updates are disabled and will fall back to global update."
            )

        self._trainer_version = 0
        self._streaming_rollout_id = start_rollout_id

        self._streaming = StreamingRolloutManager(
            self.args,
            self.data_source,
            engine_urls=engine_urls,
            groups_per_train_step=params.groups_per_train_step,
            queue_target=params.queue_target,
            queue_cap=params.queue_cap,
            inflight_target=params.inflight_target,
            min_active_engines=params.min_active_engines,
            weight_update_mode=self.args.streaming_async_weight_update_mode,
        )
        self._streaming.start()

        return self._streaming.supports_subset_engine_updates()

    async def stop_streaming(self):
        if self._streaming is None:
            return
        await self._streaming.stop()
        self._streaming = None

    async def get_next_train_batch(self) -> list[Box]:
        if self._streaming is None:
            raise RuntimeError("Streaming is not started")
        if self._streaming_rollout_id is None:
            raise RuntimeError("Streaming rollout id is not initialized")

        rollout_id = self._streaming_rollout_id

        groups, extra = await self._streaming.get_next_groups(
            num_groups=self.args.rollout_batch_size,
            target_version=self._trainer_version,
            max_staleness_versions=self.args.max_staleness_versions,
        )

        samples: list[Sample] = []
        for g in groups:
            samples.extend(g.group)

        data = self._convert_samples_to_train_data(samples)
        refs = self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])

        staleness_values = extra.get("staleness_values") or []
        ages_s = extra.get("queue_ages_s") or []
        stats = self._streaming.stats()

        log_dict: dict[str, Any] = {
            "rollout/stream/queue_size_groups": stats["queue_size_groups"],
            "rollout/stream/inflight_groups": stats["inflight_groups"],
            "rollout/stream/groups_produced_per_s": stats["groups_produced_per_s"],
            "rollout/stream/groups_consumed_per_s": stats["groups_consumed_per_s"],
            "rollout/stream/empty_wait_s": extra.get("empty_wait_s", 0.0),
            "rollout/stream/stale_groups_dropped": stats["stale_groups_dropped"],
        }

        engine_pool = stats.get("engine_pool") or {}
        log_dict["rollout/stream/drain_only_engines"] = engine_pool.get("num_drain_only", 0)
        log_dict["rollout/stream/active_engines"] = engine_pool.get("num_active", 0)

        for v, count in (engine_pool.get("versions") or {}).items():
            log_dict[f"rollout/stream/engine_versions/v{v}"] = count

        if staleness_values:
            for s, items in group_by(staleness_values).items():
                log_dict[f"rollout/stream/staleness_versions_hist/v{s}"] = len(items)

        if ages_s:
            log_dict["rollout/stream/queue_age_s_p50"] = float(np.percentile(ages_s, 50))
            log_dict["rollout/stream/queue_age_s_p95"] = float(np.percentile(ages_s, 95))

        step = compute_rollout_step(self.args, rollout_id)
        log_dict["rollout/step"] = step
        tracking_utils.log(self.args, log_dict, step_key="rollout/step")

        self._streaming_rollout_id += 1

        return refs

    async def notify_new_version(self, version: int):
        self._trainer_version = version
        if self._streaming is not None:
            self._streaming.notify_new_version(version)

    async def get_update_candidates(self) -> list[int]:
        if self._streaming is None:
            return []
        return self._streaming.get_update_candidates()

    async def mark_engines_updated(self, engine_indices: list[int], version: int):
        if self._streaming is None:
            return
        self._streaming.mark_engines_updated(engine_indices, version)

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return

        # TODO: add fault tolerance to eval
        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        metrics = _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)
        if self._metric_checker is not None:
            self._metric_checker.on_eval(metrics)

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def offload(self):
        return ray.get([engine.release_memory_occupation.remote() for engine in self.rollout_engines])

    def onload(self, tags: list[str] = None):
        return ray.get([engine.resume_memory_occupation.remote(tags=tags) for engine in self.rollout_engines])

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                open(self.args.load_debug_rollout_data.format(rollout_id=rollout_id), "rb"),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                logger.info(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            metrics = None
        else:
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = sum(data, [])

            if self.args.disable_rollout_trim_samples:
                logger.info(f"Collectd {len(data)} samples from rollout to train")
            elif len(data) % self.args.global_batch_size != 0:
                trim_len = (len(data) // self.args.global_batch_size) * self.args.global_batch_size
                origin_data_length = len(data)
                data = data[:trim_len]
                logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
        return data, metrics

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            if evaluation:
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        if samples and isinstance(samples[0], list):
            samples = sum(samples, [])

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            # group norm
            normalized_rewards = group_normalize_rewards(
                samples=samples,
                raw_rewards=raw_rewards,
                advantage_estimator=self.args.advantage_estimator,
                grpo_std_normalization=self.args.grpo_std_normalization,
            )
            return raw_rewards, normalized_rewards

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # overwriting the raw reward
        if samples[0].metadata and "raw_reward" in samples[0].metadata:
            train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if samples[0].multimodal_inputs is not None:
            train_data["multimodal_inputs"] = [sample.multimodal_inputs for sample in samples]

        if "teacher_log_probs" in samples[0].__dict__:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        return train_data

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def _split_train_data_by_dp(self, data, dp_size):
        """Split the train data by data parallel size."""
        rollout_data = {}

        if "prompt" in data:
            rollout_data["prompt"] = data["prompt"]

        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        if self.args.balance_data:
            partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
        else:
            partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

        rollout_data_refs = []

        for i in range(dp_size):
            rollout_data = {}
            partition = partitions[i]
            rollout_data["partition"] = partition
            for key in [
                "tokens",
                "multimodal_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "sample_indices",
                "rollout_log_probs",
                "rollout_routed_experts",
                "prompt",
                "teacher_log_probs",
            ]:
                if key not in data:
                    continue
                val = [data[key][j] for j in partition]
                rollout_data[key] = val
            # keys that need to be splited at train side
            for key in [
                "raw_reward",
                "total_lengths",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        return rollout_data_refs


def init_rollout_engines(args, pg, all_rollout_engines):
    if args.debug_train_only:
        return 0, None

    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
    num_engines = args.rollout_num_gpus // num_gpu_per_engine
    assert len(all_rollout_engines) == num_engines
    if args.prefill_num_servers is not None:
        prefill_num_servers = args.prefill_num_servers * args.rollout_num_gpus_per_engine // num_gpu_per_engine
        assert (
            num_engines > prefill_num_servers
        ), f"num_engines {num_engines} should be larger than prefill_num_servers {prefill_num_servers}"

    pg, reordered_bundle_indices = pg

    RolloutRayActor = ray.remote(SGLangEngine)

    rollout_engines = []
    for i in range(num_engines):
        if all_rollout_engines[i] is not None:
            continue

        num_gpus = 0.2
        num_cpus = num_gpus

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )

        env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
            "SGL_JIT_DEEPGEMM_PRECOMPILE": "false",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
            "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
            "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
        }

        # TODO: currently the amem position is hardcoded, change to a better way later.
        # note that amem does not work with update weights from distributed.
        if (
            args.offload_rollout
            and args.actor_num_nodes * args.actor_num_gpus_per_node >= args.rollout_num_gpus
            and len(glob("/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libamem_nccl.so*")) > 0
        ):
            logger.info("Enable AMEM for rollout engine.")
            ld_library_path = (
                os.environ.get("LD_LIBRARY_PATH", "") + ":/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib"
            )
            env_vars |= {
                "LD_LIBRARY_PATH": ld_library_path,
                "NCCL_CUMEM_ENABLE": "1",
                "AMEM_ENABLE": "1",
                "AMEM_GROUPID": "0",
                "GMM_LOG": "2",
            }

        worker_type = "regular"
        if args.prefill_num_servers is not None:
            if i < prefill_num_servers:
                worker_type = "prefill"
            else:
                worker_type = "decode"

        rollout_engine = RolloutRayActor.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={
                "env_vars": env_vars,
            },
        ).remote(args, rank=i, worker_type=worker_type)

        rollout_engines.append((i, rollout_engine))
        all_rollout_engines[i] = rollout_engine

    num_new_engines = len(rollout_engines)

    if num_new_engines == 0:
        return num_new_engines, None

    if args.rollout_external:
        addr_and_ports = _allocate_rollout_engine_addr_and_ports_external(args=args, rollout_engines=rollout_engines)
    else:
        addr_and_ports = _allocate_rollout_engine_addr_and_ports_normal(
            args=args, num_engines=num_engines, rollout_engines=rollout_engines
        )

    # TODO: don't ray.get here to overlap train actor init with rollout engine init.
    # somehow if we don't sync here, the --debug-rollout-only mode will crash.
    init_handles = [engine.init.remote(**(addr_and_ports[rank])) for rank, engine in rollout_engines]
    ray.get(init_handles)

    return num_new_engines


def _allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    addr_and_ports = []
    for rank, _ in rollout_engines:
        [host, port] = args.rollout_external_engine_addrs[rank].split(":")
        addr_and_ports.append(
            dict(
                dist_init_addr=None,
                nccl_port=None,
                host=host,
                port=int(port),
            )
        )
    return addr_and_ports


def _allocate_rollout_engine_addr_and_ports_normal(*, args, num_engines, rollout_engines):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    num_engines_per_node = max(
        1, min(args.num_gpus_per_node, args.rollout_num_gpus) // args.rollout_num_gpus_per_engine
    )
    addr_and_ports = [{} for _ in range(num_engines)]

    visited_nodes = set()
    for rank, engine in rollout_engines:
        if rank // num_engines_per_node in visited_nodes:
            continue
        visited_nodes.add(rank // num_engines_per_node)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (rank % num_engines_per_node)

        def get_addr_and_ports(engine):
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            start_port = 15000

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine)

        for i in range(num_engines_on_this_node):
            addr_and_ports[rank + i]["host"] = get_addr()
            addr_and_ports[rank + i]["port"] = get_port()
            addr_and_ports[rank + i]["nccl_port"] = get_port()

        if args.rollout_num_gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = args.rollout_num_gpus_per_engine // args.num_gpus_per_node
            if rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports


def _start_router(args, prefill_and_decode_urls=None):
    """start sgl router and miles router"""
    if args.sglang_router_ip is not None:
        return

    args.sglang_router_ip = _wrap_ipv6(get_host_info()[1])
    if args.sglang_router_port is None:
        args.sglang_router_port = find_available_port(random.randint(3000, 4000))

    if args.use_miles_router:
        assert args.prefill_num_servers is None, "miles router does not support prefill_num_servers."
        from miles.router.router import run_router

        router_args = args

    else:
        from sglang_router.launch_router import RouterArgs

        from miles.utils.http_utils import run_router

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
        router_args.host = args.sglang_router_ip
        router_args.port = args.sglang_router_port
        router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
        router_args.log_level = "warn"

        if args.prefill_num_servers is not None:
            router_args.pd_disaggregation = True

        if hasattr(router_args, "request_timeout_secs"):
            router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

        logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {args.sglang_router_ip}:{args.sglang_router_port}")


def _log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    log_dict = extra_metrics or {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    tracking_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    response_lengths = [sample.effective_response_length for sample in samples]
    log_dict["perf/rollout_time"] = rollout_time
    if args.rollout_num_gpus:
        log_dict["perf/tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
    log_dict["perf/longest_sample_tokens_per_sec"] = max(response_lengths) / rollout_time
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")


def compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if args.sglang_speculative_algorithm is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["rollout/spec_accept_rate"] = (
        sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    )
    metrics["rollout/spec_accept_length"] = (
        sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    )
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
