import dataclasses
import ipaddress
import logging
import os
import shlex
import sys

from sglang.srt.server_args import ServerArgs

from miles.backends.megatron_utils.lora_utils import (
    convert_target_modules_to_hf,
    lora_base_cpu_backup_enabled,
    sglang_lora_target_all_sentinel,
)
from miles.backends.sglang_utils.server_args_utils import server_args_to_argv
from miles.utils.lora import LORA_ADAPTER_NAME, lora_rollout_enabled
from miles.utils.multi_lora import is_multi_lora_enabled

logger = logging.getLogger(__name__)


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")
    if not cvd:
        return physical_gpu_id  # no remapping
    # CUDA_VISIBLE_DEVICES can be like "4,5,6,7"
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    # In a remapped process, valid torch device indices are 0..len(visible)-1
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    # If we're already getting local IDs, allow them
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible)-1} (local)."
    )


def format_v6_uri(addr: str | None) -> str | None:
    if not addr or addr.startswith("["):
        return addr
    try:
        if ipaddress.ip_address(addr).version == 6:
            return f"[{addr}]"
    except ValueError:
        pass
    return addr


def build_server_url(host: str, port: int) -> str:
    return f"http://{format_v6_uri(host)}:{port}"


@dataclasses.dataclass(frozen=True)
class EngineLaunchPlan:
    cmd: str
    api_key: str | None


def compute_engine_launch_plan(
    args,
    *,
    node_rank: int,
    worker_type: str,
    base_gpu_id: int,
    sglang_overrides: dict,
    num_gpus_per_engine: int,
    addr_and_ports: dict,
) -> EngineLaunchPlan:
    server_args_dict = _compute_server_args(
        args,
        node_rank=node_rank,
        dist_init_addr=addr_and_ports["dist_init_addr"],
        nccl_port=addr_and_ports["nccl_port"],
        host=addr_and_ports["host"],
        port=addr_and_ports["port"],
        worker_type=worker_type,
        disaggregation_bootstrap_port=addr_and_ports.get("disaggregation_bootstrap_port"),
        base_gpu_id=base_gpu_id,
        engine_info_bootstrap_port=addr_and_ports.get("engine_info_bootstrap_port"),
        sglang_overrides=sglang_overrides,
        num_gpus_per_engine=num_gpus_per_engine,
    )

    launch_args = {**server_args_dict, "host": server_args_dict["host"].strip("[]")}
    cmd = shlex.join([sys.executable, "-m", "sglang.launch_server", *server_args_to_argv(launch_args)])
    return EngineLaunchPlan(cmd=cmd, api_key=server_args_dict.get("api_key"))


def _compute_server_args(
    args,
    *,
    node_rank: int,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int,
    engine_info_bootstrap_port: int | None = None,
    sglang_overrides: dict | None = None,
    num_gpus_per_engine: int | None = None,
):
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    nnodes = max(1, _gpus_per_engine // args.num_gpus_per_node)
    base = _to_local_gpu_id(base_gpu_id)
    kwargs = {
        "model_path": args.hf_checkpoint,
        "trust_remote_code": True,
        # NOTE: do not pass random seed and let SGLang pick random ones
        # "random_seed": args.seed + rank,
        # memory
        "enable_memory_saver": args.offload_rollout,
        # distributed
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base,
        # parallel
        "tp_size": _gpus_per_engine,
        "dp_size": args.sglang_dp_size,
        "pp_size": args.sglang_pp_size,
        "ep_size": args.sglang_ep_size,
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": True,
        # always enable draft weights cpu backup so that we run training without mtp weights.
        "enable_draft_weights_cpu_backup": True,
        # always serve /metrics so Prometheus scrapers can read engine stats.
        "enable_metrics": True,
    }

    if os.environ.get("MILES_SGLANG_DUMMY_LOAD") == "1":
        kwargs["load_format"] = "dummy"

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        kwargs.setdefault("load_balance_method", "round_robin")
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.use_rollout_indexer_replay:
        kwargs["enable_return_indexer_topk"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    if engine_info_bootstrap_port is not None:
        kwargs["engine_info_bootstrap_port"] = engine_info_bootstrap_port

    if is_multi_lora_enabled(args):
        kwargs["enable_lora"] = True
        kwargs["max_loras_per_batch"] = args.multi_lora_n_adapters
        kwargs["max_lora_rank"] = max(getattr(args, "lora_rank", 0), 1)
        kwargs["lora_target_modules"] = convert_target_modules_to_hf(args.target_modules)
    elif lora_rollout_enabled(args):
        kwargs["enable_lora"] = True
        kwargs["max_loras_per_batch"] = 1
        kwargs["max_lora_rank"] = max(getattr(args, "lora_rank", 0), 1)
        if sglang_lora_target_all_sentinel(args):
            kwargs["lora_target_modules"] = ["all"]
        else:
            kwargs["lora_target_modules"] = convert_target_modules_to_hf(args.target_modules)

        if args.lora_adapter_path is not None and kwargs.get("load_format") != "dummy":
            kwargs["lora_paths"] = [f"{LORA_ADAPTER_NAME}={args.lora_adapter_path}"]
        elif args.lora_adapter_path is not None:
            logger.info("dummy base load: skipping startup lora_paths; adapter comes via weight-sync")
        else:
            logger.info("No pre-trained LoRA adapter_path provided, will use random initial weights")

        if lora_base_cpu_backup_enabled(args):
            # Host-RAM mirror of the base weights so they survive
            # torch_memory_saver.pause() across rollout/training swaps without
            # needing to be re-shipped from the trainer. The trainer mirrors
            # this by skipping the base weight sync entirely (see
            # UpdateWeightFromTensor.update_weights).
            kwargs["enable_weights_cpu_backup"] = True
            logger.info(
                "LoRA + colocate: enabling SGLang enable_weights_cpu_backup=True; "
                "the trainer will skip per-step base weight sync."
            )

    # Last, so a per-group override wins over every args-derived default above.
    if sglang_overrides:
        kwargs.update(sglang_overrides)

    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    unused_keys = set(kwargs.keys())
    for attr in dataclasses.fields(ServerArgs):
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
        unused_keys.discard(attr.name)

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    return kwargs
