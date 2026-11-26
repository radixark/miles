import argparse
import functools
import ipaddress
import json
import logging
import os
import shlex
import sys

from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.utils.lora.utils import (
    LORA_ADAPTER_NAME,
    engine_loads_adapter_from_disk,
    is_multi_lora_enabled,
    lora_base_cpu_backup_enabled,
    lora_rollout_enabled,
)
from miles.utils.workers.argv_utils import _record_field_names

# This module also runs as a diffusion engine's process, which pins its GPU first, so the text-engine functions import
# what initializes CUDA (sglang.srt) when called.

logger = logging.getLogger(__name__)

# SGLang's default SGLANG_HEALTH_CHECK_TIMEOUT.
SGLANG_DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS = 20


def compute_health_check_timeout_seconds(*, pp_size: int) -> int:
    # A cold health request crosses every pipeline stage, so the launcher's budget is taken per stage
    # and the engine gets the product instead of the launcher's value.
    per_stage_seconds = int(os.environ.get("SGLANG_HEALTH_CHECK_TIMEOUT", SGLANG_DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS))
    return per_stage_seconds * pp_size


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


def compute_engine_launch_cmd(
    args,
    *,
    interpreter_prefix: list[str],
    node_rank: int,
    worker_type: WorkerType,
    base_gpu_id: int,
    sglang_overrides: dict,
    num_gpus_per_engine: int,
    tp_size: int,
    pp_size: int,
    dist_init_addr: str,
    nccl_port: int,
    host: str,
    port: int,
    disaggregation_bootstrap_port: int | None,
    engine_info_bootstrap_port: int,
    gated_launch_port: int,
    random_seed: int,
) -> str:
    _assert_launch_gate_served()

    server_args_dict = _compute_server_args(
        args,
        node_rank=node_rank,
        dist_init_addr=dist_init_addr,
        nccl_port=nccl_port,
        host=host,
        port=port,
        worker_type=worker_type,
        disaggregation_bootstrap_port=disaggregation_bootstrap_port,
        base_gpu_id=base_gpu_id,
        engine_info_bootstrap_port=engine_info_bootstrap_port,
        sglang_overrides=sglang_overrides,
        num_gpus_per_engine=num_gpus_per_engine,
        tp_size=tp_size,
        pp_size=pp_size,
        gated_launch_port=gated_launch_port,
        random_seed=random_seed,
    )

    from miles.backends.sglang_utils.server_args_utils import server_args_to_argv

    launch_args = {**server_args_dict, "host": server_args_dict["host"].strip("[]")}
    return shlex.join([*interpreter_prefix, "-m", "sglang.launch_server", *server_args_to_argv(launch_args)])


def _compute_server_args(
    args,
    *,
    node_rank: int,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: WorkerType = WorkerType.REGULAR,
    disaggregation_bootstrap_port: int | None,
    base_gpu_id: int,
    engine_info_bootstrap_port: int | None,
    sglang_overrides: dict | None,
    num_gpus_per_engine: int | None,
    tp_size: int,
    pp_size: int,
    gated_launch_port: int,
    random_seed: int,
):
    from sglang.srt.server_args import ServerArgs

    from miles.backends.sglang_utils.arguments import sglang_attention_dp_enabled
    from miles.utils.output_store.config import sglang_output_store_server_args

    base_args = args.sglang.base_args
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    nnodes = max(1, _gpus_per_engine // args.num_gpus_per_node)
    kwargs = {
        "model_path": args.hf_checkpoint,
        "trust_remote_code": True,
        "random_seed": random_seed,
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
        "base_gpu_id": base_gpu_id,
        "gated_launch_port": gated_launch_port,
        # parallel
        "tp_size": tp_size,
        "dp_size": base_args["dp_size"],
        "pp_size": pp_size,
        "ep_size": base_args["ep_size"],
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": True,
        # always enable draft weights cpu backup so that we run training without mtp weights.
        "enable_draft_weights_cpu_backup": True,
        # always serve /metrics so Prometheus scrapers can read engine stats.
        "enable_metrics": True,
    }

    if os.environ.get("MILES_SGLANG_DUMMY_LOAD") == "1":
        kwargs["load_format"] = "dummy"

    if worker_type == WorkerType.PREFILL:
        kwargs["disaggregation_mode"] = WorkerType.PREFILL.value
        kwargs.setdefault("load_balance_method", "round_robin")
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == WorkerType.DECODE:
        kwargs["disaggregation_mode"] = WorkerType.DECODE.value
        kwargs["prefill_round_robin_balance"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.use_rollout_indexer_replay:
        kwargs["enable_return_indexer_topk"] = True
    # SGLang v0.5.23 turns DeepSeek-V4.1 decoder SWA bounded replay on when this is unset; it is not
    # numerically equivalent to full prefill, so rollout logprobs and replayed routing would drift
    # from the trainer. Keep full prefill unless --sglang-enable-decoder-swa-bounded-replay is set.
    if base_args.get("enable_decoder_swa_bounded_replay") is None:
        kwargs["enable_decoder_swa_bounded_replay"] = False
    if args.use_output_store:
        # Every engine, eval included: an eval request never asks for the store.
        kwargs.update(sglang_output_store_server_args(args, local_hostname=host.strip("[]")))
    if engine_info_bootstrap_port is not None:
        kwargs["engine_info_bootstrap_port"] = engine_info_bootstrap_port

    if is_multi_lora_enabled(args):
        kwargs["enable_lora"] = True
        kwargs["max_loras_per_batch"] = args.multi_lora_n_adapters
        kwargs["max_lora_rank"] = max(args.lora_rank, 1)
        kwargs["lora_target_modules"] = (
            ["all"] if args.lora_adapter_targets == "all-linear" else args.lora_adapter_targets
        )
    elif lora_rollout_enabled(args):
        attention_dp = sglang_attention_dp_enabled(
            attn_dp_size=base_args["attn_dp_size"],
            # absent on SGLang builds that dropped --enable-dp-attention
            enable_dp_attention=base_args.get("enable_dp_attention", False),
            dp_size=base_args["dp_size"],
        )
        kwargs["enable_lora"] = True
        # attention DP serves only pinned adapters (sgl-project/sglang#36389), and SGLang never pins the last slot
        kwargs["max_loras_per_batch"] = 2 if attention_dp else 1
        kwargs["max_lora_rank"] = max(args.lora_rank, 1)
        kwargs["lora_target_modules"] = (
            ["all"] if args.lora_adapter_targets == "all-linear" else args.lora_adapter_targets
        )

        if engine_loads_adapter_from_disk(args):
            if attention_dp:
                kwargs["lora_paths"] = [
                    {"lora_name": LORA_ADAPTER_NAME, "lora_path": args.lora_adapter_path, "pinned": True}
                ]
            else:
                kwargs["lora_paths"] = [f"{LORA_ADAPTER_NAME}={args.lora_adapter_path}"]
        elif args.lora_adapter_path is not None:
            logger.info("Skipping startup lora_paths: the trainer pushes the adapter in the first weight sync")
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

    unused_keys = set(kwargs.keys())
    for name in _record_field_names(ServerArgs):
        if worker_type == WorkerType.DECODE and name == "enable_hierarchical_cache":
            continue
        if name in base_args and name not in kwargs:
            kwargs[name] = base_args[name]
        unused_keys.discard(name)

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    if is_multi_lora_enabled(args):
        assert kwargs.get("load_format") != "dummy", "Tinker engines must load the frozen base from disk"
        if kwargs.get("max_loaded_loras") is None:
            # use --sglang-max-loaded-loras to override
            # TODO: dynamic allocation
            kwargs["max_loaded_loras"] = 2 * kwargs["max_loras_per_batch"]

    if kwargs.get("device") is None:
        kwargs["device"] = "cuda"

    return kwargs


@functools.cache
def _assert_launch_gate_served() -> None:
    from sglang.srt.server_args import ServerArgs

    assert "gated_launch_port" in _record_field_names(ServerArgs), (
        "this sglang has no --gated-launch-port, and miles launches every inference engine through "
        "that gate; upgrade sglang to one that serves it"
    )


# Miles sets these for each diffusion engine, or leaves them to SGLang (the scheduler binds a free port when none is
# set); sglang.multimodal_gen itself rejects any other topology that does not fit on the one GPU they give it.
MILES_OWNED_DIFFUSION_ARGS = ("num_gpus", "master_port", "scheduler_port")
# Ports and GPUs Miles allocated for a diffusion engine; SGLang may move a busy port when it builds its args.
_ALLOCATED_DIFFUSION_ARGS = ("num_gpus", "port", "master_port")
_DIFFUSION_ENTRY_MODULE = "miles.backends.sglang_utils.sglang_engine"


def compute_diffusion_engine_launch_cmd(
    *,
    interpreter_prefix: list[str],
    base_gpu_id: int,
    sglang_overrides: dict,
    host: str,
    port: int,
    master_port: int,
) -> str:
    server_args = compute_diffusion_server_args(
        sglang_overrides=sglang_overrides, host=host, port=port, master_port=master_port
    )
    return shlex.join(
        [
            *interpreter_prefix,
            "-m",
            _DIFFUSION_ENTRY_MODULE,
            "--base-gpu-id",
            str(base_gpu_id),
            "--server-args",
            json.dumps(server_args),
        ]
    )


def compute_diffusion_server_args(*, sglang_overrides: dict, host: str, port: int, master_port: int) -> dict:
    """``sglang.multimodal_gen`` server args, keyed by its CLI option destinations.

    The group's overrides, which carry the model's ``model_path``, win over these defaults, as they do for text
    engines; none of the run-wide ``--sglang-*`` text-server args apply.
    """
    server_args = {
        "trust_remote_code": True,
        "host": host.strip("[]"),
        "port": port,
        "master_port": master_port,
        "num_gpus": 1,
        "warmup_mode": "off",
        # Miles places engines on GPUs of their own; the automatic policy would offload by free memory at startup.
        "performance_mode": "manual",
    }
    return {**server_args, **sglang_overrides}


def pin_gpu(base_gpu_id: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[base_gpu_id] if visible else str(base_gpu_id)


def reject_unknown_server_args(server_args: dict, *, option_names: set[str]) -> None:
    """Accept only the names SGLang's own CLI registers, spelled as its option destinations (no aliases)."""
    if unknown := sorted(server_args.keys() - option_names):
        raise ValueError(f"sglang.multimodal_gen has no server option named: {unknown}")


def reject_moved_allocations(server_args: dict, *, resolved) -> None:
    resolved_values = {
        key: getattr(resolved, key) for key in _ALLOCATED_DIFFUSION_ARGS  # config-access-exempt: SGLang's args
    }
    moved = {key: value for key, value in resolved_values.items() if value != server_args[key]}
    if moved:
        raise ValueError(f"sglang.multimodal_gen resolved {moved}, not what Miles allocated for this engine")


def sglang_option_names(server_args: dict) -> set[str]:
    """The options SGLang's `serve` command takes for this model, its pipeline's own ones included."""
    from sglang.multimodal_gen.runtime.entrypoints.cli.serve import add_multimodal_gen_serve_args
    from sglang.multimodal_gen.runtime.utils.argparse import FlexibleArgumentParser

    # SGLang picks the pipeline's extra options from the command line it is parsing.
    selector = [
        f"--{key.replace('_', '-')}={server_args[key]}"
        for key in ("model_path", "pipeline_class_name")
        if key in server_args
    ]
    saved_argv, sys.argv = sys.argv, [sys.argv[0], *selector]
    try:
        parser = add_multimodal_gen_serve_args(FlexibleArgumentParser())
    finally:
        sys.argv = saved_argv
    return {action.dest for action in parser._actions} - {"help", "config"}


def serve_diffusion_engine(server_args: dict) -> None:
    from sglang.multimodal_gen.runtime.launch_server import dispatch_launch
    from sglang.multimodal_gen.runtime.server_args import ServerArgs
    from sglang.multimodal_gen.runtime.utils.process import kill_process_tree

    reject_unknown_server_args(server_args, option_names=sglang_option_names(server_args))
    # The dict constructor SGLang's own CLI ends in; it keeps structured values such as warmup_sampling_params.
    resolved = ServerArgs.from_dict(dict(server_args))
    reject_moved_allocations(server_args, resolved=resolved)
    try:
        dispatch_launch(resolved)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def main(argv: list[str] | None = None) -> None:
    """Entry point of one ``sglang.multimodal_gen`` server process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-gpu-id", type=int, required=True)
    parser.add_argument("--server-args", type=json.loads, required=True)
    cli = parser.parse_args(argv)
    pin_gpu(cli.base_gpu_id)
    serve_diffusion_engine(cli.server_args)


if __name__ == "__main__":
    main()
