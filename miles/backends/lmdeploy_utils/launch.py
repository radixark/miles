"""Launch the LMDeploy fork's Miles protocol adapter on one GPU per engine."""

import os
import shlex
from pathlib import Path


def validate_lmdeploy_args(args) -> None:
    if args.rollout_engine_backend != "lmdeploy":
        return
    if not args.lmdeploy_python or not Path(args.lmdeploy_python).is_file():
        raise ValueError(
            "--rollout-engine-backend lmdeploy requires --lmdeploy-python pointing to its Python executable"
        )
    if args.router != "miles":
        raise ValueError("LMDeploy requires --router miles")
    if not args.colocate or args.rollout_num_gpus_per_engine != 1:
        raise ValueError("LMDeploy currently requires --colocate and --rollout-num-gpus-per-engine 1")
    if args.lmdeploy_max_batch_size < 1:
        raise ValueError("--lmdeploy-max-batch-size must be positive")
    unsupported = {
        "fully async training": args.fully_async,
        "multi-node colocated transport": args.actor_num_nodes != 1,
        "partial rollout": args.partial_rollout,
        "external engine discovery": args.rollout_external,
        "LoRA": args.lora_rank > 0,
        "in-place generation pause": args.pause_generation_mode == "in_place",
        "weight checksum checks": args.check_weight_update_equal,
        "GPU peak-memory handoff": args.colocate_memory_peak_device == "gpu",
        "sampling-support replay": args.rollout_top_p != 1.0 or args.rollout_top_k not in (-1, 0),
        "speculative decoding": bool(args.sglang.base_args["speculative_algorithm"]),
        "router key affinity": bool(args.use_session_server),
        "expert routing replay": args.use_rollout_routing_replay,
        "indexer replay": args.use_rollout_indexer_replay,
    }
    if enabled := [name for name, value in unsupported.items() if value]:
        raise ValueError(f"LMDeploy's synchronous dense rollout integration does not support: {', '.join(enabled)}")


def compute_lmdeploy_launch_cmd(*, args, model_path, host, port, gate_port, base_gpu_id, cache_fraction) -> str:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    device = visible_devices.split(",")[base_gpu_id] if visible_devices else str(base_gpu_id)
    # The serving environment owns its CUDA libraries and package search path.
    # Inheriting the trainer's LD_LIBRARY_PATH would load a different Torch runtime.
    command = [
        "env",
        "-u",
        "LD_LIBRARY_PATH",
        "-u",
        "PYTHONPATH",
        f"CUDA_VISIBLE_DEVICES={device}",
        args.lmdeploy_python,
        "-m",
        "lmdeploy.serve.miles",
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--gate-port",
        str(gate_port),
        "--max-batch-size",
        str(args.lmdeploy_max_batch_size),
        "--cache-fraction",
        str(cache_fraction),
    ]
    if api_key := args.sglang.base_args["api_key"]:
        command.extend(["--api-key", api_key])
    return shlex.join(command)
