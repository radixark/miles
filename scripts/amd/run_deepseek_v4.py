"""
AMD DeepSeek V4 training launcher.

This keeps the upstream DeepSeek V4 preparation flow:
  FP8 HF checkpoint -> BF16 HF checkpoint -> Megatron torch_dist checkpoint.
The train command runs Miles async mode with separate actor and rollout nodes.
"""

import os
import shlex
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import miles.utils.external_utils.command_utils as U
from scripts import run_deepseek_v4 as upstream

app = typer.Typer()

_FULLY_ASYNC_DIR = _REPO_ROOT / "examples" / "fully_async"
_DSV4_NUM_EXPERTS = 256
_DSV4_NUM_LAYERS = 43

_NETWORK_ENV_NAMES = (
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
)

_ROCM_ENV_DEFAULTS = {
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "0",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "USE_ROCM": "1",
    "USE_CUDA": "0",
    "ROCM_HOME": "/opt/rocm",
    "ROCM_PATH": "/opt/rocm",
}

_RCCL_TUNING_DEFAULTS = {
    "NCCL_IB_TC": "160",
    "NCCL_IB_TIMEOUT": "22",
    "NCCL_IB_RETRY_CNT": "7",
    "NCCL_IB_QPS_PER_CONNECTION": "2",
    "NCCL_IB_SPLIT_DATA_ON_QPS": "0",
    "NCCL_MIN_NCHANNELS": "32",
    "NCCL_MAX_NCHANNELS": "32",
    "NCCL_ALGO": "Ring",
    "NCCL_PXN_DISABLE": "0",
    "NCCL_NET_GDR_LEVEL": "2",
}

_NCCL_DEBUG_ENV_DEFAULTS = {
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,NET,COLL",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "200000",
    "TORCH_FR_BUFFER_SIZE": "200000",
    "TORCH_NCCL_DESYNC_DEBUG": "1",
}

_MEGATRON_DSV4_ENV_DEFAULTS = {
    "MILES_DSV4_CKPT_VERSION": "2604",
    "MILES_DSV4_2604_SUBMODE": "2604B",
    "MEGATRON_USE_KV_QAT": "1",
}

_MILES_ASYNC_ENV_DEFAULTS = {
    "HIP_FORCE_DEV_KERNARG": "1",
    "HSA_NO_SCRATCH_RECLAIM": "1",
    "ROCM_QUICK_REDUCE_QUANTIZATION": "INT8",
    "MILES_MBRIDGE_MEMORY_EFFICIENT_LOAD": "1",
    "MILES_TE_ADAM_CHUNK_ELEMS": "8000000",
    "AITER_BF16_FP8_MOE_BOUND": "1",
    "MC_TRANSFER_TIMEOUT": "300",
    "RAY_DEDUP_LOGS": "0",
}

_SGLANG_DSV4_PERF_ENV_DEFAULTS = {
    "SGLANG_APPLY_CONFIG_BACKUP": "none",
    "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_ENABLE_THINKING": "1",
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_ROCM700A": "1",
    "SGLANG_MOE_PADDING": "1",
    "SGLANG_SET_CPU_AFFINITY": "0",
    "SGLANG_ROCM_FUSED_DECODE_MLA": "1",
    "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
    "SGLANG_FORCE_TRITON_MOE_FP8": "1",
    "SGLANG_OPT_USE_TILELANG_SWA_PREPARE": "false",
    "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK": "false",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "false",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "false",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "false",
    "SGLANG_OPT_USE_AITER_MHC_PRE": "true",
    "SGLANG_OPT_USE_AITER_MHC_POST": "true",
    "SGLANG_OPT_USE_TRITON_SWA_PREPARE": "true",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "true",
    "SGLANG_OPT_DPSK_V4_RADIX": "1",
    "SGLANG_OPT_USE_OLD_COMPRESSOR": "false",
    "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
    "SGLANG_OPT_USE_FUSED_PAGED_COMPRESS": "true",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "false",
    "SGLANG_OPT_USE_FUSED_STORE_CACHE": "true",
    "SGLANG_OPT_FUSE_WQA_WKV": "true",
    "SGLANG_TOPK_TRANSFORM_512_TORCH": "0",
    "SGLANG_OPT_USE_TILELANG_INDEXER": "true",
    "SGLANG_HACK_FLASHMLA_BACKEND": "triton",
    "SGLANG_REASONING_EFFORT": "max",
}


@dataclass
class ScriptArgs(upstream.ScriptArgs):
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    run_id: str = U.create_run_id()

    num_nodes: int = 4
    actor_num_nodes: int | None = None
    rollout_num_nodes: int = 1
    num_gpus_per_node: int = 8

    tensor_model_parallel_size: int = 8
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 8
    expert_tensor_parallel_size: int = 1
    decoder_first_pipeline_num_layers: int | None = None
    decoder_last_pipeline_num_layers: int | None = None
    num_layers: int = _DSV4_NUM_LAYERS

    enable_eval: bool = False
    enable_r3: bool = False
    fp8_training: bool = False
    optimizer_offload: bool = True
    skip_saving: bool = True
    use_fault_tolerance: bool = True
    train_deterministic: bool = True

    context_length: int | None = None
    rollout_num_gpus_per_engine: int | None = None
    rollout_batch_size: int | None = None
    num_rollout: int | None = None
    n_samples_per_prompt: int | None = None
    rollout_max_response_len: int | None = None
    num_steps_per_rollout: int | None = None
    micro_batch_size: int = 1
    max_tokens_per_gpu: int | None = None
    log_probs_max_tokens_per_gpu: int | None = None
    qkv_format: Literal["bshd", "thd"] = "bshd"

    sglang_mem_fraction_static: float | None = None
    sglang_max_running_requests: int | None = None
    sglang_max_total_tokens: int | None = None
    sglang_moe_runner_backend: Literal["triton"] = "triton"
    sglang_disable_cuda_graph: bool = False

    pause_generation_mode: Literal["in_place", "retract", "abort"] = "in_place"
    update_weight_transfer_mode: Literal["broadcast"] = "broadcast"
    max_weight_staleness: int | None = None
    train_memory_margin_bytes: int = 16 * 1024 * 1024 * 1024
    update_weight_buffer_size: int = 256 * 1024 * 1024
    disable_weights_backuper: bool = True
    replicate_torch_dist_release: bool = True

    external_ray_required: bool = True
    nccl_debug: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.actor_num_nodes is None:
            self.actor_num_nodes = self.num_nodes - self.rollout_num_nodes
        if self.rollout_num_gpus_per_engine is None:
            self.rollout_num_gpus_per_engine = self.num_gpus_per_node


def _pick(args: ScriptArgs, name: str, debug_value, normal_value):
    value = getattr(args, name)
    if value is not None:
        return value
    return debug_value if args.mode == "debug_minimal" else normal_value


@contextmanager
def _without_ray_address():
    old_value = os.environ.pop("RAY_ADDRESS", None)
    try:
        yield
    finally:
        if old_value is not None:
            os.environ["RAY_ADDRESS"] = old_value


def _network_env() -> dict[str, str]:
    return {name: os.environ[name] for name in _NETWORK_ENV_NAMES if os.environ.get(name)}


def _env_defaults(defaults: dict[str, str]) -> dict[str, str]:
    return {name: os.environ.get(name, value) for name, value in defaults.items()}


def _nccl_debug_env(args: ScriptArgs) -> dict[str, str]:
    if not args.nccl_debug:
        return {}
    return _env_defaults(_NCCL_DEBUG_ENV_DEFAULTS)


def _validate_layout(args: ScriptArgs) -> None:
    if args.num_nodes <= 1:
        raise ValueError("AMD DeepSeek V4 currently uses async multi-node training.")
    if args.rollout_num_nodes <= 0:
        raise ValueError("rollout_num_nodes must be positive.")
    if args.actor_num_nodes is None or args.actor_num_nodes <= 0:
        raise ValueError("actor_num_nodes must be positive.")
    if args.actor_num_nodes + args.rollout_num_nodes != args.num_nodes:
        raise ValueError(
            "actor_num_nodes + rollout_num_nodes must equal num_nodes: "
            f"{args.actor_num_nodes} + {args.rollout_num_nodes} != {args.num_nodes}."
        )
    if args.external_ray_required and os.environ.get("MILES_SCRIPT_EXTERNAL_RAY") != "1":
        raise ValueError(
            "AMD multi-node runs must use an existing Ray cluster. Start Ray on all nodes, "
            "then run this launcher on the head with MILES_SCRIPT_EXTERNAL_RAY=1."
        )
    if args.update_weight_transfer_mode != "broadcast":
        raise ValueError("AMD DeepSeek V4 currently supports broadcast weight transfer only.")

    train_world_size = args.actor_num_nodes * args.num_gpus_per_node
    model_parallel_size = (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    )
    if train_world_size % model_parallel_size != 0:
        raise ValueError(
            "actor world size must be divisible by tensor*pipeline*context parallel size: "
            f"world={train_world_size}, tp={args.tensor_model_parallel_size}, "
            f"pp={args.pipeline_model_parallel_size}, cp={args.context_parallel_size}."
        )

    if args.expert_model_parallel_size <= 0 or _DSV4_NUM_EXPERTS % args.expert_model_parallel_size != 0:
        raise ValueError(
            f"DeepSeek-V4-Flash has {_DSV4_NUM_EXPERTS} routed experts; "
            f"expert_model_parallel_size must divide {_DSV4_NUM_EXPERTS}."
        )
    expert_model_pipeline_size = (
        args.expert_tensor_parallel_size * args.expert_model_parallel_size * args.pipeline_model_parallel_size
    )
    if train_world_size % expert_model_pipeline_size != 0:
        raise ValueError(
            "actor world size must be divisible by expert_tensor*expert*pipeline parallel size: "
            f"world={train_world_size}, etp={args.expert_tensor_parallel_size}, "
            f"ep={args.expert_model_parallel_size}, pp={args.pipeline_model_parallel_size}."
        )

    _resolve_uneven_pipeline_split(args)


def _resolve_uneven_pipeline_split(args: ScriptArgs) -> None:
    if args.pipeline_model_parallel_size <= 1:
        return
    if args.decoder_first_pipeline_num_layers is not None or args.decoder_last_pipeline_num_layers is not None:
        return
    if args.num_layers % args.pipeline_model_parallel_size == 0:
        return

    layers_per_early_stage = (args.num_layers + args.pipeline_model_parallel_size - 1) // args.pipeline_model_parallel_size
    args.decoder_last_pipeline_num_layers = args.num_layers - layers_per_early_stage * (
        args.pipeline_model_parallel_size - 1
    )
    if args.decoder_last_pipeline_num_layers <= 0:
        raise ValueError(
            f"Cannot compute uneven pipeline split for num_layers={args.num_layers}, "
            f"pipeline_model_parallel_size={args.pipeline_model_parallel_size}."
        )


def _prepare_spmd(args: ScriptArgs):
    _validate_layout(args)
    extra_args = _conversion_parallel_args(args)

    num_gpus_for_convert = args.num_gpus_per_node
    if args.model_name == "DeepSeek-V4-Flash-FP8-4layer":
        num_gpus_for_convert = min(num_gpus_for_convert, 4)

    _convert_checkpoint_rocm(
        args=args,
        num_gpus_per_node=num_gpus_for_convert,
        extra_args=extra_args,
    )


def _conversion_parallel_args(args: ScriptArgs) -> str:
    if (
        args.model_name == "DeepSeek-V4-Flash-FP8"
        and args.actor_num_nodes == 3
        and args.num_gpus_per_node == 8
    ):
        return (
            "--tensor-model-parallel-size 1 "
            "--pipeline-model-parallel-size 6 "
            "--decoder-last-pipeline-num-layers 3 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 4 "
            "--expert-tensor-parallel-size 1 "
        )

    extra_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        f"--pipeline-model-parallel-size {args.pipeline_model_parallel_size} "
        f"--context-parallel-size {args.context_parallel_size} "
        f"--expert-model-parallel-size {args.expert_model_parallel_size} "
        f"--expert-tensor-parallel-size {args.expert_tensor_parallel_size} "
    )
    if args.decoder_first_pipeline_num_layers is not None:
        extra_args += f"--decoder-first-pipeline-num-layers {args.decoder_first_pipeline_num_layers} "
    if args.decoder_last_pipeline_num_layers is not None:
        extra_args += f"--decoder-last-pipeline-num-layers {args.decoder_last_pipeline_num_layers} "
    if args.model_name == "DeepSeek-V4-Pro-FP8":
        extra_args += "--make-vocab-size-divisible-by 32 "
    return extra_args


def _convert_checkpoint_rocm(args: ScriptArgs, num_gpus_per_node: int, extra_args: str) -> None:
    path_dst = f"{args.model_dir}/{args.torch_dist_name}"
    tracker = Path(path_dst) / "latest_checkpointed_iteration.txt"
    if tracker.exists() and tracker.read_text().strip() == "release":
        print(f"convert_checkpoint skip {path_dst} since tracker is 'release'")
        with _without_ray_address():
            _normalize_torch_dist_checkpoint(
                path_dst,
                args.actor_num_nodes,
                replicate_release=args.replicate_torch_dist_release,
            )
        return

    env = _conversion_env(args)
    export_env = _format_exports(env)
    multinode_args = ""
    if args.actor_num_nodes and args.actor_num_nodes > 1:
        multinode_args = (
            "--master-addr {{master_addr}} "
            "--master-port 23456 "
            "--nnodes={{nnodes}} "
            "--node-rank {{node_rank}} "
        )

    command = (
        f"{export_env} "
        f"source {_REPO_ROOT}/scripts/models/{args.megatron_model_type}.sh && "
        "torchrun "
        f"--nproc-per-node {num_gpus_per_node} "
        f"{multinode_args}"
        f"{_REPO_ROOT}/tools/convert_hf_to_torch_dist.py "
        "${MODEL_ARGS[@]} "
        f"--hf-checkpoint {args.model_dir}/{args.bf16_name} "
        f"--save {path_dst} "
        f"{extra_args}"
    )
    with _without_ray_address():
        U.exec_command_all_ray_node(command, num_nodes=args.actor_num_nodes)
        _normalize_torch_dist_checkpoint(
            path_dst,
            args.actor_num_nodes,
            replicate_release=args.replicate_torch_dist_release,
        )


def _normalize_torch_dist_checkpoint(
    path_dst: str,
    num_nodes: int | None,
    replicate_release: bool = True,
) -> None:
    if not num_nodes or num_nodes <= 1:
        return

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address="auto", ignore_reinit_error=True)
    try:
        current_ip = ray._private.services.get_node_ip_address().strip("[]")
        nodes = sorted(
            [node for node in ray.nodes() if node.get("Alive")],
            key=lambda node: (node["NodeManagerAddress"] != current_ip, node["NodeManagerAddress"]),
        )[:num_nodes]

        @ray.remote(num_cpus=0.001)
        def read_common_files(path: str) -> dict[str, bytes]:
            release = Path(path) / "release"
            files = {}
            for name in (".metadata", "common.pt", "metadata.json"):
                file_path = release / name
                if file_path.exists():
                    files[name] = file_path.read_bytes()
            if ".metadata" not in files:
                raise FileNotFoundError(f"Missing torch_dist metadata under {release}")
            return files

        @ray.remote(num_cpus=0.001)
        def normalize_local_checkpoint(path: str, common_files: dict[str, bytes]) -> str:
            root = Path(path)
            release = root / "release"
            iteration = root / "iter_0000001"
            if iteration.exists() and not release.exists():
                iteration.rename(release)
            release.mkdir(parents=True, exist_ok=True)
            for name, data in common_files.items():
                (release / name).write_bytes(data)
            (root / "latest_checkpointed_iteration.txt").write_text("release\n")
            return str(root)

        common_files = ray.get(
            read_common_files.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=nodes[0]["NodeID"], soft=False)
            ).remote(path_dst)
        )
        ray.get(
            [
                normalize_local_checkpoint.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
                ).remote(path_dst, common_files)
                for node in nodes
            ]
        )
        if replicate_release:
            _replicate_torch_dist_release(path_dst, nodes)
    finally:
        ray.shutdown()


def _replicate_torch_dist_release(path_dst: str, nodes: list[dict]) -> None:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=0.001)
    def list_release_files(path: str) -> dict[str, int]:
        release = Path(path) / "release"
        return {file.name: file.stat().st_size for file in release.iterdir() if file.is_file()}

    @ray.remote(num_cpus=0.001)
    def start_release_server(path: str, port_base: int) -> tuple[int, int]:
        import socket
        import subprocess
        import time

        release = Path(path) / "release"
        for port in range(port_base, port_base + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("127.0.0.1", port)) != 0:
                    break
        else:
            raise RuntimeError(f"No free port found in [{port_base}, {port_base + 100})")

        proc = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(port), "--bind", "0.0.0.0", "--directory", str(release)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.time() + 10
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    return port, proc.pid
            time.sleep(0.2)
        raise RuntimeError(f"Failed to start torch_dist release file server on port {port}")

    @ray.remote(num_cpus=0.001)
    def stop_release_server(pid: int) -> None:
        import os
        import signal

        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                return

    @ray.remote(num_cpus=0.001)
    def verify_release_files(path: str, manifest: dict[str, int]) -> tuple[str, int]:
        import socket

        release = Path(path) / "release"
        for name, size in manifest.items():
            file = release / name
            if not file.exists() or file.stat().st_size != size:
                raise FileNotFoundError(f"{socket.gethostname()} missing replicated checkpoint file {file}")
        return str(release), len(manifest)

    @ray.remote(num_cpus=0.001)
    def download_missing_release_files(
        path: str,
        manifest: dict[str, int],
        sources: dict[str, tuple[str, int]],
    ) -> tuple[str, int, int]:
        import os
        import urllib.parse
        import urllib.request

        release = Path(path) / "release"
        release.mkdir(parents=True, exist_ok=True)
        copied = 0
        skipped = 0
        for name, size in sorted(manifest.items()):
            dst = release / name
            if dst.exists() and dst.stat().st_size == size:
                skipped += 1
                continue
            src_ip, src_port = sources[name]
            tmp = release / f".{name}.tmp"
            url = f"http://{src_ip}:{src_port}/{urllib.parse.quote(name)}"
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size != size:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"Incomplete copy for {dst}: got {tmp.stat().st_size}, expected {size}")
            os.replace(tmp, dst)
            copied += 1
        return str(release), copied, skipped

    file_maps = ray.get(
        [
            list_release_files.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            ).remote(path_dst)
            for node in nodes
        ]
    )
    manifest: dict[str, int] = {}
    source_node_by_file: dict[str, dict] = {}
    for node, files in zip(nodes, file_maps, strict=True):
        for name, size in files.items():
            if name not in manifest:
                manifest[name] = size
                source_node_by_file[name] = node
            elif manifest[name] != size:
                raise RuntimeError(f"Mismatched torch_dist release file size for {name}")

    total_gib = sum(manifest.values()) / (1024**3)
    print(f"Replicating torch_dist release across actor nodes: {len(manifest)} files, {total_gib:.1f} GiB", flush=True)

    port_base = int(os.environ.get("MILES_TORCH_DIST_REPLICA_PORT_BASE", "27640"))
    server_infos = ray.get(
        [
            start_release_server.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            ).remote(path_dst, port_base + index * 100)
            for index, node in enumerate(nodes)
        ]
    )
    server_ports = [port for port, _pid in server_infos]
    server_pids = [pid for _port, pid in server_infos]

    sources = {
        name: (
            source_node_by_file[name]["NodeManagerAddress"],
            server_ports[nodes.index(source_node_by_file[name])],
        )
        for name in manifest
    }
    try:
        results = ray.get(
            [
                download_missing_release_files.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
                ).remote(path_dst, manifest, sources)
                for node in nodes
            ]
        )
        for release, copied, skipped in results:
            print(f"torch_dist release ready on {release}: copied={copied}, skipped={skipped}", flush=True)
        ray.get(
            [
                verify_release_files.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
                ).remote(path_dst, manifest)
                for node in nodes
            ]
        )
    finally:
        ray.get(
            [
                stop_release_server.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
                ).remote(pid)
                for node, pid in zip(nodes, server_pids, strict=True)
            ]
        )


def _base_env(args: ScriptArgs, *, include_async_path: bool = False) -> dict[str, str]:
    python_paths = [str(_REPO_ROOT)]
    if include_async_path:
        python_paths.append(str(_FULLY_ASYNC_DIR))
    python_paths.extend([args.megatron_path, os.environ.get("PYTHONPATH", "")])
    visible_devices = ",".join(str(i) for i in range(args.num_gpus_per_node))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    no_proxy = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or f"localhost,127.0.0.1,{master_addr}"
    return {
        "PYTHONPATH": os.pathsep.join(path for path in python_paths if path),
        "MASTER_ADDR": master_addr,
        "no_proxy": no_proxy,
        "NO_PROXY": no_proxy,
        "HIP_VISIBLE_DEVICES": visible_devices,
        "CUDA_VISIBLE_DEVICES": visible_devices,
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
        **_env_defaults(_ROCM_ENV_DEFAULTS),
        **_env_defaults(_RCCL_TUNING_DEFAULTS),
        **_nccl_debug_env(args),
        **_env_defaults(_MEGATRON_DSV4_ENV_DEFAULTS),
        **_network_env(),
    }


def _conversion_env(args: ScriptArgs) -> dict[str, str]:
    return {
        **_base_env(args),
        "MILES_HACK_TRAIN_TORCH_DETERMINISTIC": "1",
    }


def _format_exports(env: dict[str, str]) -> str:
    return " ".join(f"export {key}={shlex.quote(str(value))};" for key, value in env.items())


def _prepare_cp(args: ScriptArgs):
    _rsync_simple(
        path_src=f"{args.model_dir}/{args.torch_dist_name}",
        path_dst=f"{args.model_local_dir}/{args.torch_dist_name}",
        num_nodes=args.num_nodes,
    )
    _rsync_simple(
        path_src=f"{args.model_dir}/{args.model_name}",
        path_dst=f"{args.model_local_dir}/{args.model_name}",
        num_nodes=args.num_nodes,
    )


def _rsync_simple(path_src: str, path_dst: str, num_nodes: int) -> None:
    with _without_ray_address():
        U.exec_command_all_ray_node(
            f"mkdir -p {path_dst} && rsync -a --info=progress2 {path_src}/ {path_dst}",
            num_nodes=num_nodes,
        )


def _train(args: ScriptArgs):
    _validate_layout(args)
    upstream._ensure_4layer_model_type(args)

    hf_checkpoint = args.hf_checkpoint or f"{args.model_local_dir}/{args.model_name}"
    if args.replicate_torch_dist_release:
        with _without_ray_address():
            _normalize_torch_dist_checkpoint(
                f"{args.model_local_dir}/{args.torch_dist_name}",
                args.actor_num_nodes,
                replicate_release=True,
            )

    load_save_path = f"{args.save_dir}/{args.run_id}/checkpoints"
    rollout_num_gpus = args.rollout_num_nodes * args.num_gpus_per_node
    rollout_engine_size = args.rollout_num_gpus_per_engine or args.num_gpus_per_node

    ckpt_args = f"--hf-checkpoint {hf_checkpoint} --ref-load {args.model_local_dir}/{args.torch_dist_name} "
    if not args.skip_saving:
        ckpt_args += (
            f"--load {load_save_path} "
            f"--save {load_save_path} "
            "--save-interval 20 "
            "--save-retain-interval 20 "
        )

    context_length = _pick(args, "context_length", 512, 16384)
    response_len = _pick(args, "rollout_max_response_len", 64, 8192)
    rollout_max_prompt_len = max(1, context_length - response_len)
    num_rollout = _pick(args, "num_rollout", 1, 3000)
    n_samples = _pick(args, "n_samples_per_prompt", 1, 8)
    num_steps = _pick(args, "num_steps_per_rollout", 1, 1)
    max_tokens_per_gpu = _pick(args, "max_tokens_per_gpu", 512, 2048)
    log_probs_max_tokens_per_gpu = args.log_probs_max_tokens_per_gpu or max_tokens_per_gpu
    actor_dp_size = (
        args.actor_num_nodes
        * args.num_gpus_per_node
        // (
            args.tensor_model_parallel_size
            * args.pipeline_model_parallel_size
            * args.context_parallel_size
        )
    )
    batch_unit = args.micro_batch_size * actor_dp_size
    rollout_batch_size = _pick(args, "rollout_batch_size", batch_unit, 12)
    if rollout_batch_size % batch_unit != 0:
        raise ValueError(
            "rollout_batch_size must be divisible by micro_batch_size * actor data_parallel_size: "
            f"rollout_batch_size={rollout_batch_size}, micro_batch_size={args.micro_batch_size}, "
            f"actor_dp_size={actor_dp_size}."
        )

    staleness_args = ""
    if args.max_weight_staleness is not None:
        staleness_args = f"--max-weight-staleness {args.max_weight_staleness} "

    rollout_args = (
        "--rollout-function-path fully_async_rollout.generate_rollout_fully_async "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {num_rollout} "
        f"--rollout-batch-size {rollout_batch_size} "
        f"--n-samples-per-prompt {n_samples} "
        f"--rollout-max-context-len {context_length} "
        f"--rollout-max-prompt-len {rollout_max_prompt_len} "
        f"--rollout-max-response-len {response_len} "
        "--rollout-temperature 0.8 "
        f"--num-steps-per-rollout {num_steps} "
        "--balance-data "
        "--log-passrate "
        f"--pause-generation-mode {args.pause_generation_mode} "
        f"{staleness_args}"
    )
    if args.mode != "debug_minimal":
        rollout_args += (
            "--over-sampling-batch-size 512 "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    eval_enabled = args.mode != "debug_minimal" and args.enable_eval
    eval_args = ""
    if eval_enabled:
        eval_args = "--eval-interval 20 --eval-top-p 0.7 "

    match args.task:
        case "dapo_aime":
            rollout_args += (
                f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
                "--input-key prompt "
                """--apply-chat-template-kwargs '{"thinking_mode":"thinking","reasoning_effort":"max"}' """
            )
            if eval_enabled:
                eval_args += (
                    f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
                    "--n-samples-per-eval-prompt 8 "
                    "--eval-max-response-len 4096 "
                )
        case "gsm8k":
            rollout_args += (
                f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
                "--input-key messages "
            )
            if eval_enabled:
                eval_args += (
                    f"--eval-prompt-data gsm8k {args.data_dir}/gsm8k/test.parquet "
                    "--n-samples-per-eval-prompt 1 "
                    "--eval-max-response-len 256 "
                )

    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {args.pipeline_model_parallel_size} "
        f"--context-parallel-size {args.context_parallel_size} "
        f"--expert-model-parallel-size {args.expert_model_parallel_size} "
        f"--expert-tensor-parallel-size {args.expert_tensor_parallel_size} "
        f"--seq-length {context_length} "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        f"--micro-batch-size {args.micro_batch_size} "
        f"--max-tokens-per-gpu {max_tokens_per_gpu} "
        f"--log-probs-max-tokens-per-gpu {log_probs_max_tokens_per_gpu} "
    )
    if args.decoder_first_pipeline_num_layers is not None:
        perf_args += f"--decoder-first-pipeline-num-layers {args.decoder_first_pipeline_num_layers} "
    if args.decoder_last_pipeline_num_layers is not None:
        perf_args += f"--decoder-last-pipeline-num-layers {args.decoder_last_pipeline_num_layers} "

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    if args.optimizer_offload:
        optimizer_args += (
            "--optimizer-cpu-offload "
            "--use-precision-aware-optimizer "
            "--overlap-cpu-optimizer-d2h-h2d "
        )

    sglang_mem_fraction = _pick(args, "sglang_mem_fraction_static", 0.50, 0.70)
    sglang_max_running = _pick(args, "sglang_max_running_requests", 2, 32)
    sglang_max_total_tokens = _pick(args, "sglang_max_total_tokens", 2048, 262144)
    sglang_cuda_graph_args = (
        "--sglang-disable-cuda-graph --sglang-disable-piecewise-cuda-graph "
        if args.sglang_disable_cuda_graph
        else ""
    )
    sglang_args = (
        f"--rollout-num-gpus-per-engine {rollout_engine_size} "
        f"--rollout-num-gpus {rollout_num_gpus} "
        "--sglang-data-parallel-size 1 "
        "--sglang-expert-parallel-size 1 "
        "--sglang-attention-backend compressed "
        "--sglang-page-size 256 "
        f"--sglang-max-running-requests {sglang_max_running} "
        f"--sglang-chunked-prefill-size {context_length} "
        "--sglang-server-concurrency 1024 "
        f"--sglang-context-length {context_length} "
        f"--sglang-max-total-tokens {sglang_max_total_tokens} "
        "--sglang-disable-radix-cache "
        "--sglang-disable-custom-all-reduce "
        "--sglang-disable-shared-experts-fusion "
        f"--sglang-moe-runner-backend {args.sglang_moe_runner_backend} "
        "--sglang-schedule-conservativeness 1.0 "
        f"--sglang-mem-fraction-static {sglang_mem_fraction} "
        "--sglang-tool-call-parser deepseekv4 "
        "--sglang-reasoning-parser deepseek-v4 "
        f"{sglang_cuda_graph_args}"
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--grad-reduce-in-bf16 "
        "--accumulate-allreduce-grads-in-fp32 "
        f"--update-weight-buffer-size {args.update_weight_buffer_size} "
        f"--update-weight-transfer-mode {args.update_weight_transfer_mode} "
        f"--train-memory-margin-bytes {args.train_memory_margin_bytes} "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--model-name deepseekv4 "
        f"--qkv-format {args.qkv_format} "
        "--moe-router-freeze-gate "
        "--freeze-e-score-correction-bias "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
    )
    if args.use_fault_tolerance:
        misc_args += "--use-fault-tolerance "
    if args.disable_weights_backuper:
        misc_args += "--disable-weights-backuper "
    if args.dump_details:
        misc_args += f"--dump-details {args.debug_data_root}/{args.run_id}/dump_details "
    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "
        misc_args += "--use-miles-router "

    extra_env_vars = _extra_env(args)
    if args.train_deterministic:
        misc_args += "--deterministic-mode "
        extra_env_vars |= {
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "MILES_HACK_TRAIN_TORCH_DETERMINISTIC": "1",
        }

    if args.fp8_training:
        misc_args += "--transformer-impl transformer_engine --bf16 --fp8-format e4m3 --fp8-recipe blockwise "
        extra_env_vars["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] = "1"
    else:
        misc_args += "--bf16 "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        train_script="train_async.py",
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


def _extra_env(args: ScriptArgs) -> dict[str, str]:
    env = {
        **_base_env(args, include_async_path=True),
        **_env_defaults(_MILES_ASYNC_ENV_DEFAULTS),
        **_env_defaults(_SGLANG_DSV4_PERF_ENV_DEFAULTS),
    }
    if os.environ.get("WANDB_API_KEY"):
        env["WANDB_MODE"] = os.environ.get("WANDB_MODE", "online")
    return env


@app.command()
@U.dataclass_cli
def prepare_download(args: ScriptArgs):
    upstream._prepare_download(args)


@app.command()
@U.dataclass_cli
def prepare_single(args: ScriptArgs):
    upstream._prepare_single(args)


@app.command()
@U.dataclass_cli
def prepare_spmd(args: ScriptArgs):
    _prepare_spmd(args)


@app.command()
@U.dataclass_cli
def prepare_cp(args: ScriptArgs):
    _prepare_cp(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    upstream._prepare_download(args)

    bf16_dir = Path(f"{args.model_dir}/{args.bf16_name}")
    bf16_sentinel = bf16_dir / "model.safetensors.index.json"
    if not bf16_sentinel.exists():
        upstream._prepare_single(args)
    else:
        print(f"[full_train] Skipping FP8->BF16 cast: {bf16_sentinel} already exists.")

    torch_dist_dir = Path(f"{args.model_dir}/{args.torch_dist_name}")
    torch_dist_sentinel = torch_dist_dir / "latest_checkpointed_iteration.txt"
    if not torch_dist_sentinel.exists():
        _prepare_spmd(args)
    else:
        print(f"[full_train] Skipping BF16->torch_dist conversion: {torch_dist_sentinel} already exists.")
        with _without_ray_address():
            _normalize_torch_dist_checkpoint(
                str(torch_dist_dir),
                args.actor_num_nodes,
                replicate_release=args.replicate_torch_dist_release,
            )

    if args.model_local_dir != args.model_dir:
        _prepare_cp(args)
    else:
        print(f"[full_train] Skipping rsync: model_local_dir == model_dir ({args.model_dir})")

    if args.hf_checkpoint is None:
        args.hf_checkpoint = f"{args.model_local_dir}/{args.model_name}"

    _train(args)


if __name__ == "__main__":
    app()
