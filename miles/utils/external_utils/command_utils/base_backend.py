import json
import logging
import os
import shlex
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import get_args

from miles.utils.external_utils.command_utils.common import (
    MOONCAKE_MASTER_LOG_PATH,
    MOONCAKE_MASTER_METRICS_PORT,
    MOONCAKE_MASTER_PORT,
    _is_tcp_server_ready,
    _parse_extra_env_vars,
    _pythonpath_with_sources,
    detect_hardware,
    get_bool_env_var,
    repo_base_dir,
)
from miles.utils.external_utils.exec_command import exec_command_cpu, exec_command_gpu, exec_command_multi_node
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.http_utils import wait_for_server_ready

logger = logging.getLogger(__name__)


# This class can be extended by concrete scripts
@dataclass
class ExecuteTrainConfig:
    cuda_core_dump: bool = False
    num_nodes: int = field(default_factory=lambda: int(os.environ.get("SLURM_JOB_NUM_NODES", "1")))
    extra_env_vars: str = ""
    output_dir: str = "/root/shared_data"


def resolve_extra_env_vars(extra_env_vars: dict[str, str], config: ExecuteTrainConfig) -> dict[str, str]:
    return {
        **extra_env_vars,
        **_parse_extra_env_vars(config.extra_env_vars),
    }


def execute_train(
    train_args: str,
    num_gpus_per_node: int,
    megatron_model_type: str | None,
    train_script: str = "train.py",
    before_ray_job_submit=None,
    extra_env_vars=None,
    config: ExecuteTrainConfig | None = None,
    megatron_path: str = "/root/Megatron-LM",
):
    if extra_env_vars is None:
        extra_env_vars = {}
    if config is None:
        config = ExecuteTrainConfig()
    if not os.path.isabs(train_script):
        train_script = f"{repo_base_dir}/{train_script}"
    external_ray = get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY")
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")

    train_backend_fsdp = "--train-backend fsdp" in train_args
    assert train_backend_fsdp == (megatron_model_type is None)

    exec_command_cpu(
        "pkill -9 sglang; "
        "sleep 3; "
        f"{'' if external_ray else 'ray stop --force; '}"
        f"{'' if external_ray else 'pkill -9 ray; '}"
        # cannot be run in CI, o/w kill the parent script
        # TODO: do we really need this kill? (or can we instead kill miles)
        # "pkill -9 python; "
        "pkill -9 miles; "
        "sleep 3; "
        f"{'' if external_ray else 'pkill -9 ray; '}"
        # "pkill -9 python; "
        "pkill -9 miles; "
        "pkill -9 redis; "
        "true; "
    )

    if not external_ray:
        exec_command_cpu(
            # will prevent ray from buffering stdout/stderr
            f"export PYTHONUNBUFFERED=1 && "
            f"ray start --head --node-ip-address {master_addr} --num-gpus {num_gpus_per_node} --disable-usage-stats"
        )

    if (f := before_ray_job_submit) is not None:
        f()

    runtime_env_vars = {
        # exported for the submitting client too, but only the runtime env reaches the ray workers
        "PYTHONUNBUFFERED": "1",
        # If setting this in FSDP, the computation communication overlapping may have issues
        **(
            {}
            if train_backend_fsdp
            else {
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            }
        ),
        # a get() default is evaluated eagerly, which would probe even when already decided
        "NCCL_NVLS_ENABLE": os.environ.get("NCCL_NVLS_ENABLE") or str(int(check_has_nvlink())),
        **{
            k: os.environ[k]
            for k in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_DEBUG", "NCCL_DEBUG_FILE")
            if k in os.environ
        },
        "no_proxy": f"127.0.0.1,{master_addr}",
        # This is needed by megatron / torch distributed in multi-node setup
        "MASTER_ADDR": master_addr,
        **(
            {
                "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
                "CUDA_COREDUMP_SHOW_PROGRESS": "1",
                "CUDA_COREDUMP_GENERATION_FLAGS": "skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory",
                "CUDA_COREDUMP_FILE": f"{config.output_dir}/cuda_coredump_%h.%p.%t",
            }
            if config.cuda_core_dump
            else {}
        ),
        **resolve_extra_env_vars(extra_env_vars, config),
    }
    runtime_env_vars["PYTHONPATH"] = _pythonpath_with_sources(megatron_path, runtime_env_vars.get("PYTHONPATH"))
    runtime_env_json = json.dumps({"env_vars": runtime_env_vars})

    if get_bool_env_var("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1"):
        model_args = shell_safe_model_args(megatron_model_type)
        exec_command_cpu(
            f"export no_proxy=127.0.0.1 && export PYTHONUNBUFFERED=1 && "
            f"""ray job submit {'' if 'RAY_ADDRESS' in os.environ else '--address="http://127.0.0.1:8265" '}"""
            f"--runtime-env-json={shlex.quote(runtime_env_json)} "
            f"-- python3 {train_script} "
            f"{model_args} "
            f"{train_args}"
        )


def convert_checkpoint(
    model_name,
    megatron_model_type,
    num_gpus_per_node: int,
    multinode: bool = False,
    num_nodes: int | None = None,
    extra_args: str = "",
    dir_dst: str = "/root",
    hf_checkpoint: str | None = None,
    megatron_path: str = "/root/Megatron-LM",
):
    hf_checkpoint = hf_checkpoint or f"/root/models/{model_name}"

    # TODO shall we make it in host-mapped folder and thus can cache it to speedup CI
    path_dst = f"{dir_dst}/{model_name}_torch_dist"
    tracker = Path(path_dst) / "latest_checkpointed_iteration.txt"
    if tracker.exists() and tracker.read_text().strip() == "release":
        logger.info(f"convert_checkpoint skip {path_dst} since tracker is 'release'")
        return

    multinode_args = ""
    if multinode:
        multinode_args = (
            "--master-addr {{master_addr}} " "--master-port 23456 " "--nnodes={{nnodes}} " "--node-rank {{node_rank}} "
        )

    if multinode:
        fn = partial(exec_command_multi_node, num_nodes=num_nodes)
    else:
        fn = exec_command_gpu
    pythonpath = shlex.quote(_pythonpath_with_sources(megatron_path))
    fn(
        f"PYTHONPATH={pythonpath} "
        f"torchrun "
        f"--nproc-per-node {num_gpus_per_node} "
        f"{multinode_args}"
        f"{repo_base_dir}/tools/convert_hf_to_torch_dist.py "
        f"{shell_safe_model_args(megatron_model_type)} "
        f"--hf-checkpoint {hf_checkpoint} "
        f"--save {path_dst} "
        f"{extra_args}"
    )


def rsync_simple(path_src: str, path_dst: str, num_nodes: int | None = None):
    exec_command_multi_node(
        f"mkdir -p {path_dst} && rsync -a --info=progress2 {path_src}/ {path_dst}", num_nodes=num_nodes
    )


def ssh_start_ray_workers(
    master_addr: str,
    num_gpus_per_node: int,
    hostfile: str = "/root/mpi_rack_hostfile",
    head_host: str | None = None,
):
    """Join every host in an MPI-style hostfile to the ray cluster over ssh, in parallel.

    Ray itself cannot bring up the workers: the head is already running locally and the
    workers have no agent yet. Pass this as `execute_train(before_ray_job_submit=...)` so
    the cluster is complete before the job is submitted.
    """
    head_host = head_host or master_addr
    exec_command_cpu(
        f"for worker_ip in $(awk '{{print $1}}' {hostfile}); do "
        f'if [ "$worker_ip" = {shlex.quote(head_host)} ]; then continue; fi; '
        'echo "Starting Ray worker on $worker_ip"; '
        'ssh root@"$worker_ip" '
        '"pkill -9 sglang ; ray stop --force ; pkill -9 miles ; '
        f"ray start --address={master_addr}:6379 --num-gpus {num_gpus_per_node} "
        '--node-ip-address $worker_ip --disable-usage-stats" & '
        "done; wait"
    )


def hf_download_dataset(full_name: str, data_dir: str = "/root/datasets"):
    _, partial_name = full_name.split("/")
    exec_command_cpu(f"hf download --repo-type dataset {full_name} --local-dir {data_dir}/{partial_name}")


def fp8_cast_bf16(path_src, path_dst):
    sentinel = Path(path_dst) / "model.safetensors.index.json"
    if sentinel.exists():
        logger.info(f"fp8_cast_bf16 skip {path_dst} since {sentinel} exists")
        return

    exec_command_gpu(
        f"python {repo_base_dir}/tools/fp8_cast_bf16.py "
        f"--input-fp8-hf-path {path_src} "
        f"--output-bf16-hf-path {path_dst} "
    )


def check_has_nvlink():
    output = exec_command_gpu("nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l", capture_output=True)
    return int(output) > 0


def start_mooncake_master(
    rpc_port: int = MOONCAKE_MASTER_PORT,
    metrics_port: int = MOONCAKE_MASTER_METRICS_PORT,
    timeout: float = 30,
    log_path: str | Path = MOONCAKE_MASTER_LOG_PATH,
) -> None:
    host = "127.0.0.1"
    if _is_tcp_server_ready(host, rpc_port):
        logger.info(f"Mooncake master is already ready at {host}:{rpc_port}")
        return

    log_path = Path(log_path)
    quoted_log_path = shlex.quote(str(log_path))
    exec_command_cpu(
        "pkill -x mooncake_master >/dev/null 2>&1 || true; "
        f"(setsid mooncake_master --rpc_port {rpc_port} --metrics_port {metrics_port} "
        f"> {quoted_log_path} 2>&1 &)"
    )
    try:
        wait_for_server_ready(host, rpc_port, timeout=timeout)
    except RuntimeError as exc:
        exec_command_cpu("pkill -x mooncake_master >/dev/null 2>&1 || true")
        try:
            log_lines = log_path.read_text(errors="replace").splitlines()
            log_tail = "\n".join(log_lines[-100:]) or "<empty>"
        except OSError as log_error:
            log_tail = f"<unable to read {log_path}: {log_error}>"
        raise RuntimeError(
            f"Mooncake master at {host}:{rpc_port} did not become ready.\n"
            f"Last 100 lines of {log_path}:\n{log_tail}"
        ) from exc


def resolve_hardware(config: ExecuteTrainConfig) -> str:
    """`auto` asks the node the launcher runs on; anything explicit overrides it."""
    if config.hardware == "auto":
        hardware = detect_hardware()
        logger.info(f"detected --hardware {hardware}")
    else:
        hardware = config.hardware
    supported = get_args(config.__dataclass_fields__["hardware"].type)
    assert hardware in supported, f"{type(config).__name__} has no verified profile for {hardware}"
    return hardware
