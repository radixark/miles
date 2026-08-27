"""
This file is not for miles framework itself, but as an optional utility to easily launch miles jobs and tests.
"""

import base64
import datetime
import json
import os
import platform
import random
import shlex
import socket
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import get_args

from miles.utils.external_utils.exec_command import exec_command_cpu, exec_command_gpu, exec_command_multi_node
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.file_arg_utils import PSEUDO_FILE_PREFIX
from miles.utils.http_utils import wait_for_server_ready
from miles.utils.typer_utils import dataclass_cli

_ = exec_command_cpu, exec_command_gpu, exec_command_multi_node, dataclass_cli

repo_base_dir = Path(os.path.abspath(__file__)).resolve().parents[3]


def _pythonpath_with_sources(megatron_path: str, *additional_pythonpaths: str | None) -> str:
    entries = [str(repo_base_dir), megatron_path]
    for pythonpath in (*additional_pythonpaths, os.environ.get("PYTHONPATH")):
        if pythonpath:
            entries.extend(pythonpath.split(os.pathsep))
    return os.pathsep.join(dict.fromkeys(entries))


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
        print(f"convert_checkpoint skip {path_dst} since tracker is 'release'")
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
        print(f"fp8_cast_bf16 skip {path_dst} since {sentinel} exists")
        return

    exec_command_gpu(
        f"python {repo_base_dir}/tools/fp8_cast_bf16.py "
        f"--input-fp8-hf-path {path_src} "
        f"--output-bf16-hf-path {path_dst} "
    )


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


def _parse_extra_env_vars(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return {kv[0]: kv[1] for item in text.split(" ") if item.strip() != "" if (kv := item.split("=")) or True}


def check_has_nvlink():
    output = exec_command_gpu("nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l", capture_output=True)
    return int(output) > 0


def get_default_wandb_args(test_file: str, run_name_prefix: str | None = None, run_id: str | None = None):
    if not os.environ.get("WANDB_API_KEY"):
        print("Skip wandb configuration since WANDB_API_KEY is not found")
        return ""

    test_file = Path(test_file)
    test_name = test_file.stem
    if len(test_name) < 6:
        test_name = f"{test_file.parent.name}_{test_name}"

    wandb_run_name = run_id or create_run_id()
    if (x := os.environ.get("GITHUB_COMMIT_NAME")) is not None:
        wandb_run_name += f"_{x}"
    if (x := run_name_prefix) is not None:
        wandb_run_name = f"{x}_{wandb_run_name}"

    # Use the actual key value from environment to avoid shell expansion issues
    wandb_key = os.environ.get("WANDB_API_KEY")
    return (
        "--use-wandb "
        f"--wandb-project miles-{test_name} "
        f"--wandb-group {wandb_run_name} "
        f"--wandb-key '{wandb_key}' "
        "--disable-wandb-random-suffix "
    )


def create_run_id() -> str:
    return datetime.datetime.utcnow().strftime("%y%m%d-%H%M%S") + f"-{random.Random().randint(0, 999):03d}"


_warned_bool_env_var_keys = set()


# copied from SGLang
def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        if value not in _warned_bool_env_var_keys:
            print(f"get_bool_env_var({name}) see non-understandable value={value} and treat as false")
        _warned_bool_env_var_keys.add(value)

    return value in truthy_values


def get_env_enable_infinite_run():
    return get_bool_env_var("MILES_TEST_ENABLE_INFINITE_RUN", "false")


MOONCAKE_MASTER_PORT = 50051
MOONCAKE_MASTER_METRICS_PORT = 0
MOONCAKE_MASTER_LOG_PATH = Path("/tmp/mooncake_master.log")


def get_mooncake_object_store_args(master_port: int = MOONCAKE_MASTER_PORT) -> str:
    init_kwargs = {
        "protocol": "tcp",
        "master_server_address": f"127.0.0.1:{master_port}",
        "global_segment_size": "2gb",
        "local_buffer_size": "2gb",
    }
    return "--object-store-backend mooncake " f"--mooncake-store-init-kwargs {shlex.quote(json.dumps(init_kwargs))} "


def _is_tcp_server_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def start_mooncake_master(
    rpc_port: int = MOONCAKE_MASTER_PORT,
    metrics_port: int = MOONCAKE_MASTER_METRICS_PORT,
    timeout: float = 30,
    log_path: str | Path = MOONCAKE_MASTER_LOG_PATH,
) -> None:
    host = "127.0.0.1"
    if _is_tcp_server_ready(host, rpc_port):
        print(f"Mooncake master is already ready at {host}:{rpc_port}", flush=True)
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


def encode_pseudo_file(text: str) -> str:
    return PSEUDO_FILE_PREFIX + base64.b64encode(text.encode()).decode()


NUM_GPUS_OF_HARDWARE = {
    "H100": 8,
    "H200": 8,
    "B200": 8,
    "B300": 8,
    "GB200": 4,
    "GB300": 4,
    "MI350X": 8,
    "MI355X": 8,
}

GENERATION_HARDWARE = {
    "H100": "Hopper",
    "H200": "Hopper",
    "B200": "Blackwell",
    "B300": "Blackwell",
    "GB200": "Blackwell",
    "GB300": "Blackwell",
}


def detect_hardware() -> str:
    """Which NUM_GPUS_OF_HARDWARE entry this node is. Call it where the answer is used: prepare steps run GPU-free."""
    import torch

    assert torch.cuda.is_available(), "no visible GPU to detect the hardware from, pass --hardware explicitly"
    name = torch.cuda.get_device_name()
    if torch.version.hip is not None:
        detected = next((hardware for hardware in ("MI350X", "MI355X") if hardware in name), None)
    else:
        grace = platform.machine() == "aarch64"
        match torch.cuda.get_device_capability():
            case (9, 0):
                detected = "H200" if torch.cuda.get_device_properties(0).total_memory > 100 * 1024**3 else "H100"
            case (10, 0):
                detected = "GB200" if grace else "B200"
            case (10, 3):
                detected = "GB300" if grace else "B300"
            case _:
                detected = None
    assert detected is not None, f"cannot tell which hardware {name!r} is, pass --hardware explicitly"
    return detected


def resolve_hardware(config: ExecuteTrainConfig) -> str:
    """`auto` asks the node the launcher runs on; anything explicit overrides it."""
    if config.hardware == "auto":
        hardware = detect_hardware()
        print(f"detected --hardware {hardware}")
    else:
        hardware = config.hardware
    supported = get_args(config.__dataclass_fields__["hardware"].type)
    assert hardware in supported, f"{type(config).__name__} has no verified profile for {hardware}"
    return hardware
