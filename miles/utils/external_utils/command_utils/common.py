import base64
import datetime
import json
import os
import random
import shlex
import socket
import subprocess
from functools import partial
from pathlib import Path

from miles.utils.external_utils.command_utils import base_backend
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainRequest
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.file_arg_utils import PSEUDO_FILE_PREFIX
from miles.utils.http_utils import wait_for_server_ready

repo_base_dir = Path(os.path.abspath(__file__)).resolve().parents[4]


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
        fn = partial(base_backend.exec_command_multi_node, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
    else:
        fn = partial(base_backend.exec_command_gpu, num_gpus_per_node=num_gpus_per_node)
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
    base_backend.exec_command_multi_node(rsync_command(path_src=path_src, path_dst=path_dst), num_nodes=num_nodes)


def rsync_command(*, path_src: str, path_dst: str, lock_path: str | None = None) -> str:
    copy = f"mkdir -p {path_dst} && rsync -a --info=progress2 {path_src}/ {path_dst}"
    if lock_path is None:
        return copy
    lock_dir = shlex.quote(str(Path(lock_path).parent))
    return f"mkdir -p {lock_dir} && flock {shlex.quote(lock_path)} bash -c {shlex.quote(copy)}"


def hf_download_dataset(full_name: str, data_dir: str = "/root/datasets"):
    _, partial_name = full_name.split("/")
    base_backend.exec_command_cpu(f"hf download --repo-type dataset {full_name} --local-dir {data_dir}/{partial_name}")


def fp8_cast_bf16(path_src, path_dst):
    sentinel = Path(path_dst) / "model.safetensors.index.json"
    if sentinel.exists():
        print(f"fp8_cast_bf16 skip {path_dst} since {sentinel} exists")
        return

    base_backend.exec_command_gpu(
        f"python {repo_base_dir}/tools/fp8_cast_bf16.py "
        f"--input-fp8-hf-path {path_src} "
        f"--output-bf16-hf-path {path_dst} "
    )


def build_train_env_vars(request: ExecuteTrainRequest, backend_env_vars: dict[str, str]) -> dict[str, str]:
    config = request.config
    return {
        # exported for the submitting client too, but only the runtime env reaches the ray workers
        "PYTHONUNBUFFERED": "1",
        # If setting this in FSDP, the computation communication overlapping may have issues
        **(
            {}
            if request.train_backend_fsdp
            else {
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            }
        ),
        **backend_env_vars,
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
        **request.extra_env_vars,
        **_parse_extra_env_vars(config.extra_env_vars),
    }


def _parse_extra_env_vars(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return {kv[0]: kv[1] for item in text.split(" ") if item.strip() != "" if (kv := item.split("=")) or True}


def check_has_nvlink():
    output = base_backend.exec_command_gpu(
        "nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l", capture_output=True
    )
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
MOONCAKE_MASTER_METRICS_PORT = 50052
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
    base_backend.exec_command_cpu(
        "pkill -x mooncake_master >/dev/null 2>&1 || true; "
        f"(setsid mooncake_master --rpc_port {rpc_port} --metrics_port {metrics_port} "
        f"> {quoted_log_path} 2>&1 &)"
    )
    try:
        wait_for_server_ready(host, rpc_port, timeout=timeout)
    except RuntimeError as exc:
        base_backend.exec_command_cpu("pkill -x mooncake_master >/dev/null 2>&1 || true")
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
    "GB200": 4,
    "GB300": 4,
    "MI350X": 8,
    "MI355X": 8,
}

GENERATION_HARDWARE = {
    "H100": "Hopper",
    "GB200": "Blackwell",
    "GB300": "Blackwell",
}


_PLACEHOLDERS = ("{{node_rank}}", "{{nnodes}}", "{{master_addr}}", "{{node_ip}}")


def run_shell_command(cmd: str, capture_output: bool = False) -> str | None:
    print(f"EXEC: {cmd}", flush=True)

    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            shell=False,
            check=True,
            capture_output=capture_output,
            **(dict(text=True) if capture_output else {}),
        )
    except subprocess.CalledProcessError as e:
        if capture_output:
            print(f"{e.stdout=} {e.stderr=}")
        raise

    if capture_output:
        print(f"Captured stdout={result.stdout} stderr={result.stderr}")
        return result.stdout
    return None


def substitute_placeholders(cmd: str, *, node_rank: str, nnodes: str, master_addr: str, node_ip: str) -> str:
    values = {
        "{{node_rank}}": node_rank,
        "{{nnodes}}": nnodes,
        "{{master_addr}}": master_addr,
        "{{node_ip}}": node_ip,
    }
    for placeholder in _PLACEHOLDERS:
        cmd = cmd.replace(placeholder, values[placeholder])
    return cmd
