import datetime
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from miles.utils.misc import exec_command

_ = exec_command

repo_base_dir = Path(os.path.abspath(__file__)).resolve().parents[1]


def convert_checkpoint(model_name, model_type, num_gpus: int, dir_dst="/root"):
    # TODO shall we make it in host-mapped folder and thus can cache it to speedup CI
    path_dst = f"{dir_dst}/{model_name}_torch_dist"
    if Path(path_dst).exists():
        print(f"convert_checkpoint skip {path_dst} since exists")
        return

    exec_command(
        f"source {repo_base_dir}/scripts/models/{model_type}.sh && "
        f"PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node {num_gpus} tools/convert_hf_to_torch_dist.py "
        "${MODEL_ARGS[@]} "
        f"--hf-checkpoint /root/models/{model_name} "
        f"--save {path_dst}"
    )


def hf_download_dataset(full_name: str):
    _, partial_name = full_name.split("/")
    exec_command(f"hf download --repo-type dataset {full_name} --local-dir /root/datasets/{partial_name}")


def execute_train(
    train_args: str,
    num_gpus: int,
    model_type: Optional[str],
    master_addr: str = "127.0.0.1",
    train_script: str = "train.py",
    before_ray_job_submit=None,
    extra_env_vars={},
):
    _cleanup_node()

    exec_command(
        # will prevent ray from buffering stdout/stderr
        f"export PYTHONBUFFERED=16 && "
        f"ray start --head --node-ip-address {master_addr} --num-gpus {num_gpus} --disable-usage-stats"
    )

    _start_ray_worker_nodes()

    if (f := before_ray_job_submit) is not None:
        f()

    runtime_env_json = json.dumps(
        {
            "env_vars": {
                "PYTHONPATH": "/root/Megatron-LM/",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": str(int(_check_has_nvlink())),
                "no_proxy": f"127.0.0.1,{master_addr}",
                **extra_env_vars,
            }
        }
    )

    source_cmd = f'source "{repo_base_dir}/scripts/models/{model_type}.sh" && ' if model_type is not None else ""
    model_args_str = "${MODEL_ARGS[@]}" if model_type is not None else ""

    exec_command(
        f"export PYTHONBUFFERED=16 && "
        f"{source_cmd}"
        # TODO should this 127.0.0.1 be `master_addr` instead
        f'ray job submit --address="http://127.0.0.1:8265" '
        f"--runtime-env-json='{runtime_env_json}' "
        f"-- python3 {train_script} "
        f"{model_args_str} "
        f"{train_args}"
    )


def _cleanup_node():
    exec_command(
        "pkill -9 sglang; "
        "sleep 3; "
        "ray stop --force; "
        "pkill -9 ray; "
        # cannot be run in CI, o/w kill the parent script
        # TODO: do we really need this kill? (or can we instead kill miles)
        # "pkill -9 python; "
        "pkill -9 miles; "
        "sleep 3; "
        "pkill -9 ray; "
        # "pkill -9 python; "
        "pkill -9 miles; "
        "pkill -9 redis; "
        "true; "
    )


# NOTE: this is just one naive implementation for environment without Slurm or Kubernetes.
#       we can generalize this later if it is needed (e.g. someone also does not have Slurm/Kubernetes).
def _start_ray_worker_nodes():
    worker_node_ips = os.environ.get("MILES_SCRIPT_START_RAY_WORKER_NODE_IPS", "").split(",")
    if not worker_node_ips:
        return

    def _execute_ssh(node_ip: str, command_inner: str):
        exec_command(f"ssh {TODO} 'cd /data/tom/primary_synced/tom_sglang_server/misc && {command_inner}'")

    def _execute_one(node_ip: str):
        _execute_ssh(node_ip, f"just miles-docker-run-without-exec")
        _execute_ssh(node_ip, f"just miles-start-ray-worker {head_node_ip}:6379")

    print(f"Start ray worker nodes: {worker_node_ips}", flush=True)
    with ThreadPoolExecutor(max_workers=100) as executor:
        list(executor.map(_execute_one, worker_node_ips))


def _check_has_nvlink():
    output = exec_command("nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l", capture_output=True)
    return int(output) > 0


def get_default_wandb_args(test_file: str, run_name_prefix: Optional[str] = None, run_id: Optional[str] = None):
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

    # do not put wandb_api_key value here to avoid leaking to logs explicitly
    return (
        "--use-wandb "
        f"--wandb-project miles-ci-{test_name} "
        f"--wandb-group {wandb_run_name} "
        f"--wandb-key ${{WANDB_API_KEY}} "
        "--disable-wandb-random-suffix "
    )


def create_run_id() -> str:
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S") + f"-{random.Random().randint(0, 999):03d}"


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
