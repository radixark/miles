import json
import os
import subprocess
from pathlib import Path

repo_base_dir = Path(os.path.abspath(__file__)).resolve().parents[1]


def convert_checkpoint(model_name, model_type):
    exec_command(
        f"source {repo_base_dir}/scripts/models/{model_type}.sh && "
        "PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 tools/convert_hf_to_torch_dist.py "
        "${MODEL_ARGS[@]} "
        f"--hf-checkpoint /root/models/{model_name} "
        f"--save /root/{model_name}_torch_dist"
    )


def execute_train(
    train_args: str,
    num_gpus: int,
    model_type: str,
    master_addr: str = "127.0.0.1",
):
    exec_command(
        "pkill -9 sglang;"
        "sleep 3;"
        "ray stop --force;"
        "pkill -9 ray;"
        "pkill -9 python;"
        "sleep 3;"
        "pkill -9 ray;"
        "pkill -9 python;"
        "pkill -9 redis;"
    )

    exec_command(
        # will prevent ray from buffering stdout/stderr
        f"export PYTHONBUFFERED=16 && "
        f"ray start --head --node-ip-address {master_addr} --num-gpus {num_gpus} --disable-usage-stats"
    )

    runtime_env_json = json.dumps(
        {
            "env_vars": {
                "PYTHONPATH": "/root/Megatron-LM/",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": check_has_nvlink(),
                "no_proxy": f"127.0.0.1,{master_addr}",
            }
        }
    )

    exec_command(
        f"export PYTHONBUFFERED=16 && "
        f'source "{repo_base_dir}/scripts/models/{model_type}.sh" && '
        # TODO should this 127.0.0.1 be `master_addr` instead
        f'ray job submit --address="http://127.0.0.1:8265" '
        f'--runtime-env-json="{runtime_env_json}" '
        "-- python3 train.py "
        "${MODEL_ARGS[@]} "
        f"{train_args}"
    )


def check_has_nvlink():
    output = exec_command("nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l", capture_output=True)
    return int(output) > 0


def exec_command(cmd: str, capture_output: bool = False):
    print(f"EXEC: {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, check=True, capture_output=capture_output)
    if capture_output:
        return result.stdout
