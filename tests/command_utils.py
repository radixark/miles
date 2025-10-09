import json
import subprocess


def convert_checkpoint(model_name, model_type):
    exec_command(
        f"source scripts/models/{model_type}.sh && "
        "PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 tools/convert_hf_to_torch_dist.py "
        "${MODEL_ARGS[@]} "
        f"--hf-checkpoint /root/models/{model_name} "
        f"--save /root/{model_name}_torch_dist"
    )


def ray_start_and_submit(
    train_args: str,
    num_gpus: int,
    model_type: str,
    master_addr: str = "127.0.0.1",
):
    exec_command(f"ray start --head --node-ip-address {master_addr} --num-gpus {num_gpus} --disable-usage-stats")

    runtime_env_json = json.dumps({
        "env_vars": {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": has_nvlink,
            "no_proxy": f"127.0.0.1,{master_addr}",
        }
    })

    exec_command(
        # TODO should this 127.0.0.1 be `master_addr` instead
        f'source "{script_dir}/../scripts/models/{model_type}.sh" && '
        f'ray job submit --address="http://127.0.0.1:8265" '
        f'--runtime-env-json="{runtime_env_json}"'
        '-- python3 train.py '
        "${MODEL_ARGS[@]} "
        f'{train_args}'
    )


def exec_command(cmd: str):
    print(f"EXEC: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)
