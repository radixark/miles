import subprocess


def convert_checkpoint(model_name, model_type):
    exec_command(
        f"source scripts/models/{model_type}.sh && "
        "PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 tools/convert_hf_to_torch_dist.py "
        "${MODEL_ARGS[@]} "
        f"--hf-checkpoint /root/models/{model_name} "
        f"--save /root/{model_name}_torch_dist"
    )


def launch_train(num_gpus: int, master_addr: str = "127.0.0.1"):
    exec_command(f"ray start --head --node-ip-address {master_addr} --num-gpus {num_gpus} --disable-usage-stats")


def exec_command(cmd: str):
    print(f"EXEC: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)
