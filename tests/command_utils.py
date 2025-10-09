import subprocess


def exec_command(cmd: str):
    print(f"EXEC: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def convert_checkpoint():
    exec_command(
        "source scripts/models/glm4-9B.sh && "
        "PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 tools/convert_hf_to_torch_dist.py "
        "${MODEL_ARGS[@]} "
        "--hf-checkpoint /root/models/GLM-Z1-9B-0414 "
        "--save /root/GLM-Z1-9B-0414_torch_dist"
    )
