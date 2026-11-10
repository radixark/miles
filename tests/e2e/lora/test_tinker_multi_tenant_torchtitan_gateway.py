"""Concurrent tenants on torchtitan LoRA slots learn distinct markers and sample their own adapter snapshots."""

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.tinker_gateway import QWEN3_8B_BASE_MODEL, prepare_gateway, qwen3_8b_serve_cmd, running_gateway

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=500,
    suite="stage-c-8-gpu-h100",
    labels=["lora", "multi-lora", "torchtitan"],
    hardware=["hopper"],
    num_gpus=8,
)


def execute():
    U = command_utils.default_config().create_backend()
    # one packed forward_backward routes tokens of several tenants through torchtitan slots
    with running_gateway(qwen3_8b_serve_cmd("torchtitan")) as base_url:
        U.exec_command_cpu(
            "python examples/multi_lora/run_multi_tenant_example.py "
            f"--base-url {base_url} --base-model {QWEN3_8B_BASE_MODEL} "
            "--mode multi --clients 4 --lora-rank 8 --steps 12 --lr 1e-3 --max-tokens 24"
        )


if __name__ == "__main__":
    prepare_gateway(QWEN3_8B_BASE_MODEL)
    execute()
