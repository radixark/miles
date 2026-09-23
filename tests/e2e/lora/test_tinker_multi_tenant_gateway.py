"""Concurrent tenants learn distinct markers and sample their own adapter snapshots."""

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.tinker_gateway import BASE_MODEL, prepare_gateway, running_gateway

import miles.utils.external_utils.command_utils as U

register_cuda_ci(
    est_time=2400,
    suite="stage-c-8-gpu-h200",
    labels=["multi-lora"],
    hardware=["hopper"],
)


def execute():
    with running_gateway() as base_url:
        U.exec_command_cpu(
            "python examples/multi_lora/run_multi_tenant_example.py "
            f"--base-url {base_url} --base-model {BASE_MODEL} "
            "--mode multi --clients 4 --lora-rank 8 --steps 12 --lr 1e-3 --max-tokens 24"
        )


if __name__ == "__main__":
    prepare_gateway()
    execute()
