"""Run the official cookbook SFT and RL recipes against a real Tinker gateway."""

import importlib.metadata
import shlex
import sys
import tempfile
from pathlib import Path

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.tinker_gateway import BASE_MODEL, prepare_gateway, running_gateway

import miles.utils.external_utils.command_utils as U

register_cuda_ci(
    est_time=2400,
    suite="stage-c-8-gpu-h200",
    labels=["lora", "weight-update", "multi-lora"],
    hardware=["hopper"],
)

COOKBOOK_PIN = "tinker_cookbook[math-rl] @ git+https://github.com/thinking-machines-lab/tinker-cookbook@1f962eda3a2c"


def prepare(client_env):
    prepare_gateway()
    base_packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    U.exec_command_cpu(f"uv venv --python {shlex.quote(sys.executable)} {shlex.quote(str(client_env))}")
    U.exec_command_cpu(
        f"uv pip install --python {shlex.quote(str(client_env / 'bin/python'))} "
        f"--torch-backend cpu tinker==0.26.2 {shlex.quote(COOKBOOK_PIN)}"
    )
    U.exec_command_cpu(
        f"{shlex.quote(str(client_env / 'bin/python'))} " "-c 'from tinker_cookbook.recipes import sl_loop, rl_loop'"
    )
    assert {
        dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
    } == base_packages, "cookbook setup modified gateway Python packages"


def execute(client_python):
    with running_gateway() as base_url:
        U.exec_command_cpu(
            f"{shlex.quote(str(client_python))} examples/multi_lora/run_client_recipes.py "
            f"--base-url {base_url} --base-model {BASE_MODEL} --mode both --steps 2"
        )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="tinker-cookbook-") as temp_dir:
        client_env = Path(temp_dir) / "client"
        prepare(client_env)
        execute(client_env / "bin/python")
