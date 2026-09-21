"""Run the official cookbook SFT and RL recipes against a real Tinker gateway."""

import shlex
import sys
import tempfile
from importlib.metadata import version

import torch
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


def prepare():
    prepare_gateway()
    packages = ("torch", "transformers", f"nvidia-cudnn-cu{torch.version.cuda.split('.')[0]}")
    expected = {name: version(name) for name in packages}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as overrides:
        overrides.write("\n".join(f"{name}=={value}" for name, value in expected.items()))
        overrides.flush()
        U.exec_command_cpu(
            f"uv pip install --python {shlex.quote(sys.executable)} --overrides {shlex.quote(overrides.name)} "
            f"tinker==0.26.2 {shlex.quote(COOKBOOK_PIN)}"
        )
    installed = {name: version(name) for name in packages}
    assert installed == expected, f"cookbook setup changed training dependencies: {expected=} {installed=}"
    U.exec_command_cpu(f"{shlex.quote(sys.executable)} -c 'from tinker_cookbook.recipes import sl_loop, rl_loop'")


def execute():
    with running_gateway() as base_url:
        U.exec_command_cpu(
            "python examples/multi_lora/run_client_recipes.py "
            f"--base-url {base_url} --base-model {BASE_MODEL} --mode both --steps 2"
        )


if __name__ == "__main__":
    prepare()
    execute()
