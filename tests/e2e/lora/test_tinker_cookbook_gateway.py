"""Official tinker-cookbook recipes against the gateway.

The cookbook's SFT (sl_loop) and RL (rl_loop) recipes are the executable
definition of the wire contract; our own client exercises only what we wrote.
The RL recipe pads its prompt region with dummy zero targets, so this test is
the in-repo tripwire for explicit-target handling.

Requires: 8 GPUs, Qwen3-4B, network access for the cookbook datasets
(openai/gsm8k, HuggingFaceH4/no_robots) and the pinned cookbook install.
"""

import subprocess
import time
import urllib.request

from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(
    est_time=2400,
    suite="stage-c-8-gpu-h200",
    labels=["lora", "weight-update"],
    hardware=["hopper"],
    disabled="pending first GPU validation of the cookbook acceptance path",
)

MODEL_NAME = "Qwen3-4B"
BASE_MODEL = f"Qwen/{MODEL_NAME}"
COOKBOOK_PIN = "git+https://github.com/thinking-machines-lab/tinker-cookbook@1f962eda3a2c"
GATEWAY_PORT = 10613
SERVE_TIMEOUT_S = 1200


def prepare():
    U.exec_command_cpu("mkdir -p /root/models")
    U.exec_command_cpu(f"hf download {BASE_MODEL} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command_cpu(f"pip install tinker==0.26.2 {COOKBOOK_PIN}")


def _wait_for_gateway(server: subprocess.Popen) -> None:
    deadline = time.time() + SERVE_TIMEOUT_S
    url = f"http://127.0.0.1:{GATEWAY_PORT}/api/v1/healthz"
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"gateway exited during startup with code {server.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError:
            time.sleep(5)
    raise TimeoutError(f"gateway not serving after {SERVE_TIMEOUT_S}s")


def execute():
    serve_cmd = (
        "python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve "
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        "--tp 1 --ep 1 --lora-rank 8 --lora-alpha 16 "
        f'--extra-args "--tinker-base-model {BASE_MODEL}"'
    )
    server = subprocess.Popen(["bash", "-c", serve_cmd])
    try:
        _wait_for_gateway(server)
        U.exec_command_cpu(
            "python examples/multi_lora/run_client_recipes.py "
            f"--base-url http://127.0.0.1:{GATEWAY_PORT} --base-model {BASE_MODEL} --mode both --steps 2"
        )
    finally:
        server.terminate()
        server.wait(timeout=120)


if __name__ == "__main__":
    prepare()
    execute()
