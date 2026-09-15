"""Shared gateway setup for Tinker GPU acceptance tests."""

import subprocess
import time
import urllib.request
from contextlib import contextmanager

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-4B-Instruct-2507"
BASE_MODEL = f"Qwen/{MODEL_NAME}"
GATEWAY_PORT = 10613
SERVE_TIMEOUT_S = 1200


def prepare_gateway():
    U.exec_command_cpu("mkdir -p /root/models")
    U.exec_command_cpu(f"hf download {BASE_MODEL} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command_cpu("pip install tinker==0.26.2")


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


@contextmanager
def running_gateway():
    serve_cmd = (
        "python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve "
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        "--model-type qwen3-4B-Instruct-2507 --tp 2 --ep 1 --lora-rank 8 --lora-alpha 16 "
        f'--extra-args "--tinker-base-model {BASE_MODEL}"'
    )
    server = subprocess.Popen(["bash", "-c", serve_cmd])
    try:
        _wait_for_gateway(server)
        yield f"http://127.0.0.1:{GATEWAY_PORT}"
    finally:
        server.terminate()
        server.wait(timeout=120)
