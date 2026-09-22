"""Shared gateway setup for Tinker GPU acceptance tests."""

import os
import signal
import subprocess
import time
import urllib.request
from contextlib import contextmanager, suppress

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-4B-Instruct-2507"
BASE_MODEL = f"Qwen/{MODEL_NAME}"
HF_CHECKPOINT = f"/root/models/{MODEL_NAME}"
GATEWAY_PORT = 10613
SERVE_TIMEOUT_S = 1200
TITO_MODEL = "qwen3"
# mounts the recorded-session routes with TITO for the session tests; the plain gateway tests pass nothing
SESSION_SERVER_ARGS = f"--tinker-session-server --tinker-tito-model {TITO_MODEL}"


def prepare_gateway():
    U.exec_command_cpu("mkdir -p /root/models")
    U.exec_command_cpu(f"hf download {BASE_MODEL} --local-dir /root/models/{MODEL_NAME}")


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
def running_gateway(extra_args: str = ""):
    """The gateway on GATEWAY_PORT; extra_args are more Tinker flags appended to the serve script's --extra-args."""
    serve_cmd = (
        "python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve "
        f"--hf-checkpoint {HF_CHECKPOINT} "
        "--model-type qwen3-4B-Instruct-2507 --tp 2 --ep 1 --lora-rank 8 --lora-alpha 16 "
        f'--extra-args "--tinker-base-model {BASE_MODEL} {extra_args}"'
    )
    server = subprocess.Popen(["bash", "-c", serve_cmd], start_new_session=True)
    try:
        _wait_for_gateway(server)
        yield f"http://127.0.0.1:{GATEWAY_PORT}"
    finally:
        try:
            with suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=30)
        finally:
            # Descendants can retain CI stdout after the launcher has exited.
            with suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=30)
            subprocess.run(["ray", "stop", "--force"], check=True, timeout=120)
