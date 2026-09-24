"""Shared gateway setup for Tinker GPU acceptance tests."""

import json
import os
import signal
import subprocess
import time
import urllib.request
from contextlib import contextmanager, suppress

import psutil

import miles.utils.external_utils.command_utils as U
from miles.utils.http_utils import is_port_available

MODEL_NAME = "Qwen3-4B-Instruct-2507"
BASE_MODEL = f"Qwen/{MODEL_NAME}"
GATEWAY_PORT = 10613
RAY_DASHBOARD_URL = "http://127.0.0.1:8265"
SERVE_TIMEOUT_S = 1200
STOP_TIMEOUT_S = 30
HTTP_TIMEOUT_S = 10
TERMINAL_JOB_STATUSES = {"STOPPED", "SUCCEEDED", "FAILED"}


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


def _dashboard(method: str, path: str):
    request = urllib.request.Request(
        f"{RAY_DASHBOARD_URL}{path}", data=b"" if method == "POST" else None, method=method
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
        return json.load(response)


def _stop_gateway_jobs() -> None:
    try:
        jobs = _dashboard("GET", "/api/jobs/")
    except OSError:
        return
    job_ids = [
        job["submission_id"]
        for job in jobs
        if job.get("submission_id")
        and "serve_tinker.py" in (job.get("entrypoint") or "")
        and job["status"] not in TERMINAL_JOB_STATUSES
    ]
    # Ray's JobSupervisor sends the driver tree SIGTERM, then SIGKILL; _kill_leaked_gateway asserts the port is free.
    for job_id in job_ids:
        _dashboard("POST", f"/api/jobs/{job_id}/stop")


def _gateway_listener_pids() -> list[int]:
    conns = psutil.net_connections(kind="tcp")
    return sorted({c.pid for c in conns if c.status == psutil.CONN_LISTEN and c.laddr.port == GATEWAY_PORT and c.pid})


def _wait_for_port_release(timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while not is_port_available(GATEWAY_PORT):
        if time.time() > deadline:
            return False
        time.sleep(1)
    return True


def _kill_leaked_gateway() -> None:
    if _wait_for_port_release(timeout_s=5):
        return
    # Last resort so a leaked gateway cannot answer the next test's health check; the leak still fails this one.
    pids = _gateway_listener_pids()
    for pid in pids:
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    released = _wait_for_port_release(STOP_TIMEOUT_S)
    raise RuntimeError(
        f"gateway still listening on :{GATEWAY_PORT} after teardown; SIGKILLed pids {pids}, port free now: {released}"
    )


def _stop_launcher(server: subprocess.Popen) -> None:
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


@contextmanager
def running_gateway():
    if not is_port_available(GATEWAY_PORT):
        raise RuntimeError(
            f"port {GATEWAY_PORT} already has a listener (pids {_gateway_listener_pids()}); "
            "refusing to reuse a gateway not started here"
        )
    serve_cmd = (
        "python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve "
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        "--model-type qwen3-4B-Instruct-2507 --tp 2 --ep 1 --lora-rank 8 --lora-alpha 16 "
        f'--extra-args "--tinker-base-model {BASE_MODEL}"'
    )
    server = subprocess.Popen(["bash", "-c", serve_cmd], start_new_session=True)
    try:
        _wait_for_gateway(server)
        yield f"http://127.0.0.1:{GATEWAY_PORT}"
    finally:
        try:
            # The gateway is the Ray job driver in its own session, which killpg and `ray stop` both miss;
            # stop the job while the dashboard, in the launcher's process group, is still up.
            _stop_gateway_jobs()
        finally:
            try:
                _stop_launcher(server)
            finally:
                _kill_leaked_gateway()
