from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    suite="stage-b-2-gpu-h200",
    labels=["sglang"],
)

import multiprocessing
import socket
import sys
import time

import requests
from fastapi import FastAPI
from sglang_router.launch_router import RouterArgs

from miles.utils.http_utils import run_router, terminate_process, wait_for_server_ready
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

_HOST = "127.0.0.1"
_MODEL_INFO: dict[str, object] = {
    "is_generation": True,
    "model_type": "mock-model-type",
    "architectures": ["MockForCausalLM"],
}


def _free_ports(count: int) -> list[int]:
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((_HOST, 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def _mock_worker_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/server_info")
    def server_info() -> dict[str, object]:
        return {
            "model_id": "mock-model",
            "served_model_name": "mock-model",
            "dp_size": 1,
        }

    @app.get("/model_info")
    def model_info() -> dict[str, object]:
        return _MODEL_INFO

    @app.get("/get_load")
    def get_load() -> list[dict[str, int]]:
        return [
            {"dp_rank": 0, "num_tokens": 7},
            {"dp_rank": 1, "num_tokens": 11},
        ]

    return app


def _wait_for_model_info(
    session: requests.Session,
    router_url: str,
    process: multiprocessing.Process,
    timeout: float = 10.0,
) -> requests.Response:
    deadline = time.monotonic() + timeout
    last_error = "router did not respond"

    while time.monotonic() < deadline:
        if not process.is_alive():
            raise RuntimeError(f"router process exited with code {process.exitcode}")

        try:
            response = session.get(f"{router_url}/model_info", timeout=1)
        except requests.RequestException as exc:
            last_error = str(exc)
        else:
            if response.status_code == 200:
                return response
            if response.status_code == 404:
                raise AssertionError("sgl-model-gateway does not expose GET /model_info")
            last_error = f"status={response.status_code}, body={response.text}"

        time.sleep(0.1)

    raise TimeoutError(f"worker registration did not complete: {last_error}")


def test_canonical_model_info_and_loads() -> None:
    worker_port, router_port, prometheus_port = _free_ports(3)
    worker = UvicornThreadServer(_mock_worker_app(), host=_HOST, port=worker_port)
    process = None

    worker.start()
    try:
        router_args = RouterArgs(
            host=_HOST,
            port=router_port,
            prometheus_host=_HOST,
            prometheus_port=prometheus_port,
            policy="round_robin",
            log_level="warn",
            worker_startup_timeout_secs=5,
            worker_startup_check_interval=1,
            health_check_timeout_secs=1,
            shutdown_grace_period_secs=1,
        )
        process = multiprocessing.get_context("spawn").Process(target=run_router, args=(router_args,))
        process.daemon = True
        process.start()
        wait_for_server_ready(_HOST, router_port, process=process, timeout=30)

        router_url = f"http://{_HOST}:{router_port}"
        with requests.Session() as session:
            session.trust_env = False
            add_worker = session.post(
                f"{router_url}/workers",
                json={"url": worker.url, "worker_type": "regular"},
                timeout=5,
            )
            assert add_worker.status_code == 202, add_worker.text

            model_info = _wait_for_model_info(session, router_url, process)
            assert model_info.json() == _MODEL_INFO

            loads = session.get(f"{router_url}/v1/loads", timeout=5)
            assert loads.status_code == 200, loads.text
            assert loads.json() == {
                "loads": [{"worker": worker.url, "load": 18}],
                "total_workers": 1,
                "successful": 1,
                "failed": 0,
            }
    finally:
        if process is not None:
            terminate_process(process)
        worker.stop()


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
