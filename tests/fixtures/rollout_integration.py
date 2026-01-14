import asyncio
import threading
import time
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import requests
import uvicorn

from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.router.router import MilesRouter
from miles.utils.arguments import parse_args
from miles.utils.http_utils import find_available_port, init_http_client
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server


class _UvicornThreadServer:
    def __init__(self, app, host: str, port: int):
        self._app = app
        self.host = host
        self.port = port
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        config = uvicorn.Config(self._app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config)

        def run() -> None:
            asyncio.run(self._server.serve())

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        self._wait_ready()

    def _wait_ready(self) -> None:
        for _ in range(50):
            try:
                r = requests.get(f"{self.url}/list_workers", timeout=0.5)
                if r.status_code in (200, 404):
                    return
            except Exception:
                pass
            time.sleep(0.1)
        raise RuntimeError(f"Failed to start server on {self.url}")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def _boxed_math_process_fn(prompt: str) -> ProcessResult:
    if "What is 1+7?" in prompt:
        return ProcessResult(text="\\boxed{8}", finish_reason="stop")
    if "What is 1+5?" in prompt:
        return ProcessResult(text="\\boxed{6}", finish_reason="stop")
    return ProcessResult(text="\\boxed{0}", finish_reason="stop")


def _build_args(*, monkeypatch: pytest.MonkeyPatch, train_path: str, eval_path: str, router_port: int) -> Namespace:
    argv = [
        "pytest",
        "--train-backend",
        "fsdp",
        "--rollout-batch-size",
        "1",
        "--n-samples-per-prompt",
        "1",
        "--num-rollout",
        "1",
        "--rollout-num-gpus",
        "1",
        "--rollout-num-gpus-per-engine",
        "1",
        "--hf-checkpoint",
        "Qwen/Qwen3-0.6B",
        "--prompt-data",
        train_path,
        "--input-key",
        "input",
        "--label-key",
        "label",
        "--rm-type",
        "math",
        "--eval-prompt-data",
        "toy",
        eval_path,
        "--use-miles-router",
        "--sglang-router-ip",
        "127.0.0.1",
        "--sglang-router-port",
        str(router_port),
        "--rollout-max-response-len",
        "16",
    ]
    monkeypatch.setattr("sys.argv", argv)
    args = parse_args()
    args.miles_router_middleware_paths = []
    init_http_client(args)
    return args


@contextmanager
def _with_miles_router(args: Namespace) -> Iterator[_UvicornThreadServer]:
    router = MilesRouter(args, verbose=False)
    server = _UvicornThreadServer(router.app, host=args.sglang_router_ip, port=args.sglang_router_port)
    try:
        server.start()
        yield server
    finally:
        server.stop()


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(__import__("json").dumps(row, ensure_ascii=False) + "\n")


@pytest.fixture
def rollout_integration_env(tmp_path, monkeypatch):
    train_path = str(tmp_path / "train.jsonl")
    eval_path = str(tmp_path / "eval.jsonl")
    _write_jsonl(train_path, [{"input": "What is 1+7?", "label": "8"}])
    _write_jsonl(eval_path, [{"input": "What is 1+5?", "label": "6"}])

    router_port = find_available_port(20000)
    args = _build_args(monkeypatch=monkeypatch, train_path=train_path, eval_path=eval_path, router_port=router_port)

    with with_mock_server(model_name=args.hf_checkpoint, process_fn=_boxed_math_process_fn) as mock_server:
        with _with_miles_router(args) as router_server:
            r = requests.post(
                f"{router_server.url}/add_worker",
                params={"url": mock_server.url},
                timeout=5.0,
            )
            r.raise_for_status()

            data_source = RolloutDataSourceWithBuffer(args)
            yield args, data_source
