import json
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from miles.rollout.data_source import DataSource, RolloutDataSourceWithBuffer
from miles.router.router import MilesRouter
from miles.utils.arguments import parse_args
from miles.utils.http_utils import find_available_port, init_http_client
from miles.utils.misc import SingletonMeta
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


@dataclass(frozen=True)
class IntegrationEnvConfig:
    extra_argv: list[str] | None = None
    data_rows: list[dict] | None = None
    latency: float = 0.0


@dataclass(frozen=True)
class IntegrationEnv:
    args: Namespace
    data_source: DataSource
    mock_server: MockSGLangServer


def _build_args(*, data_path: str, router_port: int, extra_argv: list[str] | None = None) -> Namespace:
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
        data_path,
        "--input-key",
        "input",
        "--label-key",
        "label",
        "--rm-type",
        "math",
        "--eval-prompt-data",
        "toy",
        data_path,
        "--use-miles-router",
        "--sglang-router-ip",
        "127.0.0.1",
        "--sglang-router-port",
        str(router_port),
        "--rollout-max-response-len",
        "16",
    ] + (extra_argv or [])
    with patch("sys.argv", argv):
        args = parse_args()
    args.miles_router_middleware_paths = []
    init_http_client(args)
    return args


@contextmanager
def _with_miles_router(args: Namespace) -> Iterator[UvicornThreadServer]:
    router = MilesRouter(args, verbose=False)
    server = UvicornThreadServer(router.app, host=args.sglang_router_ip, port=args.sglang_router_port)
    try:
        server.start()
        yield server
    finally:
        server.stop()


def _write_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


DEFAULT_DATA_ROWS = [{"input": "What is 1+7?", "label": "8"}]


@pytest.fixture
def rollout_integration_env(tmp_path, request) -> IntegrationEnv:
    config = request.param
    assert isinstance(config, IntegrationEnvConfig)

    data_rows = config.data_rows or DEFAULT_DATA_ROWS

    data_path = str(tmp_path / "data.jsonl")
    _write_jsonl(data_path, data_rows)

    router_port = find_available_port(20000)
    args = _build_args(data_path=data_path, router_port=router_port, extra_argv=config.extra_argv)

    SingletonMeta.clear_all_instances()

    with with_mock_server(model_name=args.hf_checkpoint, latency=config.latency) as mock_server:
        with _with_miles_router(args) as router_server:
            r = requests.post(
                f"{router_server.url}/add_worker",
                params={"url": mock_server.url},
                timeout=5.0,
            )
            r.raise_for_status()

            data_source = RolloutDataSourceWithBuffer(args)
            yield IntegrationEnv(args=args, data_source=data_source, mock_server=mock_server)

    SingletonMeta.clear_all_instances()
