import json
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.router.router import MilesRouter
from miles.utils.arguments import parse_args
from miles.utils.http_utils import find_available_port, init_http_client
from miles.utils.test_utils.mock_sglang_server import with_mock_server
from miles.utils.test_utils.thread_server import ThreadServer


def _build_args(*, train_path: str, eval_path: str, router_port: int) -> Namespace:
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
    with patch("sys.argv", argv):
        args = parse_args()
    args.miles_router_middleware_paths = []
    init_http_client(args)
    return args


@contextmanager
def _with_miles_router(args: Namespace) -> Iterator[ThreadServer]:
    router = MilesRouter(args, verbose=False)
    server = ThreadServer(router.app, host=args.sglang_router_ip, port=args.sglang_router_port)
    try:
        server.start()
        yield server
    finally:
        server.stop()


def _write_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def rollout_integration_env(tmp_path):
    train_path = str(tmp_path / "train.jsonl")
    eval_path = str(tmp_path / "eval.jsonl")
    _write_jsonl(train_path, [{"input": "What is 1+7?", "label": "8"}])
    _write_jsonl(eval_path, [{"input": "What is 1+5?", "label": "6"}])

    router_port = find_available_port(20000)
    args = _build_args(train_path=train_path, eval_path=eval_path, router_port=router_port)

    with with_mock_server(model_name=args.hf_checkpoint) as mock_server:
        with _with_miles_router(args) as router_server:
            r = requests.post(
                f"{router_server.url}/add_worker",
                params={"url": mock_server.url},
                timeout=5.0,
            )
            r.raise_for_status()

            data_source = RolloutDataSourceWithBuffer(args)
            yield args, data_source
