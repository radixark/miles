"""
Fixtures to test custom-generate-function
"""

from argparse import Namespace
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from miles.utils.http_utils import init_http_client
from miles.utils.misc import SingletonMeta
from miles.utils.test_utils.mock_sglang_server import ProcessResult, ProcessResultMetaInfo, with_mock_server

MODEL_NAME = "Qwen/Qwen3-0.6B"
RESPONSE_TEXT = "\\boxed{8}"


@dataclass
class GenerateEnv:
    args: Namespace
    mock_server: Any


def make_args(
    *,
    router_port: int,
    use_rollout_routing_replay: bool = False,
    sglang_speculative_algorithm: str | None = None,
    model_name: str = MODEL_NAME,
) -> Namespace:
    argv = [
        "pytest",
        "--train-backend",
        "fsdp",
        "--rollout-batch-size",
        "1",
        "--num-rollout",
        "1",
        "--rollout-num-gpus",
        "1",
        "--rollout-num-gpus-per-engine",
        "1",
        "--hf-checkpoint",
        model_name,
        "--prompt-data",
        "/dev/null",
        "--rm-type",
        "math",
        "--sglang-router-ip",
        "127.0.0.1",
        "--sglang-router-port",
        str(router_port),
        "--rollout-max-response-len",
        "16",
    ]
    if use_rollout_routing_replay:
        argv.append("--use-rollout-routing-replay")
    if sglang_speculative_algorithm:
        argv.extend(["--sglang-speculative-algorithm", sglang_speculative_algorithm])

    from miles.utils.arguments import parse_args

    with patch("sys.argv", argv):
        args = parse_args()

    init_http_client(args)
    return args


@pytest.fixture
def generation_env(request):
    SingletonMeta.clear_all_instances()
    params = getattr(request, "param", {})
    args_kwargs = params.get("args_kwargs", {})
    model_name = args_kwargs.get("model_name", MODEL_NAME)

    def process_fn(_):
        x = params.get("process_fn_kwargs", {})
        return ProcessResult(
            text=x.get("response_text", RESPONSE_TEXT),
            finish_reason=x.get("finish_reason", "stop"),
            cached_tokens=x.get("cached_tokens", 0),
            meta_info=ProcessResultMetaInfo(
                weight_version=x.get("weight_version"),
                routed_experts=x.get("routed_experts"),
                spec_accept_token_num=x.get("spec_accept_token_num"),
                spec_draft_token_num=x.get("spec_draft_token_num"),
                spec_verify_ct=x.get("spec_verify_ct"),
            ),
        )

    with with_mock_server(model_name=model_name, process_fn=process_fn) as mock_server:
        other_args_kwargs = {k: v for k, v in args_kwargs.items() if k != "model_name"}
        args = make_args(router_port=mock_server.port, model_name=model_name, **other_args_kwargs)
        yield GenerateEnv(args=args, mock_server=mock_server)

    SingletonMeta.clear_all_instances()
