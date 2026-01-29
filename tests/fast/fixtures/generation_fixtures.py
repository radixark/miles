"""
Fixtures to test custom-generate-function
"""

from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import requests

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.inference_rollout.compatibility import load_generate_function
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.router.router import MilesRouter
from miles.utils.async_utils import run
from miles.utils.http_utils import find_available_port, init_http_client
from miles.utils.misc import SingletonMeta
from miles.utils.test_utils.mock_sglang_server import ProcessResult, ProcessResultMetaInfo, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

MODEL_NAME = "Qwen/Qwen3-0.6B"
RESPONSE_TEXT = "\\boxed{8}"
DEFAULT_SAMPLING_PARAMS = {"max_new_tokens": 64, "temperature": 0.7}

VARIANT_TO_GENERATE_FN_PATH = {
    "old_sglang_rollout": "miles.rollout.sglang_rollout.generate",
    "single_turn": "miles.rollout.generate_hub.single_turn.generate",
    "multi_turn_single_sample": "miles.rollout.generate_hub.multi_turn.generate",
    "multi_turn_multi_samples": "miles.rollout.generate_hub.multi_turn.generate",
    "agentic_tool_call_single_sample": "miles.rollout.generate_hub.agentic_tool_call.generate",
    "agentic_tool_call_multi_samples": "miles.rollout.generate_hub.agentic_tool_call.generate",
}


def extra_argv_for_variant(
    variant: str,
    *,
    custom_generate_function_path: str | None = None,
    generate_max_turns: int = 16,
    generate_tool_specs_path: str = "miles.utils.test_utils.mock_tools.SAMPLE_TOOLS",
    generate_tool_call_parser: str = "qwen25",
    generate_execute_tool_function_path: str = "miles.utils.test_utils.mock_tools.execute_tool_call",
) -> list[str]:
    argv = [
        "--custom-generate-function-path",
        custom_generate_function_path or VARIANT_TO_GENERATE_FN_PATH[variant],
    ]

    if variant in (
        "multi_turn_single_sample",
        "multi_turn_multi_samples",
        "agentic_tool_call_single_sample",
        "agentic_tool_call_multi_samples",
    ):
        argv += [
            "--generate-max-turns",
            str(generate_max_turns),
            "--generate-tool-specs-path",
            generate_tool_specs_path,
            "--generate-execute-tool-function-path",
            generate_execute_tool_function_path,
        ]
        if variant in ("multi_turn_single_sample", "multi_turn_multi_samples"):
            argv += ["--generate-tool-call-parser", generate_tool_call_parser]
        if variant in ("multi_turn_multi_samples", "agentic_tool_call_multi_samples"):
            argv.append("--generate-multi-samples")

    return argv


def listify(x):
    return x if isinstance(x, list) else [x]


def make_sample(
    *,
    prompt: str | list[dict] = "What is 1+7?",
    tokens: list[int] | None = None,
    response: str = "",
    response_length: int = 0,
    status: Sample.Status = Sample.Status.PENDING,
    multimodal_inputs: dict | None = None,
) -> Sample:
    return Sample(
        prompt=prompt,
        tokens=tokens or [],
        response=response,
        response_length=response_length,
        status=status,
        multimodal_inputs=multimodal_inputs,
    )


@dataclass
class GenerateEnv:
    args: Namespace
    mock_server: Any


@dataclass
class GenerateResult:
    sample: Sample | list[Sample]
    requests: list[dict]


def run_generate(
    env: GenerateEnv,
    sample: Sample,
    sampling_params: dict[str, Any] | None = None,
    *,
    variant: str = "single_turn",
) -> GenerateResult:
    env.mock_server.request_log.clear()
    result_sample = run(
        _call_generate(
            env.args,
            sample,
            sampling_params or DEFAULT_SAMPLING_PARAMS,
            variant=variant,
        )
    )
    return GenerateResult(sample=result_sample, requests=list(env.mock_server.request_log))


async def _call_generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    *,
    variant: str = "single_turn",
) -> Sample:
    generate_fn = load_generate_function(VARIANT_TO_GENERATE_FN_PATH[variant])
    state = GenerateState(args)
    input = GenerateFnInput(state=state, sample=sample, sampling_params=sampling_params.copy(), evaluation=False)
    output = await generate_fn(input)
    return output.samples


def make_args(
    *,
    variant: str,
    router_port: int,
    use_rollout_routing_replay: bool = False,
    sglang_speculative_algorithm: str | None = None,
    model_name: str = MODEL_NAME,
    extra_argv: list[str] | None = None,
    custom_generate_function_path: str | None = None,
    generate_max_turns: int = 16,
    generate_tool_specs_path: str = "miles.utils.test_utils.mock_tools.SAMPLE_TOOLS",
    generate_tool_call_parser: str = "qwen25",
    generate_execute_tool_function_path: str = "miles.utils.test_utils.mock_tools.execute_tool_call",
    rollout_max_context_len: int | None = None,
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
    if rollout_max_context_len is not None:
        argv.extend(["--rollout-max-context-len", str(rollout_max_context_len)])

    argv.extend(
        extra_argv_for_variant(
            variant,
            custom_generate_function_path=custom_generate_function_path,
            generate_max_turns=generate_max_turns,
            generate_tool_specs_path=generate_tool_specs_path,
            generate_tool_call_parser=generate_tool_call_parser,
            generate_execute_tool_function_path=generate_execute_tool_function_path,
        )
    )

    if extra_argv:
        argv.extend(extra_argv)

    from miles.utils.arguments import parse_args

    with patch("sys.argv", argv):
        args = parse_args()

    init_http_client(args)
    return args


@contextmanager
def with_miles_router(backend_url: str, model_name: str):
    router_args = SimpleNamespace(
        miles_router_max_connections=10,
        miles_router_timeout=30,
        miles_router_middleware_paths=[],
        rollout_health_check_interval=60,
        miles_router_health_check_failure_threshold=3,
        hf_checkpoint=model_name,
    )
    router = MilesRouter(router_args)

    port = find_available_port(31000)
    server = UvicornThreadServer(router.app, host="127.0.0.1", port=port)
    server.start()

    url = f"http://127.0.0.1:{port}"
    requests.post(f"{url}/add_worker", json={"url": backend_url})

    try:
        yield port
    finally:
        server.stop()


@pytest.fixture
def generation_env(request, variant):
    SingletonMeta.clear_all_instances()
    params = getattr(request, "param", {})
    args_kwargs = params.get("args_kwargs", {})
    model_name = args_kwargs.get("model_name", MODEL_NAME)
    custom_generate_function_path = VARIANT_TO_GENERATE_FN_PATH[variant]

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
        with with_miles_router(mock_server.url, model_name) as router_port:
            other_args_kwargs = {k: v for k, v in args_kwargs.items() if k != "model_name"}
            args = make_args(
                variant=variant,
                router_port=router_port,
                model_name=model_name,
                custom_generate_function_path=custom_generate_function_path,
                **other_args_kwargs,
            )
            yield GenerateEnv(args=args, mock_server=mock_server)

    SingletonMeta.clear_all_instances()
