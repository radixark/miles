from argparse import Namespace
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.modular_rollout.orchestration_common import GenerateState
from miles.utils.async_utils import run
from miles.utils.http_utils import init_http_client
from miles.utils.misc import SingletonMeta
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.types import Sample


# ------------------------------------ fixtures and consts ----------------------------------------


MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPT = "What is 1+7?"
PROMPT_TOKENS = [3838, 374, 220, 16, 10, 22, 30]
RESPONSE_TOKENS = [59, 79075, 90, 23, 92]
RESPONSE_TEXT = "\\boxed{8}"
RESPONSE_LOG_PROBS = [-0.0, -0.0078125, -0.015625, -0.0234375, -0.03125]
DEFAULT_SAMPLING_PARAMS = {"max_new_tokens": 16, "temperature": 0.7}


@pytest.fixture(params=["sglang_rollout", "modular_rollout"])
def variant(request):
    return request.param


def expected_request(
    variant: str,
    *,
    input_ids: list[int] | None = None,
    sampling_params: dict | None = None,
    return_routed_experts: bool = False,
) -> dict:
    result = {
        "input_ids": input_ids or PROMPT_TOKENS,
        "sampling_params": sampling_params or DEFAULT_SAMPLING_PARAMS,
        "return_logprob": True,
    }
    if variant == "modular_rollout" or return_routed_experts:
        result["return_routed_experts"] = return_routed_experts
    return result


def expected_sample(
    *,
    response: str = RESPONSE_TEXT,
    response_length: int = 5,
    tokens: list[int] | None = None,
    rollout_log_probs: list[float] | None = None,
    status: Sample.Status = Sample.Status.COMPLETED,
    cached_tokens: int = 0,
    prompt_tokens: int = 7,
    weight_versions: list[str] | None = None,
    rollout_routed_experts: np.ndarray | None = None,
) -> Sample:
    return Sample(
        group_index=None,
        index=None,
        prompt=PROMPT,
        tokens=tokens if tokens is not None else PROMPT_TOKENS + RESPONSE_TOKENS,
        multimodal_inputs=None,
        multimodal_train_inputs=None,
        response=response,
        response_length=response_length,
        label=None,
        reward=None,
        loss_mask=None,
        weight_versions=weight_versions or [],
        rollout_log_probs=rollout_log_probs if rollout_log_probs is not None else RESPONSE_LOG_PROBS,
        rollout_routed_experts=rollout_routed_experts,
        remove_sample=False,
        status=status,
        metadata={},
        train_metadata=None,
        non_generation_time=0.0,
        spec_info=Sample.SpecInfo(),
        prefix_cache_info=Sample.PrefixCacheInfo(cached_tokens=cached_tokens, total_prompt_tokens=prompt_tokens),
    )


def make_args(*, router_port: int, use_rollout_routing_replay: bool = False) -> Namespace:
    argv = [
        "pytest",
        "--train-backend",
        "fsdp",
        "--rollout-batch-size",
        "1",
        "--num-rollout",
        "1",
        "--hf-checkpoint",
        MODEL_NAME,
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

    from miles.utils.arguments import parse_args

    with patch("sys.argv", argv):
        args = parse_args()

    args.use_miles_router = False
    args.miles_router_middleware_paths = []
    args.ci_test = False
    init_http_client(args)
    return args


async def call_generate(variant: str, args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    if variant == "sglang_rollout":
        from miles.rollout.sglang_rollout import generate

        return await generate(args, sample, sampling_params.copy())
    elif variant == "modular_rollout":
        from miles.rollout.generate_hub.single_turn import generate

        state = GenerateState(args)
        output = await generate(
            GenerateFnInput(state=state, sample=sample, sampling_params=sampling_params.copy(), evaluation=False)
        )
        return output.samples
    else:
        raise NotImplementedError


@dataclass
class GenerateEnv:
    args: Namespace
    mock_server: Any


@dataclass
class GenerateResult:
    sample: Sample
    requests: list[dict]


@pytest.fixture
def env(request):
    SingletonMeta.clear_all_instances()
    params = getattr(request, "param", {})

    def process_fn(_):
        x = params.get("process_fn_kwargs", {})
        return ProcessResult(
            text=x.get("response_text", RESPONSE_TEXT),
            finish_reason=x.get("finish_reason", "stop"),
            cached_tokens=x.get("cached_tokens", 0),
            weight_version=x.get("weight_version"),
            routed_experts=x.get("routed_experts"),
        )

    with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as mock_server:
        args = make_args(router_port=mock_server.port, **params.get("args_kwargs", {}))
        yield GenerateEnv(args=args, mock_server=mock_server)

    SingletonMeta.clear_all_instances()


def make_sample(tokens=None, response="", response_length=0, status=Sample.Status.PENDING):
    return Sample(
        prompt=PROMPT, tokens=tokens or [], response=response, response_length=response_length, status=status
    )


def run_generate(variant: str, env: GenerateEnv, sample: Sample | None = None, sampling_params: dict | None = None):
    env.mock_server.request_log.clear()
    result_sample = run(
        call_generate(variant, env.args, sample or make_sample(), sampling_params or DEFAULT_SAMPLING_PARAMS)
    )
    return GenerateResult(sample=result_sample, requests=list(env.mock_server.request_log))


# ------------------------------------ tests ----------------------------------------


class TestBasicGeneration:
    def test_basic_generation(self, variant, env):
        result = run_generate(variant, env)
        assert result.requests == [expected_request(variant)]
        assert result.sample == expected_sample()

    @pytest.mark.parametrize("env", [{"process_fn_kwargs": {"response_text": ""}}], indirect=True)
    def test_empty_response(self, variant, env):
        result = run_generate(variant, env)
        assert result.requests == [expected_request(variant)]
        assert result.sample == expected_sample(
            response="", response_length=0, tokens=PROMPT_TOKENS, rollout_log_probs=[]
        )


class TestResumedSingleTurn:
    def test_two_consecutive_calls_on_same_sample(self, variant, env):
        partial_text = "\\boxed"
        partial_tokens = [59, 79075]
        partial_log_probs = [-0.0, -0.0078125]

        remaining_text = "{8}"
        remaining_tokens = [90, 23, 92]
        remaining_log_probs = [-0.0, -0.0078125, -0.015625]

        env.mock_server.process_fn = lambda _: ProcessResult(text=partial_text, finish_reason="abort")
        sample = make_sample()
        result1 = run_generate(variant, env, sample)
        assert result1.requests == [expected_request(variant)]
        assert result1.sample == expected_sample(
            response=partial_text,
            response_length=2,
            tokens=PROMPT_TOKENS + partial_tokens,
            rollout_log_probs=partial_log_probs,
            status=Sample.Status.ABORTED,
        )

        env.mock_server.process_fn = lambda _: ProcessResult(text=remaining_text, finish_reason="stop")
        result2 = run_generate(variant, env, result1.sample)
        tokens_after_turn1 = PROMPT_TOKENS + partial_tokens
        assert result2.requests == [expected_request(variant, input_ids=tokens_after_turn1)]
        assert result2.sample == expected_sample(
            response=partial_text + remaining_text,
            response_length=2 + 3,
            tokens=tokens_after_turn1 + remaining_tokens,
            rollout_log_probs=partial_log_probs + remaining_log_probs,
            prompt_tokens=len(tokens_after_turn1),
            status=Sample.Status.COMPLETED,
        )


class TestBoundaryConditions:
    def test_max_new_tokens_zero_returns_truncated(self, variant, env):
        existing_tokens = [1, 2, 3, 4, 5, 6, 7] + list(range(100, 110))
        sample = make_sample(tokens=existing_tokens, response="x" * 10, response_length=10)

        result = run_generate(variant, env, sample, {"max_new_tokens": 10, "temperature": 0.7})
        assert result.requests == []
        assert result.sample.status == Sample.Status.TRUNCATED


class TestFinishReason:
    @pytest.mark.parametrize("env", [{"process_fn_kwargs": {"finish_reason": "stop"}}], indirect=True)
    def test_finish_stop_sets_completed(self, variant, env):
        result = run_generate(variant, env)
        assert result.requests == [expected_request(variant)]
        assert result.sample == expected_sample(status=Sample.Status.COMPLETED)

    @pytest.mark.parametrize("env", [{"process_fn_kwargs": {"finish_reason": "length"}}], indirect=True)
    def test_finish_length_sets_truncated(self, variant, env):
        result = run_generate(variant, env)
        assert result.requests == [expected_request(variant)]
        assert result.sample == expected_sample(status=Sample.Status.TRUNCATED)

    @pytest.mark.parametrize("env", [{"process_fn_kwargs": {"finish_reason": "abort"}}], indirect=True)
    def test_finish_abort_sets_aborted(self, variant, env):
        result = run_generate(variant, env)
        assert result.requests == [expected_request(variant)]
        assert result.sample == expected_sample(status=Sample.Status.ABORTED)


class TestRoutedExperts:
    @pytest.mark.parametrize("env", [{"args_kwargs": {"use_rollout_routing_replay": False}}], indirect=True)
    def test_routed_experts_disabled(self, variant, env):
        result = run_generate(variant, env)
        assert result.requests == [expected_request(variant, return_routed_experts=False)]
        assert result.sample == expected_sample()

    @pytest.mark.parametrize(
        "env",
        [
            {
                "args_kwargs": {"use_rollout_routing_replay": True},
                "process_fn_kwargs": {"routed_experts": b"placeholder"},
            }
        ],
        indirect=True,
    )
    def test_routed_experts_enabled_and_parsed(self, variant, env, request):
        num_layers, moe_router_topk = 2, 4
        num_tokens = len(PROMPT_TOKENS) + len(RESPONSE_TOKENS)
        routed_experts_array = np.arange((num_tokens - 1) * num_layers * moe_router_topk, dtype=np.int32).reshape(
            num_tokens - 1, num_layers, moe_router_topk
        )

        env.args.num_layers = num_layers
        env.args.moe_router_topk = moe_router_topk
        env.mock_server.process_fn = lambda _: ProcessResult(
            text=RESPONSE_TEXT, finish_reason="stop", routed_experts=routed_experts_array.tobytes()
        )

        result = run_generate(variant, env)
        assert result.requests == [expected_request(variant, return_routed_experts=True)]
        assert result.sample.rollout_routed_experts is not None
        assert result.sample.rollout_routed_experts.shape == (num_tokens - 1, num_layers, moe_router_topk)
        np.testing.assert_array_equal(result.sample.rollout_routed_experts, routed_experts_array)


class TestMetaInfo:
    @pytest.mark.parametrize("env", [{"process_fn_kwargs": {"cached_tokens": 3}}], indirect=True)
    def test_prefix_cache_info_updated(self, variant, env):
        result = run_generate(variant, env)
        assert result.requests == [expected_request(variant)]
        assert result.sample == expected_sample(cached_tokens=3)

    @pytest.mark.parametrize("env", [{"process_fn_kwargs": {"weight_version": "v1.0"}}], indirect=True)
    def test_weight_version_collected(self, variant, env):
        result = run_generate(variant, env)
        assert result.requests == [expected_request(variant)]
        assert result.sample == expected_sample(weight_versions=["v1.0"])


class TestPayloadStructure:
    def test_payload_has_required_fields(self, variant, env):
        result = run_generate(variant, env, sampling_params={"max_new_tokens": 16, "temperature": 0.7, "top_p": 0.9})
        assert result.requests == [
            expected_request(variant, sampling_params={"max_new_tokens": 16, "temperature": 0.7, "top_p": 0.9})
        ]
        assert result.sample == expected_sample()

    @pytest.mark.parametrize("env", [{"args_kwargs": {"use_rollout_routing_replay": True}}], indirect=True)
    def test_payload_routed_experts_flag_when_enabled(self, variant, env):
        result = run_generate(variant, env)
        assert result.requests == [expected_request(variant, return_routed_experts=True)]
        assert result.sample == expected_sample()


class TestInputStatusValidation:
    @pytest.mark.parametrize("status", [Sample.Status.PENDING, Sample.Status.ABORTED])
    def test_allowed_statuses(self, variant, env, status):
        result = run_generate(variant, env, make_sample(status=status))
        assert result.requests == [expected_request(variant)]
        assert result.sample.status == Sample.Status.COMPLETED

    @pytest.mark.parametrize("status", [Sample.Status.COMPLETED, Sample.Status.TRUNCATED])
    def test_rejected_statuses(self, variant, env, status):
        with pytest.raises(AssertionError):
            run_generate(variant, env, make_sample(status=status))
