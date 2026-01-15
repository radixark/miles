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


GENERATE_VARIANTS = [
    pytest.param("sglang_rollout", id="sglang_rollout"),
    pytest.param("modular_rollout", id="modular_rollout"),
]


def expected_sample(
    *,
    response: str = "\\boxed{8}",
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
        prompt="What is 1+7?",
        tokens=tokens if tokens is not None else [3838, 374, 220, 16, 10, 22, 30, 59, 79075, 90, 23, 92],
        multimodal_inputs=None,
        multimodal_train_inputs=None,
        response=response,
        response_length=response_length,
        label=None,
        reward=None,
        loss_mask=None,
        weight_versions=weight_versions or [],
        rollout_log_probs=rollout_log_probs if rollout_log_probs is not None else [-0.0, -0.0078125, -0.015625, -0.0234375, -0.03125],
        rollout_routed_experts=rollout_routed_experts,
        remove_sample=False,
        status=status,
        metadata={},
        train_metadata=None,
        non_generation_time=0.0,
        spec_info=Sample.SpecInfo(
            spec_accept_token_num=0, spec_draft_token_num=0, spec_verify_ct=0, completion_token_num=0
        ),
        prefix_cache_info=Sample.PrefixCacheInfo(cached_tokens=cached_tokens, total_prompt_tokens=prompt_tokens),
    )


def make_process_fn(
    response_text: str = "\\boxed{8}",
    finish_reason: str = "stop",
    cached_tokens: int = 0,
    weight_version: str | None = None,
    routed_experts: bytes | None = None,
):
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(
            text=response_text,
            finish_reason=finish_reason,
            cached_tokens=cached_tokens,
            weight_version=weight_version,
            routed_experts=routed_experts,
        )

    return process_fn


def make_args(
    *,
    router_port: int,
    use_rollout_routing_replay: bool = False,
    use_miles_router: bool = False,
    miles_router_middleware_paths: list[str] | None = None,
) -> Namespace:
    argv = [
        "pytest",
        "--train-backend", "fsdp",
        "--rollout-batch-size", "1",
        "--n-samples-per-prompt", "1",
        "--num-rollout", "1",
        "--rollout-num-gpus", "1",
        "--rollout-num-gpus-per-engine", "1",
        "--hf-checkpoint", "Qwen/Qwen3-0.6B",
        "--prompt-data", "/dev/null",
        "--input-key", "input",
        "--label-key", "label",
        "--rm-type", "math",
        "--sglang-router-ip", "127.0.0.1",
        "--sglang-router-port", str(router_port),
        "--rollout-max-response-len", "16",
    ]
    if use_rollout_routing_replay:
        argv.append("--use-rollout-routing-replay")

    from miles.utils.arguments import parse_args
    with patch("sys.argv", argv):
        args = parse_args()

    args.use_miles_router = use_miles_router
    args.miles_router_middleware_paths = miles_router_middleware_paths or []
    args.ci_test = False
    init_http_client(args)
    return args


def make_sample(
    prompt: str = "What is 1+7?",
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


async def call_generate(variant: str, args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    if variant == "sglang_rollout":
        from miles.rollout.sglang_rollout import generate
        return await generate(args, sample, sampling_params.copy())
    else:
        from miles.rollout.generate_hub.single_turn import generate
        state = GenerateState(args)
        input_obj = GenerateFnInput(
            state=state,
            sample=sample,
            sampling_params=sampling_params.copy(),
            evaluation=False,
        )
        output = await generate(input_obj)
        return output.samples


@dataclass
class GenerateEnv:
    args: Namespace
    mock_server: Any


@pytest.fixture
def generate_env(request):
    SingletonMeta.clear_instances(SingletonMeta)
    process_fn_kwargs = getattr(request, "param", {}).get("process_fn_kwargs", {})
    args_kwargs = getattr(request, "param", {}).get("args_kwargs", {})

    process_fn = make_process_fn(**process_fn_kwargs)

    with with_mock_server(
        model_name="Qwen/Qwen3-0.6B",
        process_fn=process_fn,
    ) as mock_server:
        args = make_args(router_port=mock_server.port, **args_kwargs)
        yield GenerateEnv(args=args, mock_server=mock_server)

    SingletonMeta.clear_instances(SingletonMeta)


class TestBasicGeneration:
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_basic_generation(self, variant, generate_env):
        sample = make_sample()
        sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

        result = run(call_generate(variant, generate_env.args, sample, sampling_params))

        assert result == expected_sample()

    @pytest.mark.parametrize("generate_env", [{"process_fn_kwargs": {"response_text": ""}}], indirect=True)
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_empty_response(self, variant, generate_env):
        sample = make_sample()
        sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

        result = run(call_generate(variant, generate_env.args, sample, sampling_params))

        assert result == expected_sample(
            response="",
            response_length=0,
            tokens=[3838, 374, 220, 16, 10, 22, 30],
            rollout_log_probs=[],
        )


class TestPromptProcessingPath:
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_tokenizer_path(self, variant, generate_env):
        sample = make_sample(prompt="What is 1+7?")
        sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

        result = run(call_generate(variant, generate_env.args, sample, sampling_params))

        assert len(generate_env.mock_server.request_log) == 1
        assert generate_env.mock_server.request_log[0] == {
            "input_ids": [3838, 374, 220, 16, 10, 22, 30],
            "sampling_params": {"max_new_tokens": 16, "temperature": 0.7},
            "return_logprob": True,
            "return_routed_experts": False,
        }


class TestMultiTurn:
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_first_turn_initializes_tokens(self, variant, generate_env):
        sample = make_sample(tokens=[])
        sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

        result = run(call_generate(variant, generate_env.args, sample, sampling_params))

        assert result == expected_sample()

    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_subsequent_turn_appends_tokens(self, variant, generate_env):
        existing_tokens = [1, 2, 3, 4, 5, 6, 7, 100, 101, 102]  # prompt + previous response
        sample = make_sample(
            tokens=existing_tokens,
            response="previous",
            response_length=3,
        )
        sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

        result = run(call_generate(variant, generate_env.args, sample, sampling_params))

        assert result == expected_sample(
            response="previous\\boxed{8}",
            response_length=3 + 5,
            tokens=existing_tokens + [59, 79075, 90, 23, 92],
        )

    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_multi_turn_max_tokens_adjusted(self, variant, generate_env):
        existing_tokens = [1, 2, 3, 4, 5, 6, 7, 100, 101, 102]
        sample = make_sample(
            tokens=existing_tokens,
            response="prev",
            response_length=3,
        )
        sampling_params = {"max_new_tokens": 10, "temperature": 0.7}

        run(call_generate(variant, generate_env.args, sample, sampling_params))

        assert generate_env.mock_server.request_log[0] == {
            "input_ids": existing_tokens,
            "sampling_params": {"max_new_tokens": 7, "temperature": 0.7},
            "return_logprob": True,
            "return_routed_experts": False,
        }


class TestBoundaryConditions:
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_max_new_tokens_zero_returns_truncated(self, variant, generate_env):
        existing_tokens = [1, 2, 3, 4, 5, 6, 7] + list(range(100, 110))
        sample = make_sample(
            tokens=existing_tokens,
            response="x" * 10,
            response_length=10,
        )
        sampling_params = {"max_new_tokens": 10, "temperature": 0.7}

        result = run(call_generate(variant, generate_env.args, sample, sampling_params))

        assert result.status == Sample.Status.TRUNCATED
        assert generate_env.mock_server.request_log == []


class TestFinishReason:
    @pytest.mark.parametrize("generate_env", [{"process_fn_kwargs": {"finish_reason": "stop"}}], indirect=True)
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_finish_stop_sets_completed(self, variant, generate_env):
        sample = make_sample()
        sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

        result = run(call_generate(variant, generate_env.args, sample, sampling_params))

        assert result == expected_sample(status=Sample.Status.COMPLETED)

    @pytest.mark.parametrize("generate_env", [{"process_fn_kwargs": {"finish_reason": "length"}}], indirect=True)
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_finish_length_sets_truncated(self, variant, generate_env):
        sample = make_sample()
        sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

        result = run(call_generate(variant, generate_env.args, sample, sampling_params))

        assert result == expected_sample(status=Sample.Status.TRUNCATED)

    @pytest.mark.parametrize("generate_env", [{"process_fn_kwargs": {"finish_reason": "abort"}}], indirect=True)
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_finish_abort_sets_aborted(self, variant, generate_env):
        sample = make_sample()
        sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

        result = run(call_generate(variant, generate_env.args, sample, sampling_params))

        assert result == expected_sample(status=Sample.Status.ABORTED)


class TestRoutedExperts:
    @pytest.mark.parametrize("generate_env", [{"args_kwargs": {"use_rollout_routing_replay": False}}], indirect=True)
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_routed_experts_disabled(self, variant, generate_env):
        sample = make_sample()
        sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

        result = run(call_generate(variant, generate_env.args, sample, sampling_params))

        assert result == expected_sample(rollout_routed_experts=None)
        assert generate_env.mock_server.request_log[0] == {
            "input_ids": [3838, 374, 220, 16, 10, 22, 30],
            "sampling_params": {"max_new_tokens": 16, "temperature": 0.7},
            "return_logprob": True,
            "return_routed_experts": False,
        }

    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_routed_experts_enabled_and_parsed(self, variant):
        SingletonMeta.clear_instances(SingletonMeta)
        num_layers = 2
        moe_router_topk = 4
        num_tokens = 7 + 5  # prompt + response
        routed_experts_array = np.arange(
            (num_tokens - 1) * num_layers * moe_router_topk, dtype=np.int32
        ).reshape(num_tokens - 1, num_layers, moe_router_topk)
        routed_experts_bytes = routed_experts_array.tobytes()

        process_fn = make_process_fn(routed_experts=routed_experts_bytes)
        with with_mock_server(model_name="Qwen/Qwen3-0.6B", process_fn=process_fn) as mock_server:
            args = make_args(router_port=mock_server.port, use_rollout_routing_replay=True)
            args.num_layers = num_layers
            args.moe_router_topk = moe_router_topk
            sample = make_sample()
            sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

            result = run(call_generate(variant, args, sample, sampling_params))

            assert result.rollout_routed_experts is not None
            assert result.rollout_routed_experts.shape == (num_tokens - 1, num_layers, moe_router_topk)
            np.testing.assert_array_equal(result.rollout_routed_experts, routed_experts_array)

        SingletonMeta.clear_instances(SingletonMeta)


class TestMetaInfo:
    @pytest.mark.parametrize("generate_env", [{"process_fn_kwargs": {"cached_tokens": 3}}], indirect=True)
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_prefix_cache_info_updated(self, variant, generate_env):
        sample = make_sample()
        sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

        result = run(call_generate(variant, generate_env.args, sample, sampling_params))

        assert result == expected_sample(cached_tokens=3, prompt_tokens=7)

    @pytest.mark.parametrize("generate_env", [{"process_fn_kwargs": {"weight_version": "v1.0"}}], indirect=True)
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_weight_version_collected(self, variant, generate_env):
        sample = make_sample()
        sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

        result = run(call_generate(variant, generate_env.args, sample, sampling_params))

        assert result == expected_sample(weight_versions=["v1.0"])


class TestPayloadStructure:
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_payload_has_required_fields(self, variant, generate_env):
        sample = make_sample()
        sampling_params = {"max_new_tokens": 16, "temperature": 0.7, "top_p": 0.9}

        run(call_generate(variant, generate_env.args, sample, sampling_params))

        assert generate_env.mock_server.request_log[0] == {
            "input_ids": [3838, 374, 220, 16, 10, 22, 30],
            "sampling_params": {"max_new_tokens": 16, "temperature": 0.7, "top_p": 0.9},
            "return_logprob": True,
            "return_routed_experts": False,
        }

    @pytest.mark.parametrize("generate_env", [{"args_kwargs": {"use_rollout_routing_replay": True}}], indirect=True)
    @pytest.mark.parametrize("variant", GENERATE_VARIANTS)
    def test_payload_routed_experts_flag_when_enabled(self, variant, generate_env):
        sample = make_sample()
        sampling_params = {"max_new_tokens": 16, "temperature": 0.7}

        run(call_generate(variant, generate_env.args, sample, sampling_params))

        assert generate_env.mock_server.request_log[0] == {
            "input_ids": [3838, 374, 220, 16, 10, 22, 30],
            "sampling_params": {"max_new_tokens": 16, "temperature": 0.7},
            "return_logprob": True,
            "return_routed_experts": True,
        }
