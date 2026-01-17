from argparse import Namespace
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import numpy as np
import pybase64
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.modular_rollout.orchestration_common import GenerateState
from miles.utils.async_utils import run
from miles.utils.http_utils import init_http_client
from miles.utils.misc import SingletonMeta
from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.test_utils.mock_sglang_server import ProcessResult, ProcessResultMetaInfo, with_mock_server
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
    image_data: list[str] | None = None,
) -> dict:
    result = {
        "input_ids": input_ids or PROMPT_TOKENS,
        "sampling_params": sampling_params or DEFAULT_SAMPLING_PARAMS,
        "return_logprob": True,
    }
    if variant == "modular_rollout" or return_routed_experts:
        result["return_routed_experts"] = return_routed_experts
    if image_data is not None:
        result["image_data"] = image_data
    return result


def expected_sample(
    *,
    prompt: str = PROMPT,
    response: str = RESPONSE_TEXT,
    response_length: int = 5,
    tokens: list[int] | None = None,
    rollout_log_probs: list[float] | None = None,
    status: Sample.Status = Sample.Status.COMPLETED,
    cached_tokens: int = 0,
    prompt_tokens: int = 7,
    weight_versions: list[str] | None = None,
    rollout_routed_experts: np.ndarray | None = None,
    spec_info: Sample.SpecInfo | None = None,
    multimodal_inputs: dict | None = None,
    multimodal_train_inputs: dict | None = None,
) -> Sample:
    return Sample(
        group_index=None,
        index=None,
        prompt=prompt,
        tokens=tokens if tokens is not None else PROMPT_TOKENS + RESPONSE_TOKENS,
        multimodal_inputs=multimodal_inputs,
        multimodal_train_inputs=multimodal_train_inputs,
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
        spec_info=spec_info or Sample.SpecInfo(),
        prefix_cache_info=Sample.PrefixCacheInfo(cached_tokens=cached_tokens, total_prompt_tokens=prompt_tokens),
    )


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


def make_sample(tokens=None, response="", response_length=0, status=Sample.Status.PENDING, multimodal_inputs=None):
    return Sample(
        prompt=PROMPT,
        tokens=tokens or [],
        response=response,
        response_length=response_length,
        status=status,
        multimodal_inputs=multimodal_inputs,
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
        assert result2.requests == [
            expected_request(
                variant,
                input_ids=tokens_after_turn1,
                sampling_params={"max_new_tokens": 14, "temperature": 0.7},
            )
        ]
        assert result2.sample == expected_sample(
            response=partial_text + remaining_text,
            response_length=2 + 3,
            tokens=tokens_after_turn1 + remaining_tokens,
            rollout_log_probs=partial_log_probs + remaining_log_probs,
            prompt_tokens=len(PROMPT_TOKENS) + len(tokens_after_turn1),
            status=Sample.Status.COMPLETED,
        )


class TestFinishReason:
    @pytest.mark.parametrize(
        "env,expected_status",
        [
            ({"process_fn_kwargs": {"finish_reason": "stop"}}, Sample.Status.COMPLETED),
            ({"process_fn_kwargs": {"finish_reason": "length"}}, Sample.Status.TRUNCATED),
            ({"process_fn_kwargs": {"finish_reason": "abort"}}, Sample.Status.ABORTED),
        ],
        indirect=["env"],
    )
    def test_finish_reason_sets_status(self, variant, env, expected_status):
        result = run_generate(variant, env)
        assert result.requests == [expected_request(variant)]
        assert result.sample == expected_sample(status=expected_status)


class TestRoutedExperts:
    @pytest.mark.parametrize(
        "env",
        [
            {
                "args_kwargs": {"use_rollout_routing_replay": True},
                "process_fn_kwargs": {"routed_experts": "placeholder"},
            }
        ],
        indirect=True,
    )
    def test_routed_experts_enabled_and_parsed(self, variant, env):
        num_layers, moe_router_topk = 2, 4
        num_tokens = len(PROMPT_TOKENS) + len(RESPONSE_TOKENS)
        routed_experts_array = np.arange((num_tokens - 1) * num_layers * moe_router_topk, dtype=np.int32).reshape(
            num_tokens - 1, num_layers, moe_router_topk
        )

        env.args.num_layers = num_layers
        env.args.moe_router_topk = moe_router_topk
        routed_experts_str = pybase64.b64encode(routed_experts_array.tobytes()).decode("ascii")
        env.mock_server.process_fn = lambda _: ProcessResult(
            text=RESPONSE_TEXT,
            finish_reason="stop",
            meta_info=ProcessResultMetaInfo(routed_experts=routed_experts_str),
        )

        result = run_generate(variant, env)
        assert result.requests == [expected_request(variant, return_routed_experts=True)]
        assert result.sample.rollout_routed_experts is not None
        assert result.sample.rollout_routed_experts.shape == (num_tokens - 1, num_layers, moe_router_topk)
        np.testing.assert_array_equal(result.sample.rollout_routed_experts, routed_experts_array)


class TestMetaInfo:
    @pytest.mark.parametrize(
        "env", [{"process_fn_kwargs": {"cached_tokens": 3, "weight_version": "v1.0"}}], indirect=True
    )
    def test_meta_info_fields_updated(self, variant, env):
        result = run_generate(variant, env)
        assert result.requests == [expected_request(variant)]
        assert result.sample == expected_sample(cached_tokens=3, weight_versions=["v1.0"])

    @pytest.mark.parametrize(
        "env",
        [
            {
                "args_kwargs": {"sglang_speculative_algorithm": "EAGLE"},
                "process_fn_kwargs": {"spec_accept_token_num": 10, "spec_draft_token_num": 15, "spec_verify_ct": 3},
            }
        ],
        indirect=True,
    )
    def test_spec_info_updated(self, variant, env):
        result = run_generate(variant, env)
        assert result.requests == [expected_request(variant)]
        assert result.sample == expected_sample(
            spec_info=Sample.SpecInfo(
                spec_accept_token_num=10, spec_draft_token_num=15, spec_verify_ct=3, completion_token_num=5
            )
        )


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


class TestPayloadStructure:
    def test_sampling_params_passed_through(self, variant, env):
        result = run_generate(variant, env, sampling_params={"max_new_tokens": 16, "temperature": 0.5, "top_p": 0.9})
        assert result.requests == [
            expected_request(variant, sampling_params={"max_new_tokens": 16, "temperature": 0.5, "top_p": 0.9})
        ]
        assert result.sample == expected_sample()


class TestBoundaryConditions:
    def test_max_new_tokens_zero_returns_truncated(self, variant, env):
        existing_tokens = [1, 2, 3, 4, 5, 6, 7] + list(range(100, 110))
        sample = make_sample(tokens=existing_tokens, response="x" * 10, response_length=10)

        result = run_generate(variant, env, sample, {"max_new_tokens": 10, "temperature": 0.7})
        assert result.requests == []
        assert result.sample.status == Sample.Status.TRUNCATED


class TestEmptyResponse:
    @pytest.mark.parametrize("env", [{"process_fn_kwargs": {"response_text": ""}}], indirect=True)
    def test_empty_response(self, variant, env):
        result = run_generate(variant, env)
        assert result.requests == [expected_request(variant)]
        assert result.sample == expected_sample(
            response="", response_length=0, tokens=PROMPT_TOKENS, rollout_log_probs=[]
        )


VLM_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


class TestMultimodal:
    @pytest.mark.parametrize("env", [{"args_kwargs": {"model_name": VLM_MODEL_NAME}}], indirect=True)
    def test_multimodal_inputs_processed(self, variant, env):
        test_image = Image.new("RGB", (64, 64), color="red")
        multimodal_inputs = {"images": [test_image]}
        processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME, trust_remote_code=True)
        expected_mti = {
            k: v
            for k, v in processor(text=PROMPT, **multimodal_inputs).items()
            if k not in ["input_ids", "attention_mask"]
        }

        result = run_generate(variant, env, make_sample(multimodal_inputs=multimodal_inputs))

        assert result.requests == [
            expected_request(
                variant,
                input_ids=PROMPT_TOKENS,
                image_data=[encode_image_for_rollout_engine(test_image)],
            )
        ]
        actual_mti = result.sample.multimodal_train_inputs
        assert actual_mti is not None
        assert set(actual_mti.keys()) == set(expected_mti.keys())
        assert torch.all(actual_mti["pixel_values"] == expected_mti["pixel_values"])
        assert torch.all(actual_mti["image_grid_thw"] == expected_mti["image_grid_thw"])
        assert result.sample == expected_sample(
            tokens=PROMPT_TOKENS + RESPONSE_TOKENS,
            multimodal_inputs=multimodal_inputs,
            multimodal_train_inputs=actual_mti,
        )
