import numpy as np
import pybase64
import pytest
import torch
from PIL import Image
from tests.fast.fixtures.generation_fixtures import GenerateEnv, generation_env, listify, make_sample, run_generate
from transformers import AutoProcessor

from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.test_utils.mock_sglang_server import ProcessResult, ProcessResultMetaInfo
from miles.utils.types import Sample

_ = generation_env

# ------------------------------------ fixtures and consts ----------------------------------------


MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPT = "What is 1+7?"
PROMPT_TOKENS = [3838, 374, 220, 16, 10, 22, 30]
PROMPT_TOKEN_LEN = len(PROMPT_TOKENS)
RESPONSE_TOKENS = [59, 79075, 90, 23, 92]
RESPONSE_TEXT = "\\boxed{8}"
RESPONSE_LOG_PROBS = [-0.0, -0.0078125, -0.015625, -0.0234375, -0.03125]
SAMPLING_PARAMS = {"max_new_tokens": 16, "temperature": 0.7}
DEFAULT_MAX_NEW_TOKENS = SAMPLING_PARAMS["max_new_tokens"]


@pytest.fixture(params=["old_sglang_rollout", "single_turn", "multi_turn_single_sample", "multi_turn_multi_samples"])
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
        "sampling_params": sampling_params or SAMPLING_PARAMS,
        "return_logprob": True,
    }
    if variant in ("single_turn", "multi_turn_single_sample", "multi_turn_multi_samples") or return_routed_experts:
        result["return_routed_experts"] = return_routed_experts
    if image_data is not None:
        result["image_data"] = image_data
    return result


class _Unset:
    pass


_UNSET = _Unset()


def expected_sample(
    variant: str,
    *,
    prompt: str = PROMPT,
    response: str = RESPONSE_TEXT,
    response_length: int = 5,
    tokens: list[int] | None | _Unset = _UNSET,
    rollout_log_probs: list[float] | None | _Unset = _UNSET,
    status: Sample.Status = Sample.Status.COMPLETED,
    cached_tokens: int = 0,
    prompt_tokens: int = 7,
    weight_versions: list[str] | None = None,
    rollout_routed_experts: np.ndarray | None = None,
    spec_info: Sample.SpecInfo | None = None,
    multimodal_inputs: dict | None = None,
    multimodal_train_inputs: dict | None = None,
    loss_mask: list[int] | None | _Unset = _UNSET,
) -> Sample:
    actual_response_length = response_length if response_length is not None else len(RESPONSE_TOKENS)
    if isinstance(loss_mask, _Unset):
        loss_mask = (
            [1] * actual_response_length
            if variant in ("multi_turn_single_sample", "multi_turn_multi_samples")
            else None
        )

    return Sample(
        group_index=None,
        index=None,
        prompt=prompt,
        tokens=PROMPT_TOKENS + RESPONSE_TOKENS if isinstance(tokens, _Unset) else tokens,
        multimodal_inputs=multimodal_inputs,
        multimodal_train_inputs=multimodal_train_inputs,
        response=response,
        response_length=response_length,
        label=None,
        reward=None,
        loss_mask=loss_mask,
        weight_versions=weight_versions or [],
        rollout_log_probs=RESPONSE_LOG_PROBS if isinstance(rollout_log_probs, _Unset) else rollout_log_probs,
        rollout_routed_experts=rollout_routed_experts,
        remove_sample=False,
        status=status,
        metadata={},
        train_metadata=None,
        non_generation_time=0.0,
        spec_info=spec_info or Sample.SpecInfo(),
        prefix_cache_info=Sample.PrefixCacheInfo(cached_tokens=cached_tokens, total_prompt_tokens=prompt_tokens),
    )


def _make_sample(tokens=None, response="", response_length=0, status=Sample.Status.PENDING, multimodal_inputs=None):
    return make_sample(
        prompt=PROMPT,
        tokens=tokens,
        response=response,
        response_length=response_length,
        status=status,
        multimodal_inputs=multimodal_inputs,
    )


def _run_generate(variant: str, env: GenerateEnv, sample: Sample | None = None, sampling_params: dict | None = None):
    return run_generate(env, sample or _make_sample(), sampling_params or SAMPLING_PARAMS, variant=variant)


# ------------------------------------ tests ----------------------------------------


class TestBasicGeneration:
    def test_basic_generation(self, variant, generation_env):
        result = _run_generate(variant, generation_env)
        assert result.requests == [expected_request(variant)]
        assert listify(result.sample) == [expected_sample(variant)]


class TestResumedSingleTurn:
    def test_two_consecutive_calls_on_same_sample(self, variant, generation_env):
        if variant in ("multi_turn_single_sample", "multi_turn_multi_samples"):
            pytest.skip("not tested yet")
        partial_text = "\\boxed"
        partial_tokens = [59, 79075]
        partial_log_probs = [-0.0, -0.0078125]

        remaining_text = "{8}"
        remaining_tokens = [90, 23, 92]
        remaining_log_probs = [-0.0, -0.0078125, -0.015625]

        generation_env.mock_server.process_fn = lambda _: ProcessResult(text=partial_text, finish_reason="abort")
        sample = _make_sample()
        result1 = _run_generate(variant, generation_env, sample)
        assert result1.requests == [expected_request(variant)]
        assert result1.sample == expected_sample(
            variant,
            response=partial_text,
            response_length=2,
            tokens=PROMPT_TOKENS + partial_tokens,
            rollout_log_probs=partial_log_probs,
            status=Sample.Status.ABORTED,
        )

        generation_env.mock_server.process_fn = lambda _: ProcessResult(text=remaining_text, finish_reason="stop")
        result2 = _run_generate(variant, generation_env, result1.sample)
        tokens_after_turn1 = PROMPT_TOKENS + partial_tokens
        assert result2.requests == [
            expected_request(
                variant,
                input_ids=tokens_after_turn1,
                sampling_params={"max_new_tokens": 14, "temperature": 0.7},
            )
        ]
        assert result2.sample == expected_sample(
            variant,
            response=partial_text + remaining_text,
            response_length=2 + 3,
            tokens=tokens_after_turn1 + remaining_tokens,
            rollout_log_probs=partial_log_probs + remaining_log_probs,
            prompt_tokens=len(PROMPT_TOKENS) + len(tokens_after_turn1),
            status=Sample.Status.COMPLETED,
        )


class TestFinishReason:
    @pytest.mark.parametrize(
        "generation_env,expected_status",
        [
            ({"process_fn_kwargs": {"finish_reason": "stop"}}, Sample.Status.COMPLETED),
            ({"process_fn_kwargs": {"finish_reason": "length"}}, Sample.Status.TRUNCATED),
            ({"process_fn_kwargs": {"finish_reason": "abort"}}, Sample.Status.ABORTED),
        ],
        indirect=["generation_env"],
    )
    def test_finish_reason_sets_status(self, variant, generation_env, expected_status):
        result = _run_generate(variant, generation_env)
        assert result.requests == [expected_request(variant)]
        assert listify(result.sample) == [expected_sample(variant, status=expected_status)]


class TestRoutedExperts:
    @pytest.mark.parametrize(
        "generation_env",
        [
            {
                "args_kwargs": {"use_rollout_routing_replay": True},
                "process_fn_kwargs": {"routed_experts": "placeholder"},
            }
        ],
        indirect=True,
    )
    def test_routed_experts_enabled_and_parsed(self, variant, generation_env):
        num_layers, moe_router_topk = 2, 4
        num_tokens = len(PROMPT_TOKENS) + len(RESPONSE_TOKENS)
        routed_experts_array = np.arange((num_tokens - 1) * num_layers * moe_router_topk, dtype=np.int32).reshape(
            num_tokens - 1, num_layers, moe_router_topk
        )

        generation_env.args.num_layers = num_layers
        generation_env.args.moe_router_topk = moe_router_topk
        routed_experts_str = pybase64.b64encode(routed_experts_array.tobytes()).decode("ascii")
        generation_env.mock_server.process_fn = lambda _: ProcessResult(
            text=RESPONSE_TEXT,
            finish_reason="stop",
            meta_info=ProcessResultMetaInfo(routed_experts=routed_experts_str),
        )

        result = _run_generate(variant, generation_env)
        assert result.requests == [expected_request(variant, return_routed_experts=True)]
        sample = result.sample[0] if isinstance(result.sample, list) else result.sample
        assert sample.rollout_routed_experts is not None
        assert sample.rollout_routed_experts.shape == (num_tokens - 1, num_layers, moe_router_topk)
        np.testing.assert_array_equal(sample.rollout_routed_experts, routed_experts_array)


class TestMetaInfo:
    @pytest.mark.parametrize(
        "generation_env", [{"process_fn_kwargs": {"cached_tokens": 3, "weight_version": "v1.0"}}], indirect=True
    )
    def test_meta_info_fields_updated(self, variant, generation_env):
        result = _run_generate(variant, generation_env)
        assert result.requests == [expected_request(variant)]
        assert listify(result.sample) == [expected_sample(variant, cached_tokens=3, weight_versions=["v1.0"])]

    @pytest.mark.parametrize(
        "generation_env",
        [
            {
                "args_kwargs": {"sglang_speculative_algorithm": "EAGLE"},
                "process_fn_kwargs": {"spec_accept_token_num": 10, "spec_draft_token_num": 15, "spec_verify_ct": 3},
            }
        ],
        indirect=True,
    )
    def test_spec_info_updated(self, variant, generation_env):
        result = _run_generate(variant, generation_env)
        assert result.requests == [expected_request(variant)]
        assert listify(result.sample) == [
            expected_sample(
                variant,
                spec_info=Sample.SpecInfo(
                    spec_accept_token_num=10, spec_draft_token_num=15, spec_verify_ct=3, completion_token_num=5
                ),
            )
        ]


class TestInputStatusValidation:
    @pytest.mark.parametrize("status", [Sample.Status.PENDING, Sample.Status.ABORTED])
    def test_allowed_statuses(self, variant, generation_env, status):
        result = _run_generate(variant, generation_env, _make_sample(status=status))
        assert result.requests == [expected_request(variant)]
        assert listify(result.sample) == [expected_sample(variant)]

    @pytest.mark.parametrize("status", [Sample.Status.COMPLETED, Sample.Status.TRUNCATED])
    def test_rejected_statuses(self, variant, generation_env, status):
        if variant in ("multi_turn_single_sample", "multi_turn_multi_samples"):
            pytest.skip("not tested yet")
        with pytest.raises(AssertionError):
            _run_generate(variant, generation_env, _make_sample(status=status))


class TestPayloadStructure:
    def test_sampling_params_passed_through(self, variant, generation_env):
        result = _run_generate(
            variant, generation_env, sampling_params={"max_new_tokens": 16, "temperature": 0.5, "top_p": 0.9}
        )
        assert result.requests == [
            expected_request(variant, sampling_params={"max_new_tokens": 16, "temperature": 0.5, "top_p": 0.9})
        ]
        assert listify(result.sample) == [expected_sample(variant)]


class TestBoundaryConditions:
    def test_max_new_tokens_zero_returns_truncated(self, variant, generation_env):
        if variant in ("multi_turn_single_sample", "multi_turn_multi_samples"):
            pytest.skip("not tested yet")
        existing_tokens = [1, 2, 3, 4, 5, 6, 7] + list(range(100, 110))
        sample = _make_sample(tokens=existing_tokens, response="x" * 10, response_length=10)

        result = _run_generate(variant, generation_env, sample, {"max_new_tokens": 10, "temperature": 0.7})
        assert result.requests == []
        assert result.sample == expected_sample(
            variant,
            response="x" * 10,
            response_length=10,
            tokens=existing_tokens,
            rollout_log_probs=None,
            status=Sample.Status.TRUNCATED,
            prompt_tokens=0,
        )

    @pytest.mark.parametrize("generation_env", [{"args_kwargs": {"rollout_max_context_len": 5}}], indirect=True)
    def test_prompt_exceeds_max_context_len_returns_truncated(self, variant, generation_env):
        if variant == "old_sglang_rollout":
            pytest.skip("old_sglang_rollout does not support rollout_max_context_len")
        if variant == "multi_turn_multi_samples":
            pytest.skip("multi_turn_multi_samples returns empty list when first turn fails")
        result = _run_generate(variant, generation_env)
        assert result.requests == []
        tokens = PROMPT_TOKENS if variant in ("multi_turn_single_sample", "multi_turn_multi_samples") else []
        assert listify(result.sample) == [
            expected_sample(
                variant,
                response="",
                response_length=0,
                tokens=tokens,
                rollout_log_probs=None,
                status=Sample.Status.TRUNCATED,
                prompt_tokens=0,
                loss_mask=None if variant == "multi_turn_single_sample" else _UNSET,
            )
        ]

    @pytest.mark.parametrize(
        "generation_env,expected_max_new_tokens",
        [
            ({"args_kwargs": {"rollout_max_context_len": 10}}, 10 - PROMPT_TOKEN_LEN),
            ({"args_kwargs": {"rollout_max_context_len": 8}}, 8 - PROMPT_TOKEN_LEN),
            ({"args_kwargs": {"rollout_max_context_len": 100}}, DEFAULT_MAX_NEW_TOKENS),
        ],
        indirect=["generation_env"],
    )
    def test_moderate_length_input_adjusts_max_new_tokens(self, variant, generation_env, expected_max_new_tokens):
        if variant == "old_sglang_rollout":
            pytest.skip("old_sglang_rollout does not support rollout_max_context_len")
        result = _run_generate(variant, generation_env)
        assert len(result.requests) == 1
        assert result.requests[0]["sampling_params"]["max_new_tokens"] == expected_max_new_tokens
        assert result.requests[0]["sampling_params"]["temperature"] == SAMPLING_PARAMS["temperature"]
        assert listify(result.sample) == [expected_sample(variant)]

    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"rollout_max_context_len": PROMPT_TOKEN_LEN}}],
        indirect=True,
    )
    def test_adjusted_max_new_tokens_zero_returns_truncated(self, variant, generation_env):
        if variant == "old_sglang_rollout":
            pytest.skip("old_sglang_rollout does not support rollout_max_context_len")
        if variant == "multi_turn_multi_samples":
            pytest.skip("multi_turn_multi_samples returns empty list when first turn fails")
        result = _run_generate(variant, generation_env)
        assert result.requests == []
        tokens = PROMPT_TOKENS if variant == "multi_turn_single_sample" else []
        assert listify(result.sample) == [
            expected_sample(
                variant,
                response="",
                response_length=0,
                tokens=tokens,
                rollout_log_probs=None,
                status=Sample.Status.TRUNCATED,
                prompt_tokens=0,
                loss_mask=None if variant == "multi_turn_single_sample" else _UNSET,
            )
        ]


class TestEmptyResponse:
    @pytest.mark.parametrize("generation_env", [{"process_fn_kwargs": {"response_text": ""}}], indirect=True)
    def test_empty_response(self, variant, generation_env):
        result = _run_generate(variant, generation_env)
        assert result.requests == [expected_request(variant)]
        assert listify(result.sample) == [
            expected_sample(variant, response="", response_length=0, tokens=PROMPT_TOKENS, rollout_log_probs=[])
        ]


VLM_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


class TestMultimodal:
    @pytest.mark.parametrize("generation_env", [{"args_kwargs": {"model_name": VLM_MODEL_NAME}}], indirect=True)
    def test_multimodal_inputs_processed(self, variant, generation_env):
        if variant in ("multi_turn_single_sample", "multi_turn_multi_samples"):
            pytest.skip("not tested yet")
        test_image = Image.new("RGB", (64, 64), color="red")
        multimodal_inputs = {"images": [test_image]}
        processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME, trust_remote_code=True)
        expected_mti = {
            k: v
            for k, v in processor(text=PROMPT, **multimodal_inputs).items()
            if k not in ["input_ids", "attention_mask"]
        }

        result = _run_generate(variant, generation_env, _make_sample(multimodal_inputs=multimodal_inputs))

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
            variant,
            tokens=PROMPT_TOKENS + RESPONSE_TOKENS,
            multimodal_inputs=multimodal_inputs,
            multimodal_train_inputs=actual_mti,
        )
