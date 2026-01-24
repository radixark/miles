from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import groupby

import numpy as np
import pybase64
import pytest
from tests.fast.fixtures.generation_fixtures import GenerateEnv, generation_env, listify, make_sample, run_generate
from transformers import AutoTokenizer

from miles.utils.test_utils.mock_sglang_server import ProcessResult, ProcessResultMetaInfo
from miles.utils.test_utils.mock_tools import SAMPLE_TOOLS, ThreeTurnStub, TwoTurnStub
from miles.utils.types import Sample

_ = generation_env, SAMPLE_TOOLS, TwoTurnStub, ThreeTurnStub


def is_agentic_variant(variant: str) -> bool:
    return variant in ("agentic_tool_call_single_sample", "agentic_tool_call_multi_samples")


# ------------------------------------ fixtures and consts ----------------------------------------


MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_SAMPLING_PARAMS = {"max_new_tokens": 64, "temperature": 0.7}
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


@pytest.fixture(
    params=[
        "multi_turn_single_sample",
        "multi_turn_multi_samples",
        "agentic_tool_call_single_sample",
        "agentic_tool_call_multi_samples",
    ]
)
def variant(request):
    return request.param


@dataclass(frozen=True)
class SampleParsedChunk:
    tokens_decoded_str: str
    loss_mask_value: int
    rollout_log_probs: list[float]


@dataclass
class ExpectedSampleInfo:
    chunks: list[SampleParsedChunk]
    partial_sample: Sample


def token_len(text: str) -> int:
    return len(TOKENIZER(text, add_special_tokens=False)["input_ids"])


def expected_chunk(text: str, loss_mask: int) -> SampleParsedChunk:
    n = token_len(text)
    log_probs = [-1 / 128 * i for i in range(n)] if loss_mask else [0.0] * n
    return SampleParsedChunk(text, loss_mask, log_probs)


def parse_sample_into_chunks(sample: Sample, tokenizer) -> list[SampleParsedChunk]:
    prompt_len = len(sample.tokens) - sample.response_length
    response_tokens = sample.tokens[prompt_len:]
    loss_mask = sample.loss_mask or []
    log_probs = sample.rollout_log_probs or []

    chunks = []
    idx = 0
    for mask_val, group in groupby(loss_mask):
        group_len = len(list(group))
        sli = slice(idx, idx + group_len)
        chunks.append(
            SampleParsedChunk(
                tokens_decoded_str=tokenizer.decode(response_tokens[sli]),
                loss_mask_value=mask_val,
                rollout_log_probs=log_probs[sli],
            )
        )
        idx += group_len
    return chunks


def expected_partial_sample(
    *,
    prompt: list[dict],
    response: str,
    response_length: int,
    status: Sample.Status = Sample.Status.COMPLETED,
) -> Sample:
    return Sample(
        prompt=prompt,
        response=response,
        response_length=response_length,
        status=status,
        tokens=[],
        loss_mask=[],
        rollout_log_probs=[],
        weight_versions=[],
        spec_info=Sample.SpecInfo(),
        prefix_cache_info=Sample.PrefixCacheInfo(),
    )


def verify_samples(actual: Sample | list[Sample], expected: list[ExpectedSampleInfo]):
    actual = listify(actual)
    assert len(actual) == len(expected)

    for actual_item, expected_item in zip(actual, expected, strict=True):
        actual_chunks = parse_sample_into_chunks(actual_item, TOKENIZER)
        assert actual_chunks == expected_item.chunks

        actual_partial = replace(
            deepcopy(actual_item),
            tokens=[],
            loss_mask=[],
            rollout_log_probs=[],
            prefix_cache_info=Sample.PrefixCacheInfo(),
        )
        assert actual_partial == expected_item.partial_sample


def _run_generate(variant: str, env: GenerateEnv, sample: Sample, sampling_params: dict | None = None):
    return run_generate(env, sample, sampling_params, variant=variant)


def expected_request(input_ids: list[int], sampling_params: dict | None = None) -> dict:
    return {
        "input_ids": input_ids,
        "sampling_params": sampling_params or DEFAULT_SAMPLING_PARAMS,
        "return_logprob": True,
        "return_routed_experts": False,
    }


def expected_openai_request(messages: list[dict]) -> dict:
    return {"messages": messages, "model": "default", "tools": SAMPLE_TOOLS}


SINGLE_TURN_PROMPT = [{"role": "user", "content": "What is 1+1?"}]
SINGLE_TURN_RESPONSE = "The answer is 2."
_SINGLE_TURN_PROMPT_TEXT = TOKENIZER.apply_chat_template(
    SINGLE_TURN_PROMPT, tokenize=False, add_generation_prompt=True, tools=SAMPLE_TOOLS
)
SINGLE_TURN_PROMPT_TOKEN_IDS = TOKENIZER(_SINGLE_TURN_PROMPT_TEXT, add_special_tokens=False)["input_ids"]
SINGLE_TURN_PROMPT_TOKEN_LEN = len(SINGLE_TURN_PROMPT_TOKEN_IDS)


# ------------------------------------ tests ----------------------------------------


class TestBasicMultiTurn:
    def test_single_turn_no_tool_call(self, variant, generation_env):
        generation_env.mock_server.process_fn = lambda _: ProcessResult(
            text=SINGLE_TURN_RESPONSE, finish_reason="stop"
        )

        result = _run_generate(variant, generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))

        if is_agentic_variant(variant):
            assert result.requests == [expected_openai_request(SINGLE_TURN_PROMPT)]
        else:
            assert result.requests == [expected_request(SINGLE_TURN_PROMPT_TOKEN_IDS)]
        verify_samples(
            result.sample,
            [
                ExpectedSampleInfo(
                    chunks=[
                        SampleParsedChunk(
                            tokens_decoded_str=SINGLE_TURN_RESPONSE,
                            loss_mask_value=1,
                            rollout_log_probs=[-1 / 128 * i for i in range(6)],
                        ),
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=SINGLE_TURN_PROMPT, response=SINGLE_TURN_RESPONSE, response_length=6
                    ),
                ),
            ],
        )

    def test_two_turns_with_tool_call(self, variant, generation_env):
        generation_env.mock_server.process_fn = TwoTurnStub.process_fn

        S = TwoTurnStub
        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT))

        if is_agentic_variant(variant):
            assert result.requests == [
                expected_openai_request(S.OPENAI_MESSAGES_FIRST_TURN),
                expected_openai_request(S.OPENAI_MESSAGES_SECOND_TURN_FROM_CLIENT),
            ]
        else:
            assert result.requests == [
                expected_request(S.FIRST_PROMPT_TOKEN_IDS),
                expected_request(S.SECOND_PROMPT_TOKEN_IDS),
            ]
        if variant in ("multi_turn_single_sample", "agentic_tool_call_single_sample"):
            full_response = S.FIRST_RESPONSE + S.FIRST_TOOL_RESPONSE + S.SECOND_RESPONSE
            expected = [
                ExpectedSampleInfo(
                    chunks=[
                        expected_chunk(S.FIRST_RESPONSE, 1),
                        expected_chunk(S.FIRST_TOOL_RESPONSE, 0),
                        expected_chunk(S.SECOND_RESPONSE, 1),
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=S.PROMPT,
                        response=full_response,
                        response_length=token_len(full_response),
                    ),
                ),
            ]
        else:
            expected = [
                ExpectedSampleInfo(
                    chunks=[expected_chunk(S.FIRST_RESPONSE, 1)],
                    partial_sample=expected_partial_sample(
                        prompt=S.PROMPT,
                        response=S.FIRST_RESPONSE,
                        response_length=token_len(S.FIRST_RESPONSE),
                    ),
                ),
                ExpectedSampleInfo(
                    chunks=[expected_chunk(S.SECOND_RESPONSE, 1)],
                    partial_sample=expected_partial_sample(
                        prompt=S.PROMPT,
                        response=S.SECOND_RESPONSE,
                        response_length=token_len(S.SECOND_RESPONSE),
                    ),
                ),
            ]
        verify_samples(result.sample, expected)


class TestExitConditions:
    def test_partial_rollout_not_supported(self, variant, generation_env):
        if is_agentic_variant(variant):
            pytest.skip("agentic_tool_call does not check partial_rollout flag")
        generation_env.args.partial_rollout = True

        with pytest.raises(AssertionError, match="Partial rollout is not supported"):
            _run_generate(variant, generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))

    def test_abort_preserves_content(self, variant, generation_env):
        if is_agentic_variant(variant):
            pytest.skip("agentic_tool_call does not handle abort finish_reason")
        generation_env.mock_server.process_fn = lambda _: ProcessResult(
            text=SINGLE_TURN_RESPONSE, finish_reason="abort"
        )

        result = _run_generate(variant, generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))

        assert result.requests == [expected_request(SINGLE_TURN_PROMPT_TOKEN_IDS)]
        verify_samples(
            result.sample,
            [
                ExpectedSampleInfo(
                    chunks=[
                        SampleParsedChunk(
                            tokens_decoded_str=SINGLE_TURN_RESPONSE,
                            loss_mask_value=1,
                            rollout_log_probs=[-1 / 128 * i for i in range(6)],
                        ),
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=SINGLE_TURN_PROMPT,
                        response=SINGLE_TURN_RESPONSE,
                        response_length=6,
                        status=Sample.Status.ABORTED,
                    ),
                ),
            ],
        )

    def test_finish_reason_length_exits_and_preserves_content(self, variant, generation_env):
        S = TwoTurnStub
        generation_env.mock_server.process_fn = lambda _: ProcessResult(text=S.FIRST_RESPONSE, finish_reason="length")

        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT))

        if is_agentic_variant(variant):
            assert result.requests == [expected_openai_request(S.OPENAI_MESSAGES_FIRST_TURN)]
        else:
            assert result.requests == [expected_request(S.FIRST_PROMPT_TOKEN_IDS)]
        verify_samples(
            result.sample,
            [
                ExpectedSampleInfo(
                    chunks=[expected_chunk(S.FIRST_RESPONSE, 1)],
                    partial_sample=expected_partial_sample(
                        prompt=S.PROMPT,
                        response=S.FIRST_RESPONSE,
                        response_length=token_len(S.FIRST_RESPONSE),
                        status=Sample.Status.TRUNCATED,
                    ),
                ),
            ],
        )

    @pytest.mark.parametrize("generation_env", [{"args_kwargs": {"generate_max_turns": 1}}], indirect=True)
    def test_max_turns_reached(self, variant, generation_env):
        S = TwoTurnStub
        generation_env.mock_server.process_fn = lambda _: ProcessResult(text=S.FIRST_RESPONSE, finish_reason="stop")

        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT))

        if is_agentic_variant(variant):
            assert result.requests == [expected_openai_request(S.OPENAI_MESSAGES_FIRST_TURN)]
        else:
            assert result.requests == [expected_request(S.FIRST_PROMPT_TOKEN_IDS)]
        if variant == "multi_turn_single_sample":
            expected = [
                ExpectedSampleInfo(
                    chunks=[
                        expected_chunk(S.FIRST_RESPONSE, 1),
                        expected_chunk(S.FIRST_TOOL_RESPONSE, 0),
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=S.PROMPT,
                        response=S.FIRST_RESPONSE + S.FIRST_TOOL_RESPONSE,
                        response_length=token_len(S.FIRST_RESPONSE + S.FIRST_TOOL_RESPONSE),
                    ),
                ),
            ]
        else:
            expected = [
                ExpectedSampleInfo(
                    chunks=[expected_chunk(S.FIRST_RESPONSE, 1)],
                    partial_sample=expected_partial_sample(
                        prompt=S.PROMPT,
                        response=S.FIRST_RESPONSE,
                        response_length=token_len(S.FIRST_RESPONSE),
                    ),
                ),
            ]
        verify_samples(result.sample, expected)


class TestRespectMaxContextLen:
    @pytest.mark.parametrize(
        "generation_env", [{"args_kwargs": {"rollout_max_context_len": SINGLE_TURN_PROMPT_TOKEN_LEN}}], indirect=True
    )
    def test_prompt_exceeds_max_context_len_returns_truncated(self, variant, generation_env):
        if is_agentic_variant(variant):
            pytest.skip("TODO: implement")
        result = _run_generate(variant, generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))
        assert result.requests == []
        if variant == "multi_turn_single_sample":
            expected = [
                ExpectedSampleInfo(
                    chunks=[],
                    partial_sample=expected_partial_sample(
                        prompt=SINGLE_TURN_PROMPT, response="", response_length=0, status=Sample.Status.TRUNCATED
                    ),
                )
            ]
        else:
            expected = []
        verify_samples(result.sample, expected)

    @pytest.mark.parametrize(
        "generation_env",
        [
            {
                "args_kwargs": {
                    "rollout_max_context_len": len(TwoTurnStub.FIRST_PROMPT_TOKEN_IDS)
                    + token_len(TwoTurnStub.FIRST_RESPONSE)
                    + token_len(TwoTurnStub.FIRST_TOOL_RESPONSE)
                }
            }
        ],
        indirect=True,
    )
    def test_second_turn_exceeds_max_context_len_returns_truncated(self, variant, generation_env):
        if is_agentic_variant(variant):
            pytest.skip("TODO: implement")
        S = TwoTurnStub
        generation_env.mock_server.process_fn = S.process_fn

        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT))

        assert result.requests == [expected_request(S.FIRST_PROMPT_TOKEN_IDS)]
        if variant == "multi_turn_single_sample":
            partial_response = S.FIRST_RESPONSE + S.FIRST_TOOL_RESPONSE
            expected = [
                ExpectedSampleInfo(
                    chunks=[
                        expected_chunk(S.FIRST_RESPONSE, 1),
                        expected_chunk(S.FIRST_TOOL_RESPONSE, 0),
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=S.PROMPT,
                        response=partial_response,
                        response_length=token_len(partial_response),
                        status=Sample.Status.TRUNCATED,
                    ),
                ),
            ]
        else:
            expected = [
                ExpectedSampleInfo(
                    chunks=[expected_chunk(S.FIRST_RESPONSE, 1)],
                    partial_sample=expected_partial_sample(
                        prompt=S.PROMPT,
                        response=S.FIRST_RESPONSE,
                        response_length=token_len(S.FIRST_RESPONSE),
                        status=Sample.Status.TRUNCATED,
                    ),
                ),
            ]
        verify_samples(result.sample, expected)

    @pytest.mark.parametrize(
        "generation_env,expected_max_new_tokens",
        [
            (
                {"args_kwargs": {"rollout_max_context_len": len(TwoTurnStub.SECOND_PROMPT_TOKEN_IDS) + 10}},
                10,
            ),
            (
                {"args_kwargs": {"rollout_max_context_len": len(TwoTurnStub.SECOND_PROMPT_TOKEN_IDS) + 100}},
                64,
            ),
        ],
        indirect=["generation_env"],
    )
    def test_second_turn_adjusts_max_new_tokens(self, variant, generation_env, expected_max_new_tokens):
        if is_agentic_variant(variant):
            pytest.skip("TODO: implement")
        S = TwoTurnStub
        generation_env.mock_server.process_fn = S.process_fn

        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT))

        assert len(result.requests) >= 2
        assert result.requests[1]["sampling_params"]["max_new_tokens"] == expected_max_new_tokens
        assert result.requests[1]["sampling_params"]["temperature"] == DEFAULT_SAMPLING_PARAMS["temperature"]


class TestThreeTurn:
    """Need to test 3-turn case besides 2-turn, because e.g. merge_samples may behave differently."""

    def test_three_turns_with_sequential_tool_calls(self, variant, generation_env):
        generation_env.mock_server.process_fn = ThreeTurnStub.process_fn

        S = ThreeTurnStub
        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT))

        if is_agentic_variant(variant):
            assert result.requests == [
                expected_openai_request(S.OPENAI_MESSAGES_FIRST_TURN),
                expected_openai_request(S.OPENAI_MESSAGES_SECOND_TURN_FROM_CLIENT),
                expected_openai_request(S.OPENAI_MESSAGES_THIRD_TURN_FROM_CLIENT),
            ]
        else:
            assert result.requests == [
                expected_request(S.FIRST_PROMPT_TOKEN_IDS),
                expected_request(S.SECOND_PROMPT_TOKEN_IDS),
                expected_request(S.THIRD_PROMPT_TOKEN_IDS),
            ]
        if variant in ("multi_turn_single_sample", "agentic_tool_call_single_sample"):
            full_response = (
                S.FIRST_RESPONSE
                + S.FIRST_TOOL_RESPONSE
                + S.SECOND_RESPONSE
                + S.SECOND_TOOL_RESPONSE
                + S.THIRD_RESPONSE
            )
            expected = [
                ExpectedSampleInfo(
                    chunks=[
                        expected_chunk(S.FIRST_RESPONSE, 1),
                        expected_chunk(S.FIRST_TOOL_RESPONSE, 0),
                        expected_chunk(S.SECOND_RESPONSE, 1),
                        expected_chunk(S.SECOND_TOOL_RESPONSE, 0),
                        expected_chunk(S.THIRD_RESPONSE, 1),
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=S.PROMPT,
                        response=full_response,
                        response_length=token_len(full_response),
                    ),
                ),
            ]
        else:
            expected = [
                ExpectedSampleInfo(
                    chunks=[expected_chunk(S.FIRST_RESPONSE, 1)],
                    partial_sample=expected_partial_sample(
                        prompt=S.PROMPT,
                        response=S.FIRST_RESPONSE,
                        response_length=token_len(S.FIRST_RESPONSE),
                    ),
                ),
                ExpectedSampleInfo(
                    chunks=[expected_chunk(S.SECOND_RESPONSE, 1)],
                    partial_sample=expected_partial_sample(
                        prompt=S.PROMPT,
                        response=S.SECOND_RESPONSE,
                        response_length=token_len(S.SECOND_RESPONSE),
                    ),
                ),
                ExpectedSampleInfo(
                    chunks=[expected_chunk(S.THIRD_RESPONSE, 1)],
                    partial_sample=expected_partial_sample(
                        prompt=S.PROMPT,
                        response=S.THIRD_RESPONSE,
                        response_length=token_len(S.THIRD_RESPONSE),
                    ),
                ),
            ]
        verify_samples(result.sample, expected)


class TestRoutedExpertsMultiTurn:
    @pytest.mark.parametrize(
        "generation_env",
        [
            {
                "args_kwargs": {
                    "use_rollout_routing_replay": True,
                }
            }
        ],
        indirect=True,
    )
    def test_two_turns_routed_experts(self, variant, generation_env):
        if is_agentic_variant(variant):
            pytest.skip("TODO: implement")

        S = TwoTurnStub
        num_layers, moe_router_topk = 2, 4
        generation_env.args.num_layers = num_layers
        generation_env.args.moe_router_topk = moe_router_topk

        def make_routed_experts(prompt_token_ids, response_text):
            total_tokens = len(prompt_token_ids) + token_len(response_text)
            routed_experts_len = total_tokens - 1
            return np.arange(routed_experts_len * num_layers * moe_router_topk, dtype=np.int32).reshape(
                routed_experts_len, num_layers, moe_router_topk
            )

        first_routed_experts = make_routed_experts(S.FIRST_PROMPT_TOKEN_IDS, S.FIRST_RESPONSE)
        second_routed_experts = make_routed_experts(S.SECOND_PROMPT_TOKEN_IDS, S.SECOND_RESPONSE)

        def process_fn(prompt: str) -> ProcessResult:
            if prompt == S.FIRST_PROMPT:
                text, routed_experts = S.FIRST_RESPONSE, first_routed_experts
            elif prompt == S.SECOND_PROMPT:
                text, routed_experts = S.SECOND_RESPONSE, second_routed_experts
            else:
                raise ValueError(f"Unexpected prompt: {prompt}")
            return ProcessResult(
                text=text,
                finish_reason="stop",
                meta_info=ProcessResultMetaInfo(
                    routed_experts=pybase64.b64encode(routed_experts.tobytes()).decode("ascii")
                ),
            )

        generation_env.mock_server.process_fn = process_fn
        result = _run_generate(variant, generation_env, make_sample(prompt=S.PROMPT), DEFAULT_SAMPLING_PARAMS)

        sample = result.sample[-1] if isinstance(result.sample, list) else result.sample
        assert sample.rollout_routed_experts is not None
        assert sample.rollout_routed_experts.shape == second_routed_experts.shape
        np.testing.assert_array_equal(sample.rollout_routed_experts, second_routed_experts)
        assert len(sample.tokens) - 1 == second_routed_experts.shape[0]
