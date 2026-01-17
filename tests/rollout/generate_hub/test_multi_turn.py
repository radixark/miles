from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import groupby

import pytest
from tests.fixtures.generation_fixtures import GenerateEnv, generation_env, listify, make_sample, run_generate
from transformers import AutoTokenizer

from miles.utils.test_utils.mock_sglang_server import ProcessResult
from miles.utils.test_utils.mock_tools import (
    MULTI_TURN_FIRST_PROMPT,
    MULTI_TURN_FIRST_RESPONSE,
    MULTI_TURN_SECOND_PROMPT,
    MULTI_TURN_SECOND_RESPONSE,
    SAMPLE_TOOLS,
    multi_turn_tool_call_process_fn,
)
from miles.utils.types import Sample

_ = generation_env, SAMPLE_TOOLS, multi_turn_tool_call_process_fn


# ------------------------------------ fixtures and consts ----------------------------------------


MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_SAMPLING_PARAMS = {"max_new_tokens": 64, "temperature": 0.7}
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
FIRST_PROMPT_TOKEN_IDS = TOKENIZER(MULTI_TURN_FIRST_PROMPT, add_special_tokens=False)["input_ids"]
SECOND_PROMPT_TOKEN_IDS = TOKENIZER(MULTI_TURN_SECOND_PROMPT, add_special_tokens=False)["input_ids"]


@pytest.fixture(params=["multi_turn_single_sample", "multi_turn_multi_samples"])
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


SINGLE_TURN_PROMPT = [{"role": "user", "content": "What is 1+1?"}]
SINGLE_TURN_RESPONSE = "The answer is 2."
_SINGLE_TURN_PROMPT_TEXT = TOKENIZER.apply_chat_template(
    SINGLE_TURN_PROMPT, tokenize=False, add_generation_prompt=True, tools=SAMPLE_TOOLS
)
SINGLE_TURN_PROMPT_TOKEN_IDS = TOKENIZER(_SINGLE_TURN_PROMPT_TEXT, add_special_tokens=False)["input_ids"]
SINGLE_TURN_PROMPT_TOKEN_LEN = len(SINGLE_TURN_PROMPT_TOKEN_IDS)

TWO_TURN_USER_QUESTION = "What is 42 + year + temperature?"
TWO_TURN_PROMPT = [{"role": "user", "content": TWO_TURN_USER_QUESTION}]
TWO_TURN_TOOL_RESPONSE = (
    "<|im_start|>user\n"
    "<tool_response>\n"
    '{"year": 2026}\n'
    "</tool_response>\n"
    "<tool_response>\n"
    '{"temperature": -60}\n'
    "</tool_response><|im_end|>\n"
    "<|im_start|>assistant\n"
)


# ------------------------------------ tests ----------------------------------------


class TestBasicMultiTurn:
    def test_single_turn_no_tool_call(self, variant, generation_env):
        generation_env.mock_server.process_fn = lambda _: ProcessResult(
            text=SINGLE_TURN_RESPONSE, finish_reason="stop"
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
                        prompt=SINGLE_TURN_PROMPT, response=SINGLE_TURN_RESPONSE, response_length=6
                    ),
                ),
            ],
        )

    def test_two_turns_with_tool_call(self, variant, generation_env):
        generation_env.mock_server.process_fn = multi_turn_tool_call_process_fn

        result = _run_generate(variant, generation_env, make_sample(prompt=TWO_TURN_PROMPT))

        assert result.requests == [
            expected_request(FIRST_PROMPT_TOKEN_IDS),
            expected_request(SECOND_PROMPT_TOKEN_IDS),
        ]
        if variant == "multi_turn_single_sample":
            expected = [
                ExpectedSampleInfo(
                    chunks=[
                        SampleParsedChunk(
                            tokens_decoded_str=MULTI_TURN_FIRST_RESPONSE,
                            loss_mask_value=1,
                            rollout_log_probs=[-1 / 128 * i for i in range(45)],
                        ),
                        SampleParsedChunk(
                            tokens_decoded_str=TWO_TURN_TOOL_RESPONSE, loss_mask_value=0, rollout_log_probs=[0.0] * 31
                        ),
                        SampleParsedChunk(
                            tokens_decoded_str=MULTI_TURN_SECOND_RESPONSE,
                            loss_mask_value=1,
                            rollout_log_probs=[-1 / 128 * i for i in range(24)],
                        ),
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=TWO_TURN_PROMPT,
                        response=MULTI_TURN_FIRST_RESPONSE + TWO_TURN_TOOL_RESPONSE + MULTI_TURN_SECOND_RESPONSE,
                        response_length=45 + 31 + 24,
                    ),
                ),
            ]
        else:
            expected = [
                ExpectedSampleInfo(
                    chunks=[
                        SampleParsedChunk(
                            tokens_decoded_str=MULTI_TURN_FIRST_RESPONSE,
                            loss_mask_value=1,
                            rollout_log_probs=[-1 / 128 * i for i in range(45)],
                        )
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=TWO_TURN_PROMPT,
                        response=MULTI_TURN_FIRST_RESPONSE,
                        response_length=45,
                    ),
                ),
                ExpectedSampleInfo(
                    chunks=[
                        SampleParsedChunk(
                            tokens_decoded_str=MULTI_TURN_SECOND_RESPONSE,
                            loss_mask_value=1,
                            rollout_log_probs=[-1 / 128 * i for i in range(24)],
                        )
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=TWO_TURN_PROMPT,
                        response=MULTI_TURN_SECOND_RESPONSE,
                        response_length=24,
                    ),
                ),
            ]
        verify_samples(result.sample, expected)


class TestExitConditions:
    def test_partial_rollout_not_supported(self, variant, generation_env):
        generation_env.args.partial_rollout = True

        with pytest.raises(AssertionError, match="Partial rollout is not supported"):
            _run_generate(variant, generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))

    def test_abort_preserves_content(self, variant, generation_env):
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
        generation_env.mock_server.process_fn = lambda _: ProcessResult(
            text=MULTI_TURN_FIRST_RESPONSE, finish_reason="length"
        )

        result = _run_generate(variant, generation_env, make_sample(prompt=TWO_TURN_PROMPT))

        assert result.requests == [expected_request(FIRST_PROMPT_TOKEN_IDS)]
        verify_samples(
            result.sample,
            [
                ExpectedSampleInfo(
                    chunks=[
                        SampleParsedChunk(
                            tokens_decoded_str=MULTI_TURN_FIRST_RESPONSE,
                            loss_mask_value=1,
                            rollout_log_probs=[-1 / 128 * i for i in range(45)],
                        )
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=TWO_TURN_PROMPT,
                        response=MULTI_TURN_FIRST_RESPONSE,
                        response_length=45,
                        status=Sample.Status.TRUNCATED,
                    ),
                ),
            ],
        )

    @pytest.mark.parametrize("generation_env", [{"args_kwargs": {"generate_max_turns": 1}}], indirect=True)
    def test_max_turns_reached(self, variant, generation_env):
        generation_env.mock_server.process_fn = lambda _: ProcessResult(
            text=MULTI_TURN_FIRST_RESPONSE, finish_reason="stop"
        )

        result = _run_generate(variant, generation_env, make_sample(prompt=TWO_TURN_PROMPT))

        assert result.requests == [expected_request(FIRST_PROMPT_TOKEN_IDS)]
        if variant == "multi_turn_single_sample":
            expected = [
                ExpectedSampleInfo(
                    chunks=[
                        SampleParsedChunk(
                            tokens_decoded_str=MULTI_TURN_FIRST_RESPONSE,
                            loss_mask_value=1,
                            rollout_log_probs=[-1 / 128 * i for i in range(45)],
                        ),
                        SampleParsedChunk(
                            tokens_decoded_str=TWO_TURN_TOOL_RESPONSE, loss_mask_value=0, rollout_log_probs=[0.0] * 31
                        ),
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=TWO_TURN_PROMPT,
                        response=MULTI_TURN_FIRST_RESPONSE + TWO_TURN_TOOL_RESPONSE,
                        response_length=45 + 31,
                    ),
                ),
            ]
        else:
            expected = [
                ExpectedSampleInfo(
                    chunks=[
                        SampleParsedChunk(
                            tokens_decoded_str=MULTI_TURN_FIRST_RESPONSE,
                            loss_mask_value=1,
                            rollout_log_probs=[-1 / 128 * i for i in range(45)],
                        )
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=TWO_TURN_PROMPT,
                        response=MULTI_TURN_FIRST_RESPONSE,
                        response_length=45,
                    ),
                ),
            ]
        verify_samples(result.sample, expected)


class TestRespectMaxContextLen:
    @pytest.mark.parametrize(
        "generation_env", [{"args_kwargs": {"rollout_max_context_len": SINGLE_TURN_PROMPT_TOKEN_LEN}}], indirect=True
    )
    def test_prompt_exceeds_max_context_len_returns_truncated(self, variant, generation_env):
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
        [{"args_kwargs": {"rollout_max_context_len": len(FIRST_PROMPT_TOKEN_IDS) + 45 + 31}}],
        indirect=True,
    )
    def test_second_turn_exceeds_max_context_len_returns_truncated(self, variant, generation_env):
        generation_env.mock_server.process_fn = multi_turn_tool_call_process_fn

        result = _run_generate(variant, generation_env, make_sample(prompt=TWO_TURN_PROMPT))

        assert result.requests == [expected_request(FIRST_PROMPT_TOKEN_IDS)]
        if variant == "multi_turn_single_sample":
            expected = [
                ExpectedSampleInfo(
                    chunks=[
                        SampleParsedChunk(
                            tokens_decoded_str=MULTI_TURN_FIRST_RESPONSE,
                            loss_mask_value=1,
                            rollout_log_probs=[-1 / 128 * i for i in range(45)],
                        ),
                        SampleParsedChunk(
                            tokens_decoded_str=TWO_TURN_TOOL_RESPONSE, loss_mask_value=0, rollout_log_probs=[0.0] * 31
                        ),
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=TWO_TURN_PROMPT,
                        response=MULTI_TURN_FIRST_RESPONSE + TWO_TURN_TOOL_RESPONSE,
                        response_length=45 + 31,
                        status=Sample.Status.TRUNCATED,
                    ),
                ),
            ]
        else:
            expected = [
                ExpectedSampleInfo(
                    chunks=[
                        SampleParsedChunk(
                            tokens_decoded_str=MULTI_TURN_FIRST_RESPONSE,
                            loss_mask_value=1,
                            rollout_log_probs=[-1 / 128 * i for i in range(45)],
                        )
                    ],
                    partial_sample=expected_partial_sample(
                        prompt=TWO_TURN_PROMPT,
                        response=MULTI_TURN_FIRST_RESPONSE,
                        response_length=45,
                        status=Sample.Status.TRUNCATED,
                    ),
                ),
            ]
        verify_samples(result.sample, expected)
