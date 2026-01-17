from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import groupby

import pytest
from tests.fixtures.generation_fixtures import GenerateEnv, generation_env, make_sample, run_generate
from transformers import AutoTokenizer

from miles.utils.test_utils.mock_sglang_server import ProcessResult
from miles.utils.test_utils.mock_tools import (
    MULTI_TURN_FIRST_RESPONSE,
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

MULTI_TURN_EXTRA_ARGV = [
    "--generate-max-turns",
    "4",
    "--generate-max-tool-calls",
    "4",
    "--generate-tool-specs-path",
    "miles.utils.test_utils.mock_tools.SAMPLE_TOOLS",
    "--generate-tool-call-parser",
    "qwen25",
    "--generate-execute-tool-function-path",
    "miles.utils.test_utils.mock_tools.execute_tool_call",
    "--rollout-max-context-len",
    "4096",
]


@pytest.fixture(params=["multi_turn_single_sample"])
def variant(request):
    return request.param


@dataclass(frozen=True)
class SampleParsedChunk:
    tokens_decoded_str: str
    loss_mask_value: int
    rollout_log_probs: list[float]


def parse_sample_into_chunks(sample: Sample, tokenizer) -> list[SampleParsedChunk]:
    prompt_len = len(sample.tokens) - sample.response_length
    response_tokens = sample.tokens[prompt_len:]
    loss_mask = sample.loss_mask
    log_probs = sample.rollout_log_probs

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


def verify_sample(
    actual: Sample,
    *,
    expected_chunks: list[SampleParsedChunk],
    expected_partial_sample: Sample,
):
    actual_chunks = parse_sample_into_chunks(actual, TOKENIZER)
    assert actual_chunks == expected_chunks

    actual_partial = replace(
        deepcopy(actual),
        tokens=[],
        loss_mask=[],
        rollout_log_probs=[],
        prefix_cache_info=Sample.PrefixCacheInfo(),
    )
    assert actual_partial == expected_partial_sample


def _run_generate(variant: str, env: GenerateEnv, sample: Sample, sampling_params: dict | None = None):
    return run_generate(env, sample, sampling_params, variant=variant)


SINGLE_TURN_PROMPT = [{"role": "user", "content": "What is 1+1?"}]
SINGLE_TURN_RESPONSE = "The answer is 2."

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
    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"extra_argv": MULTI_TURN_EXTRA_ARGV}}],
        indirect=True,
    )
    def test_single_turn_no_tool_call(self, variant, generation_env):
        generation_env.mock_server.process_fn = lambda _: ProcessResult(
            text=SINGLE_TURN_RESPONSE, finish_reason="stop"
        )

        result = _run_generate(variant, generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))

        assert len(result.requests) == 1
        verify_sample(
            result.sample,
            expected_chunks=[
                SampleParsedChunk(
                    tokens_decoded_str=SINGLE_TURN_RESPONSE,
                    loss_mask_value=1,
                    rollout_log_probs=[-1 / 128 * i for i in range(6)],
                ),
            ],
            expected_partial_sample=expected_partial_sample(
                prompt=SINGLE_TURN_PROMPT,
                response=SINGLE_TURN_RESPONSE,
                response_length=6,
            ),
        )

    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"extra_argv": MULTI_TURN_EXTRA_ARGV}}],
        indirect=True,
    )
    def test_two_turns_with_tool_call(self, variant, generation_env):
        generation_env.mock_server.process_fn = multi_turn_tool_call_process_fn

        result = _run_generate(variant, generation_env, make_sample(prompt=TWO_TURN_PROMPT))

        assert len(result.requests) == 2
        verify_sample(
            result.sample,
            expected_chunks=[
                SampleParsedChunk(
                    tokens_decoded_str=MULTI_TURN_FIRST_RESPONSE,
                    loss_mask_value=1,
                    rollout_log_probs=[-1 / 128 * i for i in range(45)],
                ),
                SampleParsedChunk(
                    tokens_decoded_str=TWO_TURN_TOOL_RESPONSE,
                    loss_mask_value=0,
                    rollout_log_probs=[0.0] * 31,
                ),
                SampleParsedChunk(
                    tokens_decoded_str=MULTI_TURN_SECOND_RESPONSE,
                    loss_mask_value=1,
                    rollout_log_probs=[-1 / 128 * i for i in range(24)],
                ),
            ],
            expected_partial_sample=expected_partial_sample(
                prompt=TWO_TURN_PROMPT,
                response=MULTI_TURN_FIRST_RESPONSE + TWO_TURN_TOOL_RESPONSE + MULTI_TURN_SECOND_RESPONSE,
                response_length=45 + 31 + 24,
            ),
        )
