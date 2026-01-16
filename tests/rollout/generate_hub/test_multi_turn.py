from dataclasses import dataclass
from itertools import groupby

import pytest
from transformers import AutoTokenizer

from miles.utils.test_utils.mock_sglang_server import ProcessResult
from miles.utils.test_utils.mock_tools import (
    MULTI_TURN_FIRST_PROMPT,
    SAMPLE_TOOLS,
    mock_execute_tool_function,
    multi_turn_tool_call_process_fn,
)
from miles.utils.types import Sample
from tests.fixtures.generation_fixtures import (
    GenerateEnv,
    GenerateResult,
    generation_env,
    make_sample,
    run_generate,
)

_ = generation_env, SAMPLE_TOOLS, mock_execute_tool_function, multi_turn_tool_call_process_fn


# ------------------------------------ fixtures and consts ----------------------------------------


MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_SAMPLING_PARAMS = {"max_new_tokens": 64, "temperature": 0.7}
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

MULTI_TURN_EXTRA_ARGV = [
    "--generate-max-turns", "4",
    "--generate-max-tool-calls", "4",
    "--generate-tool-specs-path", "miles.utils.test_utils.mock_tools:SAMPLE_TOOLS",
    "--generate-tool-call-parser", "qwen25",
    "--generate-execute-tool-function-path", "miles.utils.test_utils.mock_tools:mock_execute_tool_function",
]


@dataclass
class GenerateResult:
    sample: Sample
    requests: list[dict]


@dataclass(frozen=True)
class SampleParsedChunk:
    tokens_decoded_str: str
    loss_mask_value: int
    rollout_log_probs: tuple[float, ...]


def parse_sample_chunks(sample: Sample, tokenizer) -> list[SampleParsedChunk]:
    prompt_len = len(sample.tokens) - sample.response_length
    response_tokens = sample.tokens[prompt_len:]
    loss_mask = sample.loss_mask
    log_probs = sample.rollout_log_probs

    chunks = []
    idx = 0
    for mask_val, group in groupby(loss_mask):
        group_len = len(list(group))
        chunk_tokens = response_tokens[idx : idx + group_len]
        chunk_log_probs = log_probs[idx : idx + group_len]
        chunks.append(
            SampleParsedChunk(
                tokens_decoded_str=tokenizer.decode(chunk_tokens),
                loss_mask_value=mask_val,
                rollout_log_probs=tuple(chunk_log_probs),
            )
        )
        idx += group_len
    return chunks


def verify_sample(
    actual: Sample,
    *,
    expected_chunks: list[SampleParsedChunk],
    prompt: list[dict],
    status: Sample.Status = Sample.Status.COMPLETED,
):
    actual_chunks = parse_sample_chunks(actual, TOKENIZER)
    assert actual_chunks == expected_chunks

    expected_other_fields = Sample(
        group_index=None,
        index=None,
        prompt=prompt,
        tokens=actual.tokens,
        multimodal_inputs=None,
        multimodal_train_inputs=None,
        response=actual.response,
        response_length=actual.response_length,
        label=None,
        reward=None,
        loss_mask=actual.loss_mask,
        weight_versions=[],
        rollout_log_probs=actual.rollout_log_probs,
        rollout_routed_experts=None,
        remove_sample=False,
        status=status,
        metadata={},
        train_metadata=None,
        non_generation_time=0.0,
        spec_info=Sample.SpecInfo(),
        prefix_cache_info=Sample.PrefixCacheInfo(cached_tokens=0, total_prompt_tokens=0),
    )
    assert actual == expected_other_fields


MULTI_TURN_GENERATE_FN_PATH = "miles.rollout.generate_hub.multi_turn_single_sample:generate"


def make_sample(prompt=None):
    return Sample(
        prompt=prompt or [{"role": "user", "content": "What is 1+1?"}],
        tokens=[],
        response="",
        response_length=0,
        status=Sample.Status.PENDING,
    )


def run_generate(env: GenerateEnv, sample: Sample | None = None, sampling_params: dict | None = None):
    env.mock_server.request_log.clear()
    result_sample = run(
        call_generate(
            env.args,
            sample or make_sample(),
            sampling_params or DEFAULT_SAMPLING_PARAMS,
            generate_fn_path=MULTI_TURN_GENERATE_FN_PATH,
        )
    )
    return GenerateResult(sample=result_sample, requests=list(env.mock_server.request_log))


SINGLE_TURN_PROMPT = [{"role": "user", "content": "What is 1+1?"}]
SINGLE_TURN_RESPONSE = "The answer is 2."

TWO_TURN_PROMPT = [{"role": "user", "content": MULTI_TURN_FIRST_PROMPT}]
TWO_TURN_FIRST_RESPONSE = (
    "Let me get the year and temperature first.\n"
    "<tool_call>\n"
    '{"name": "get_year", "arguments": {}}\n'
    "</tool_call>\n"
    "<tool_call>\n"
    '{"name": "get_temperature", "arguments": {"location": "Mars"}}\n'
    "</tool_call>"
)
TWO_TURN_TOOL_RESPONSE = (
    '<|im_end|>\n<|im_start|>tool (tool_call_id: call00000)<|im_end|>\n{"year": 2026}'
    '<|im_start|>tool (tool_call_id: call00001)<|im_end|>\n{"temperature": -60}<|im_start|>assistant\n'
)
TWO_TURN_SECOND_RESPONSE = "The answer is: 42 + 2026 + -60 = 2008."


# ------------------------------------ tests ----------------------------------------


class TestBasicMultiTurn:
    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"extra_argv": MULTI_TURN_EXTRA_ARGV}}],
        indirect=True,
    )
    def test_single_turn_no_tool_call(self, generation_env):
        generation_env.mock_server.process_fn = lambda _: ProcessResult(text=SINGLE_TURN_RESPONSE, finish_reason="stop")

        result = run_generate(generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))

        assert len(result.requests) == 1
        verify_sample(
            result.sample,
            expected_chunks=[
                SampleParsedChunk(
                    tokens_decoded_str=SINGLE_TURN_RESPONSE,
                    loss_mask_value=1,
                    rollout_log_probs=tuple(-1 / 128 * i for i in range(6)),
                ),
            ],
            prompt=SINGLE_TURN_PROMPT,
        )

    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"extra_argv": MULTI_TURN_EXTRA_ARGV}}],
        indirect=True,
    )
    def test_two_turns_with_tool_call(self, generation_env):
        generation_env.mock_server.process_fn = multi_turn_tool_call_process_fn

        result = run_generate(generation_env, make_sample(prompt=TWO_TURN_PROMPT))

        assert len(result.requests) == 2
        verify_sample(
            result.sample,
            expected_chunks=[
                SampleParsedChunk(
                    tokens_decoded_str=TWO_TURN_FIRST_RESPONSE,
                    loss_mask_value=1,
                    rollout_log_probs=tuple(-1 / 128 * i for i in range(57)),
                ),
                SampleParsedChunk(
                    tokens_decoded_str=TWO_TURN_TOOL_RESPONSE,
                    loss_mask_value=0,
                    rollout_log_probs=tuple([0.0] * 47),
                ),
                SampleParsedChunk(
                    tokens_decoded_str=TWO_TURN_SECOND_RESPONSE,
                    loss_mask_value=1,
                    rollout_log_probs=tuple(-1 / 128 * i for i in range(25)),
                ),
            ],
            prompt=TWO_TURN_PROMPT,
        )
