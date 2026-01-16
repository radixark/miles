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
_SINGLE_TURN_PROMPT_TEXT = TOKENIZER.apply_chat_template(
    SINGLE_TURN_PROMPT, tokenize=False, add_generation_prompt=True, tools=SAMPLE_TOOLS
)
SINGLE_TURN_PROMPT_TOKEN_LEN = len(TOKENIZER(_SINGLE_TURN_PROMPT_TEXT, add_special_tokens=False)["input_ids"])

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


def _make_extra_argv(**overrides) -> list[str]:
    base = {
        "generate-max-turns": "4",
        "generate-max-tool-calls": "4",
        "generate-tool-specs-path": "miles.utils.test_utils.mock_tools.SAMPLE_TOOLS",
        "generate-tool-call-parser": "qwen25",
        "generate-execute-tool-function-path": "miles.utils.test_utils.mock_tools.execute_tool_call",
        "rollout-max-context-len": "4096",
    }
    base.update(overrides)
    result = []
    for k, v in base.items():
        if v is not None:
            result.extend([f"--{k}", str(v)])
    return result


class TestExitConditions:
    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"extra_argv": MULTI_TURN_EXTRA_ARGV}}],
        indirect=True,
    )
    def test_partial_rollout_not_supported(self, variant, generation_env):
        generation_env.args.partial_rollout = True

        with pytest.raises(AssertionError, match="Partial rollout is not supported"):
            _run_generate(variant, generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))

    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"extra_argv": MULTI_TURN_EXTRA_ARGV}}],
        indirect=True,
    )
    def test_abort_returns_immediately(self, variant, generation_env):
        generation_env.mock_server.process_fn = lambda _: ProcessResult(
            text=SINGLE_TURN_RESPONSE, finish_reason="abort"
        )

        result = _run_generate(variant, generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))

        assert len(result.requests) == 1
        assert result.sample.status == Sample.Status.ABORTED

    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"extra_argv": MULTI_TURN_EXTRA_ARGV}}],
        indirect=True,
    )
    def test_finish_reason_length_exits_and_preserves_content(self, variant, generation_env):
        generation_env.mock_server.process_fn = lambda _: ProcessResult(
            text=MULTI_TURN_FIRST_RESPONSE, finish_reason="length"
        )

        result = _run_generate(variant, generation_env, make_sample(prompt=TWO_TURN_PROMPT))

        assert len(result.requests) == 1
        assert result.sample.status == Sample.Status.TRUNCATED
        verify_sample(
            result.sample,
            expected_chunks=[
                SampleParsedChunk(
                    tokens_decoded_str=MULTI_TURN_FIRST_RESPONSE,
                    loss_mask_value=1,
                    rollout_log_probs=[-1 / 128 * i for i in range(45)],
                ),
            ],
            expected_partial_sample=expected_partial_sample(
                prompt=TWO_TURN_PROMPT,
                response=MULTI_TURN_FIRST_RESPONSE,
                response_length=45,
                status=Sample.Status.TRUNCATED,
            ),
        )

    # TODO: This test exposes a bug in multi_turn_single_sample.py where `output` is undefined
    # when the loop breaks on the first iteration due to context length exceeded.
    # @pytest.mark.parametrize(
    #     "generation_env",
    #     [{"args_kwargs": {"extra_argv": _make_extra_argv(**{"rollout-max-context-len": "10"})}}],
    #     indirect=True,
    # )
    # def test_context_length_exceeded_truncates(self, variant, generation_env):
    #     result = _run_generate(variant, generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))
    #
    #     assert len(result.requests) == 0
    #     assert result.sample.status == Sample.Status.TRUNCATED

    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"extra_argv": _make_extra_argv(**{"generate-max-turns": "1"})}}],
        indirect=True,
    )
    def test_max_turns_reached(self, variant, generation_env):
        call_count = [0]

        def always_tool_call_process_fn(_):
            call_count[0] += 1
            return ProcessResult(text=MULTI_TURN_FIRST_RESPONSE, finish_reason="stop")

        generation_env.mock_server.process_fn = always_tool_call_process_fn

        result = _run_generate(variant, generation_env, make_sample(prompt=TWO_TURN_PROMPT))

        assert call_count[0] == 1
        assert len(result.requests) == 1

    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"extra_argv": _make_extra_argv(**{"generate-max-tool-calls": "0"})}}],
        indirect=True,
    )
    def test_max_tool_calls_reached(self, variant, generation_env):
        generation_env.mock_server.process_fn = lambda _: ProcessResult(
            text=MULTI_TURN_FIRST_RESPONSE, finish_reason="stop"
        )

        result = _run_generate(variant, generation_env, make_sample(prompt=TWO_TURN_PROMPT))

        assert len(result.requests) == 1
        assert result.sample.response_length > 0


class TestBoundaryConditions:
    # TODO: This test exposes a bug in multi_turn_single_sample.py where `output` is undefined
    # when the loop breaks on the first iteration due to context length exceeded.
    # @pytest.mark.parametrize(
    #     "generation_env",
    #     [{"args_kwargs": {"extra_argv": _make_extra_argv(**{"rollout-max-context-len": str(SINGLE_TURN_PROMPT_TOKEN_LEN)})}}],
    #     indirect=True,
    # )
    # def test_exact_context_limit(self, variant, generation_env):
    #     result = _run_generate(variant, generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))
    #
    #     assert len(result.requests) == 0
    #     assert result.sample.status == Sample.Status.TRUNCATED

    # TODO: This test exposes that when rollout_max_context_len=None and max_tokens_per_gpu=None,
    # the code will fail with TypeError. This scenario may not be realistic in practice.
    # @pytest.mark.parametrize(
    #     "generation_env",
    #     [{"args_kwargs": {"extra_argv": _make_extra_argv(**{"rollout-max-context-len": None})}}],
    #     indirect=True,
    # )
    # def test_no_rollout_max_context_len(self, variant, generation_env):
    #     generation_env.mock_server.process_fn = lambda _: ProcessResult(
    #         text=SINGLE_TURN_RESPONSE, finish_reason="stop"
    #     )
    #
    #     assert generation_env.args.rollout_max_context_len is None
    #
    #     result = _run_generate(variant, generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))
    #
    #     assert len(result.requests) == 1
    #     assert result.sample.status == Sample.Status.COMPLETED
    pass
