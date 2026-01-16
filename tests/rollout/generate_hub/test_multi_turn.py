from dataclasses import dataclass

import pytest

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.modular_rollout.orchestration_common import GenerateState
from miles.utils.async_utils import run
from miles.utils.test_utils.mock_sglang_server import ProcessResult
from miles.utils.test_utils.mock_tools import (
    MULTI_TURN_FIRST_PROMPT,
    SAMPLE_TOOLS,
    mock_execute_tool_function,
    multi_turn_tool_call_process_fn,
)
from miles.utils.types import Sample
from tests.fixtures.generation_fixtures import GenerateEnv, generation_env

_ = generation_env, SAMPLE_TOOLS, mock_execute_tool_function, multi_turn_tool_call_process_fn

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_SAMPLING_PARAMS = {"max_new_tokens": 64, "temperature": 0.7}

SINGLE_TURN_PROMPT = [{"role": "user", "content": "What is 1+1?"}]
SINGLE_TURN_RESPONSE = "The answer is 2."
SINGLE_TURN_PROMPT_TOKENS = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220, 16, 10, 16, 30, 151645, 198, 151644, 77091, 198]  # fmt: skip
SINGLE_TURN_RESPONSE_TOKENS = [785, 4226, 374, 220, 17, 13]
SINGLE_TURN_RESPONSE_LOG_PROBS = [-0.0, -0.0078125, -0.015625, -0.0234375, -0.03125, -0.0390625]

TWO_TURN_PROMPT = [{"role": "user", "content": MULTI_TURN_FIRST_PROMPT}]
TWO_TURN_FIRST_RESPONSE_TOKENS = [10061, 752, 633, 279, 1042, 323, 9444, 1156, 624, 198, 27, 14449, 4356, 397, 5765, 606, 794, 330, 455, 14987, 497, 330, 14799, 794, 4687, 534, 522, 14449, 4356, 397, 27, 14449, 4356, 397, 5765, 606, 794, 330, 455, 54625, 497, 330, 14799, 794, 5765, 2588, 794, 330, 41, 1590, 45034, 534, 522, 14449, 4356, 29]  # fmt: skip
TWO_TURN_TOOL_RESPONSE_TOKENS = [151645, 198, 151644, 11880, 320, 14449, 4356, 1754, 25, 6253, 931, 15, 8, 151645, 198, 5765, 3236, 794, 220, 17, 15, 17, 21, 534, 151644, 11880, 320, 14449, 4356, 1754, 25, 6253, 931, 16, 8, 151645, 198, 5765, 35264, 794, 481, 21, 15, 534, 151644, 77091, 198]  # fmt: skip
TWO_TURN_SECOND_RESPONSE_TOKENS = [785, 4226, 374, 25, 220, 19, 17, 488, 220, 17, 15, 17, 21, 488, 481, 21, 15, 284, 220, 17, 15, 15, 23, 13]  # fmt: skip

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


def expected_sample(
    *,
    prompt: list[dict],
    response: str,
    response_length: int,
    tokens: list[int],
    rollout_log_probs: list[float],
    loss_mask: list[int] | None = None,
    status: Sample.Status = Sample.Status.COMPLETED,
) -> Sample:
    return Sample(
        group_index=None,
        index=None,
        prompt=prompt,
        tokens=tokens,
        multimodal_inputs=None,
        multimodal_train_inputs=None,
        response=response,
        response_length=response_length,
        label=None,
        reward=None,
        loss_mask=loss_mask,
        weight_versions=[],
        rollout_log_probs=rollout_log_probs,
        rollout_routed_experts=None,
        remove_sample=False,
        status=status,
        metadata={},
        train_metadata=None,
        non_generation_time=0.0,
        spec_info=Sample.SpecInfo(),
        prefix_cache_info=Sample.PrefixCacheInfo(cached_tokens=0, total_prompt_tokens=0),
    )


def make_sample(prompt=None):
    return Sample(
        prompt=prompt or [{"role": "user", "content": "What is 1+1?"}],
        tokens=[],
        response="",
        response_length=0,
        status=Sample.Status.PENDING,
    )


async def call_multi_turn_generate(args, sample: Sample, sampling_params: dict) -> Sample:
    from miles.rollout.generate_hub.multi_turn_single_sample import generate

    state = GenerateState(args)
    output = await generate(
        GenerateFnInput(state=state, sample=sample, sampling_params=sampling_params.copy(), evaluation=False)
    )
    return output.samples


def run_generate(env: GenerateEnv, sample: Sample | None = None, sampling_params: dict | None = None):
    env.mock_server.request_log.clear()
    result_sample = run(
        call_multi_turn_generate(env.args, sample or make_sample(), sampling_params or DEFAULT_SAMPLING_PARAMS)
    )
    return GenerateResult(sample=result_sample, requests=list(env.mock_server.request_log))


class TestBasicMultiTurn:
    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"extra_argv": MULTI_TURN_EXTRA_ARGV}}],
        indirect=True,
    )
    def test_single_turn_no_tool_call(self, generation_env):
        generation_env.mock_server.process_fn = lambda _: ProcessResult(text=SINGLE_TURN_RESPONSE, finish_reason="stop")

        result = run_generate(generation_env, make_sample(prompt=SINGLE_TURN_PROMPT))

        assert result.requests == [
            {
                "input_ids": SINGLE_TURN_PROMPT_TOKENS,
                "sampling_params": DEFAULT_SAMPLING_PARAMS,
                "return_logprob": True,
            }
        ]
        assert result.sample == expected_sample(
            prompt=SINGLE_TURN_PROMPT,
            response=SINGLE_TURN_RESPONSE,
            response_length=len(SINGLE_TURN_RESPONSE_TOKENS),
            tokens=SINGLE_TURN_PROMPT_TOKENS + SINGLE_TURN_RESPONSE_TOKENS,
            rollout_log_probs=SINGLE_TURN_RESPONSE_LOG_PROBS,
            loss_mask=[1] * len(SINGLE_TURN_RESPONSE_TOKENS),
        )

    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"extra_argv": MULTI_TURN_EXTRA_ARGV}}],
        indirect=True,
    )
    def test_two_turns_with_tool_call(self, generation_env):
        generation_env.mock_server.process_fn = multi_turn_tool_call_process_fn

        result = run_generate(generation_env, make_sample(prompt=TWO_TURN_PROMPT))

        prompt_tokens = result.requests[0]["input_ids"]
        all_response_tokens = TWO_TURN_FIRST_RESPONSE_TOKENS + TWO_TURN_TOOL_RESPONSE_TOKENS + TWO_TURN_SECOND_RESPONSE_TOKENS
        expected_loss_mask = (
            [1] * len(TWO_TURN_FIRST_RESPONSE_TOKENS)
            + [0] * len(TWO_TURN_TOOL_RESPONSE_TOKENS)
            + [1] * len(TWO_TURN_SECOND_RESPONSE_TOKENS)
        )
        expected_log_probs = (
            [(-1 / 128 * i) for i in range(len(TWO_TURN_FIRST_RESPONSE_TOKENS))]
            + [0.0] * len(TWO_TURN_TOOL_RESPONSE_TOKENS)
            + [(-1 / 128 * i) for i in range(len(TWO_TURN_SECOND_RESPONSE_TOKENS))]
        )

        assert result.requests == [
            {
                "input_ids": prompt_tokens,
                "sampling_params": DEFAULT_SAMPLING_PARAMS,
                "return_logprob": True,
            },
            {
                "input_ids": prompt_tokens + TWO_TURN_FIRST_RESPONSE_TOKENS + TWO_TURN_TOOL_RESPONSE_TOKENS,
                "sampling_params": DEFAULT_SAMPLING_PARAMS,
                "return_logprob": True,
            },
        ]
        assert result.sample == expected_sample(
            prompt=TWO_TURN_PROMPT,
            response=result.sample.response,
            response_length=len(all_response_tokens),
            tokens=prompt_tokens + all_response_tokens,
            rollout_log_probs=expected_log_probs,
            loss_mask=expected_loss_mask,
        )
