from dataclasses import dataclass

import pytest
from transformers import AutoTokenizer

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.modular_rollout.orchestration_common import GenerateState
from miles.utils.async_utils import run
from miles.utils.test_utils.mock_sglang_server import ProcessResult
from miles.utils.test_utils.mock_tools import (
    MULTI_TURN_FIRST_PROMPT,
    MULTI_TURN_FIRST_RESPONSE,
    MULTI_TURN_SECOND_RESPONSE,
    SAMPLE_TOOLS,
    mock_execute_tool_function,
    multi_turn_tool_call_process_fn,
)
from miles.utils.types import Sample
from tests.fixtures.generation_fixtures import GenerateEnv, generation_env

_ = generation_env, SAMPLE_TOOLS, mock_execute_tool_function, multi_turn_tool_call_process_fn

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
        response_text = "The answer is 2."
        generation_env.mock_server.process_fn = lambda _: ProcessResult(text=response_text, finish_reason="stop")

        prompt = [{"role": "user", "content": "What is 1+1?"}]
        result = run_generate(generation_env, make_sample(prompt=prompt))

        prompt_with_tools = TOKENIZER.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, tools=SAMPLE_TOOLS
        )
        prompt_token_ids = TOKENIZER(prompt_with_tools, add_special_tokens=False)["input_ids"]
        response_token_ids = TOKENIZER.encode(response_text, add_special_tokens=False)
        response_log_probs = [(-1 / 128 * i) for i in range(len(response_token_ids))]

        assert result.requests == [
            {
                "input_ids": prompt_token_ids,
                "sampling_params": DEFAULT_SAMPLING_PARAMS,
                "return_logprob": True,
            }
        ]
        assert result.sample == expected_sample(
            prompt=prompt,
            response=response_text,
            response_length=len(response_token_ids),
            tokens=prompt_token_ids + response_token_ids,
            rollout_log_probs=response_log_probs,
            loss_mask=[1] * len(response_token_ids),
        )

    @pytest.mark.parametrize(
        "generation_env",
        [{"args_kwargs": {"extra_argv": MULTI_TURN_EXTRA_ARGV}}],
        indirect=True,
    )
    def test_two_turns_with_tool_call(self, generation_env):
        generation_env.mock_server.process_fn = multi_turn_tool_call_process_fn

        prompt = [{"role": "user", "content": MULTI_TURN_FIRST_PROMPT}]
        result = run_generate(generation_env, make_sample(prompt=prompt))

        prompt_with_tools = TOKENIZER.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, tools=SAMPLE_TOOLS
        )
        prompt_token_ids = TOKENIZER(prompt_with_tools, add_special_tokens=False)["input_ids"]

        first_response_token_ids = TOKENIZER.encode(MULTI_TURN_FIRST_RESPONSE, add_special_tokens=False)
        tool_response_token_ids = result.sample.tokens[
            len(prompt_token_ids) + len(first_response_token_ids) : -len(
                TOKENIZER.encode(MULTI_TURN_SECOND_RESPONSE, add_special_tokens=False)
            )
        ]
        second_response_token_ids = TOKENIZER.encode(MULTI_TURN_SECOND_RESPONSE, add_special_tokens=False)

        all_response_token_ids = first_response_token_ids + tool_response_token_ids + second_response_token_ids
        expected_loss_mask = (
            [1] * len(first_response_token_ids)
            + [0] * len(tool_response_token_ids)
            + [1] * len(second_response_token_ids)
        )
        expected_log_probs = (
            [(-1 / 128 * i) for i in range(len(first_response_token_ids))]
            + [0.0] * len(tool_response_token_ids)
            + [(-1 / 128 * i) for i in range(len(second_response_token_ids))]
        )

        assert len(result.requests) == 2
        assert result.sample == expected_sample(
            prompt=prompt,
            response=TOKENIZER.decode(all_response_token_ids),
            response_length=len(all_response_token_ids),
            tokens=prompt_token_ids + all_response_token_ids,
            rollout_log_probs=expected_log_probs,
            loss_mask=expected_loss_mask,
        )
