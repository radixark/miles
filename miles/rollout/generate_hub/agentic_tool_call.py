"""
Simple agentic demo with tool calling.
"""

import argparse
import logging
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.agentic_types import AgentResult
from miles.rollout.generate_utils.openai_endpoint_utils import (
    OpenAIEndpointTracer,
    compute_samples_from_openai_records,
)
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_STATUS_MAP = {
    "completed": Sample.Status.COMPLETED,
    "truncated": Sample.Status.TRUNCATED,
    "aborted": Sample.Status.ABORTED,
}


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    tracer = await OpenAIEndpointTracer.create(input.args)

    custom_agent_function: Callable = load_function(input.args.custom_agent_function_path)
    assert (
        custom_agent_function is not None
    ), f"Custom agent function {input.args.custom_agent_function_path} not found"

    # Get agent result from custom agent function
    agent_result = await custom_agent_function(
        base_url=tracer.base_url,
        prompt=input.sample.prompt,
        request_kwargs=build_chat_request_kwargs(input.sampling_params),
        metadata=input.sample.metadata,
    )

    if agent_result is None:
        agent_result = AgentResult()

    records = await tracer.collect_records()

    if not records:
        logger.warning("No model calls recorded for sample")
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        sample.reward = 0.0
        return GenerateFnOutput(samples=sample)

    # Convert session records to samples, but now all rewards are 0.0
    samples = compute_samples_from_openai_records(input.sample, records, input.state.tokenizer)
    
    # Apply agent result to samples
    _apply_agent_result(samples, agent_result)

    if not input.args.generate_multi_samples:
        if any(s.status == Sample.Status.ABORTED for s in samples):
            return GenerateFnOutput(samples=samples[-1])
        merged = merge_samples(samples, input.state.tokenizer)
        return GenerateFnOutput(samples=merged)

    return GenerateFnOutput(samples=samples)


def _apply_agent_result(samples: list[Sample], result: AgentResult) -> None:
    """Apply agent result (reward, status, metrics) to all samples."""
    status = _STATUS_MAP.get(result.status)

    for s in samples:
        s.reward = result.reward
        s.metadata.update({"reward": result.reward, **result.metadata})
        if result.metrics:
            s.metadata["agent_metrics"] = result.metrics

    if status == Sample.Status.COMPLETED:
        for s in samples[:-1]:
            s.status = Sample.Status.COMPLETED
    elif status == Sample.Status.TRUNCATED:
        samples[-1].status = Sample.Status.TRUNCATED
    elif status == Sample.Status.ABORTED:
        for s in samples:
            s.status = Sample.Status.ABORTED
            s.reward = 0.0


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--custom-agent-function-path", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true", default=False)


generate.add_arguments = _add_arguments


# Process keys to match ChatCompletionRequest input
def build_chat_request_kwargs(sampling_params: dict[str, Any]) -> dict[str, Any]:
    request_kwargs = dict(sampling_params)
    key_map = {
        "max_new_tokens": "max_tokens",
        "min_new_tokens": "min_tokens",
        "sampling_seed": "seed",
    }
    for src, dst in key_map.items():
        if src in request_kwargs:
            if dst not in request_kwargs:
                request_kwargs[dst] = request_kwargs[src]
            request_kwargs.pop(src, None)

    # Notice: Here we force the inference backend to return token information and start from 0
    # The start len should be 0 to make sure prompt token ids and be correctly returned from SGLang.
    request_kwargs["logprobs"] = True
    request_kwargs["logprob_start_len"] = 0

    reserved_keys = {"model", "messages"}
    allowed_keys = set(ChatCompletionRequest.model_fields) - reserved_keys
    return {key: value for key, value in request_kwargs.items() if key in allowed_keys and value is not None}
