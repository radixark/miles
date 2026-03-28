"""
Generic agentic generate function for agent-environment RL training.

The agent logic is fully encapsulated in a user-provided async function
(--custom-agent-function-path). This generate function only handles:
  1. TITO session tracing (OpenAIEndpointTracer)
  2. Converting session records to training samples
  3. Multi-turn merge

Agent function contract:
  async def my_agent(
      base_url: str,
      prompt: ...,
      request_kwargs: dict,
      metadata: dict,       # sample.metadata — env-specific fields
      **kwargs,
  ) -> dict | None:
      ...

  Returning None means no extra metadata to attach.
  Returning a dict merges it into every sample's metadata, so downstream
  reward models (--custom-rm-path) can read whatever the agent left there.
"""

import argparse
import logging
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.openai_endpoint_utils import (
    OpenAIEndpointTracer,
    compute_samples_from_openai_records,
    truncate_samples_by_total_tokens,
)
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    tracer = await OpenAIEndpointTracer.create(input.args)

    custom_agent_function: Callable = load_function(input.args.custom_agent_function_path)
    assert (
        custom_agent_function is not None
    ), f"Custom agent function {input.args.custom_agent_function_path} not found"

    max_seq_len = getattr(input.args, "max_seq_len", None)

    metadata = input.sample.metadata
    if max_seq_len is not None:
        metadata = {**metadata, "max_seq_len": max_seq_len}

    agent_metadata = await custom_agent_function(
        base_url=tracer.base_url,
        prompt=input.sample.prompt,
        request_kwargs=build_chat_request_kwargs(input.sampling_params),
        metadata=metadata,
    )

    records, session_metadata = await tracer.collect_records()

    if not records:
        logger.warning("No model calls recorded for sample")
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)

    samples = compute_samples_from_openai_records(
        input.args,
        input.sample,
        records,
        input.state.tokenizer,
        accumulated_token_ids=session_metadata.get("accumulated_token_ids"),
        max_trim_tokens=session_metadata.get("max_trim_tokens", 0),
    )

    for s in samples:
        s.metadata.update(agent_metadata or {})

    if max_seq_len is not None:
        samples = truncate_samples_by_total_tokens(samples, max_seq_len, input.state.tokenizer)

    if not samples:
        logger.warning("All samples truncated (prompt already exceeds max_seq_len)")
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)

    if not input.args.generate_multi_samples:
        samples = merge_samples(samples, input.state.tokenizer)
        samples.metadata.update(session_metadata)
    else:
        samples[-1].metadata.update(session_metadata)
    return GenerateFnOutput(samples=samples)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--custom-agent-function-path", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true", default=False)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        dest="max_seq_len",
        help="Max sequence length in tokens (prompt + completion, including env responses) "
        "per session. Truncates samples on the Miles side and is forwarded to the "
        "Harbor agent server (as max_seq_len) to abort the trial early.",
    )


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

    reserved_keys = {"model", "messages"}
    allowed_keys = set(ChatCompletionRequest.model_fields) - reserved_keys
    return {key: value for key, value in request_kwargs.items() if key in allowed_keys and value is not None}
