"""
Generic agentic generate function for agent-environment RL training.

The agent logic is fully encapsulated in a user-provided async function
(--custom-agent-function-path). This generate function only handles:
  1. TITO session tracing (OpenAIEndpointTracer)
  2. Collecting the worker-assembled training samples (the session server
     converts records to samples, truncates and merges in the owning worker)
  3. Driver-side metadata application (agent_metadata, session_metadata)

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
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    assert getattr(input.args, "session_server_ip", None) and getattr(input.args, "session_server_ports", None), (
        "agentic_tool_call.generate requires session_server_ip/session_server_ports. "
        "Pass --use-session-server to start the session server."
    )
    tracer = await OpenAIEndpointTracer.create(input.args)

    custom_agent_function: Callable = load_function(input.args.custom_agent_function_path)
    assert (
        custom_agent_function is not None
    ), f"Custom agent function {input.args.custom_agent_function_path} not found"

    max_seq_len = getattr(input.args, "max_seq_len", None)

    metadata = input.sample.metadata
    if max_seq_len is not None:
        metadata = {**metadata, "max_seq_len": max_seq_len}
    if tracer.session_server_instance_id:
        metadata = {**metadata, "session_server_instance_id": tracer.session_server_instance_id}

    log_prefix = f"[session={tracer.session_id}]"

    # From the tracer, not args: with multiple instances the owning ip:port is per-session.
    metadata = {**metadata, "session_server_id": tracer.session_server_id}

    agent_metadata = None
    t_start = time.monotonic()
    try:
        logger.debug(f"{log_prefix} Starting agent function call")
        agent_metadata = await custom_agent_function(
            base_url=tracer.base_url,
            prompt=input.sample.prompt,
            request_kwargs=build_chat_request_kwargs(input.sampling_params),
            metadata=metadata,
        )
        logger.debug(f"{log_prefix} Agent function returned in {time.monotonic()-t_start:.1f}s")
    except Exception as e:
        logger.warning(f"{log_prefix} Agent function failed: {e}", exc_info=True)

    finally:
        # The session server assembles the samples on the owning instance; records
        # never leave it. Runs even when the agent function failed, like the old
        # collect_records; a collect failure (422/5xx/timeout) raises loudly.
        logger.debug(f"{log_prefix} Calling collect_samples...")
        result = await tracer.collect_samples(
            input.sample,
            multi_samples=input.args.generate_multi_samples,
            max_seq_len=max_seq_len,
        )
        logger.debug(
            f"{log_prefix} collect_samples done: {len(result.samples)} samples, "
            f"total_time={time.monotonic()-t_start:.1f}s"
        )

    if not result.samples:
        if result.empty_reason == "all_truncated":
            logger.warning("All samples truncated (prompt already exceeds max_seq_len)")
        else:
            logger.warning("No model calls recorded for sample")
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)

    samples = result.samples
    for s in samples:
        s.metadata.update(agent_metadata or {})

    # If the agent function reports wall-clock time spent outside policy generation
    # (env/tool steps), surface it on Sample.non_generation_time so throughput
    # accounting subtracts it (applied to every returned sample, merged or per-turn).
    ngt = ((agent_metadata or {}).get("agent_metrics") or {}).get("total_tool_time")
    if ngt is not None:
        for s in samples:
            s.non_generation_time = ngt

    if not input.args.generate_multi_samples:
        # The server merged already; unwrap to a scalar Sample — downstream
        # forks on isinstance(sample, list) to pick the multi-samples branch.
        (merged,) = samples
        merged.metadata.update(result.session_metadata)
        return GenerateFnOutput(samples=merged)
    samples[-1].metadata.update(result.session_metadata)
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
        "per session. Truncation happens inside the session server during sample assembly; "
        "also forwarded to the Harbor agent server (as max_seq_len) to abort the trial early.",
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
