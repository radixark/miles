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
import os
import time
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
from miles.rollout.inference_rollout.inference_rollout_common import AbortHandle
from miles.utils.misc import load_function
from miles.utils.http_utils import post
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

AGENTIC_TERMINAL_TURN_STATUS_KEY = "agentic_terminal_turn_status"
AGENTIC_TERMINAL_TURN_INDEX_KEY = "agentic_terminal_turn_index"
AGENTIC_TERMINAL_TURN_DROPPED_COUNT_KEY = "agentic_terminal_turn_dropped_count"
AGENTIC_INTERMEDIATE_TRUNCATED_KEY = "agentic_intermediate_truncated"


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    assert getattr(input.args, "session_server_ip", None) and getattr(input.args, "session_server_port", None), (
        "agentic_tool_call.generate requires session_server_ip/session_server_port. "
        "Pass --use-session-server to start the session server."
    )
    tracer = await OpenAIEndpointTracer.create(input.args)

    agent_server_url = os.getenv(
        "AGENT_SERVER_URL",
        os.getenv("SWE_AGENT_URL", "http://localhost:11000"),
    )
    logger.info(
        "[rollout-abort] registering Harbor cancel handler session=%s url=%s",
        tracer.session_id,
        agent_server_url,
    )

    async def cancel_harbor_session() -> None:
        logger.info("[rollout-abort] closing Miles session=%s", tracer.session_id)
        session_closed = await tracer.close_session()
        logger.info("[rollout-abort] Miles session close finished session=%s closed=%s", tracer.session_id, session_closed)
        logger.info("[rollout-abort] sending Harbor cancel session=%s", tracer.session_id)
        response = await post(
            f"{agent_server_url}/admin/abort",
            {"request_id": tracer.session_id},
            max_retries=2,
        )
        logger.info("[rollout-abort] Harbor cancel finished session=%s response=%s", tracer.session_id, response)

    await input.state.add_abort_handle(
        AbortHandle(
            key=f"harbor-session:{tracer.session_id}",
            abort=cancel_harbor_session,
            label="harbor-session",
        )
    )

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

    session_ip = getattr(input.args, "session_server_ip", None)
    session_port = getattr(input.args, "session_server_port", None)
    if session_ip and session_port:
        metadata = {**metadata, "session_server_id": f"{session_ip}:{session_port}"}

    agent_metadata = None
    t_start = time.monotonic()
    agent_kwargs = {
        "trials_subdir": input.args.wandb_group,
        "environment_build_timeout_sec": input.args.agent_environment_build_timeout_sec,
        "agent_setup_timeout_sec": input.args.agent_setup_timeout_sec,
        "agent_timeout_sec": input.args.agent_timeout_sec,
        "verifier_timeout_sec": input.args.agent_verifier_timeout_sec,
        "agent_run_max_attempts": input.args.agent_run_max_attempts,
        "agent_run_retry_backoff_sec": input.args.agent_run_retry_backoff_sec,
        "session_id": tracer.session_id,
    }
    agent_kwargs = {k: v for k, v in agent_kwargs.items() if v is not None}
    try:
        logger.debug(f"{log_prefix} Starting agent function call")
        agent_metadata = await custom_agent_function(
            base_url=tracer.base_url,
            prompt=input.sample.prompt,
            request_kwargs=build_chat_request_kwargs(input.sampling_params),
            metadata=metadata,
            **agent_kwargs,
            # rollout_id=input.state.rollout_id
        )

        logger.debug(f"{log_prefix} Agent function returned in {time.monotonic()-t_start:.1f}s")
    except Exception as e:
        logger.warning(f"{log_prefix} Agent function failed: {e}", exc_info=True)

    finally:
        try:
            logger.debug(f"{log_prefix} Calling collect_records...")
            records, session_metadata = await tracer.collect_records()
            logger.debug(f"{log_prefix} collect_records done: {len(records)} records")
        finally:
            logger.info("[rollout-abort] removing Harbor cancel handler session=%s", tracer.session_id)
            await input.state.remove_abort_handle(f"harbor-session:{tracer.session_id}")

    if not records:
        logger.warning("No model calls recorded for sample")
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)

    logger.debug(f"{log_prefix} Computing samples from {len(records)} records...")
    samples = compute_samples_from_openai_records(
        input.args,
        input.sample,
        records,
        input.state.tokenizer,
        accumulated_token_ids=session_metadata.get("accumulated_token_ids"),
        max_trim_tokens=session_metadata.get("max_trim_tokens", 0),
    )

    logger.debug(
        f"{log_prefix} compute_samples done: {len(samples)} samples, total_time={time.monotonic()-t_start:.1f}s"
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

    samples, terminal_turn_metadata = _cut_after_first_terminal_turn(samples, log_prefix)

    if not input.args.generate_multi_samples:
        samples = merge_samples(samples, input.state.tokenizer)
        samples.metadata.update(terminal_turn_metadata)
        samples.metadata.update(session_metadata)
    else:
        for sample in samples:
            sample.metadata.update(terminal_turn_metadata)
        samples[-1].metadata.update(session_metadata)
    return GenerateFnOutput(samples=samples)


def _cut_after_first_terminal_turn(samples: list[Sample], log_prefix: str) -> tuple[list[Sample], dict[str, Any]]:
    """Stop multi-turn merging after the first non-completed model turn.

    A terminal middle turn means the agent kept issuing requests after one
    response was already truncated/aborted. ``merge_samples`` can represent a
    terminal final turn, but it cannot safely merge more turns after it.
    """
    for turn_index, sample in enumerate(samples[:-1]):
        if sample.status == Sample.Status.COMPLETED:
            continue

        dropped_count = len(samples) - turn_index - 1
        metadata = {
            AGENTIC_TERMINAL_TURN_STATUS_KEY: str(sample.status),
            AGENTIC_TERMINAL_TURN_INDEX_KEY: turn_index,
            AGENTIC_TERMINAL_TURN_DROPPED_COUNT_KEY: dropped_count,
            AGENTIC_INTERMEDIATE_TRUNCATED_KEY: sample.status == Sample.Status.TRUNCATED,
        }
        logger.warning(
            "%s terminal intermediate agent turn: status=%s turn_index=%d total_turns=%d dropped_later_turns=%d",
            log_prefix,
            sample.status,
            turn_index,
            len(samples),
            dropped_count,
        )
        return samples[: turn_index + 1], metadata

    return samples, {}


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
    parser.add_argument(
        "--agent-environment-build-timeout-sec",
        type=float,
        default=300,
        help="Timeout for building the agent Docker environment (forwarded to Harbor agent server).",
    )
    parser.add_argument(
        "--agent-setup-timeout-sec",
        type=float,
        default=1200,
        help="Timeout for agent setup before execution starts (forwarded to Harbor agent server).",
    )
    parser.add_argument(
        "--agent-timeout-sec",
        type=float,
        default=3600,
        help="Timeout for agent execution per trial (forwarded to Harbor agent server).",
    )
    parser.add_argument(
        "--agent-verifier-timeout-sec",
        type=float,
        default=600,
        help="Timeout for verifier/grading per trial (forwarded to Harbor agent server).",
    )
    parser.add_argument(
        "--agent-run-max-attempts",
        type=int,
        default=2,
        help="Max attempts for submitting Harbor /run. Only transient connection failures are retried.",
    )
    parser.add_argument(
        "--agent-run-retry-backoff-sec",
        type=float,
        default=0.5,
        help="Linear backoff base for retrying transient Harbor /run submit failures.",
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
