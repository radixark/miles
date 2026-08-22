"""Convert Polar rollout results into Miles training samples.

Every trace in ``Trajectory.traces`` becomes one miles-native ``Sample``
(``miles.utils.types.Sample``).  All samples produced from the same session
share ``Sample.group_index`` so the RL loss reducer counts the trajectory once
even when it fans out into multiple trace samples.  Builders own trace
curation and per-token loss masks — the adapter does not infer trainable
positions from bridge details.  Traces that lack training tokens are dropped
and represented as fully masked samples so callers can keep the rest of the
group trainable.

This module is a faithful port of ``slime_bridge.adapter`` aimed at the Miles
rollout package.  The public surface (function names and signatures) matches
the slime source so a sibling rollout module can import it identically.  Polar
is not required at import time: all Polar model types are only referenced as
lazily-imported ``Any`` (attribute-based duck typing), so the module loads
under a plain Miles environment without the ``polar`` package installed.
"""

from __future__ import annotations

from copy import deepcopy
import logging
import os
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from polar.rollout.models import SessionResult
    from polar.trajectory.models import Trace

logger = logging.getLogger(__name__)

__all__ = [
    "RolloutLogprobError",
    "session_result_to_samples",
    "_build_sample",
    "_build_dummy_sample",
    "_reward_value",
    "_scheduler_metadata",
    "_sample_status",
    "_extract_rollout_log_probs",
    "_loss_mask_from_trace",
    "_load_sample_type",
]


class RolloutLogprobError(ValueError):
    """Raised when a trainable Polar trace lacks aligned rollout logprobs."""


def session_result_to_samples(
    result: SessionResult,
    group_index: int,
    *,
    trajectory_index: int,
    reward_key: str = "score",
    max_tokens: int | None = None,
) -> list[Any]:
    """Convert one Polar session result into Miles samples — one per trace.

    Every usable trace becomes an independent ``Sample`` sharing the same
    ``group_index`` key. The RL loss reducer then averages all trace
    contributions as one trajectory, while the reward post-processor can still
    assign each trace its own advantage.

    Traces with empty tokens or exceeding ``max_tokens`` are dropped
    (logged). If *all* traces are dropped we emit a single zero-gradient
    placeholder so the downstream flattener doesn't crash on an empty list and
    the rest of the group can still train.
    """
    Sample = _load_sample_type()
    traces = list(result.trajectory.traces)
    max_traces = int(os.environ.get("MILES_POLAR_MAX_TRACES_PER_SESSION", "0") or 0)
    if max_traces > 0 and len(traces) > max_traces:
        logger.info(
            "Session %s has %d traces; keeping the last %d for the bounded smoke run",
            result.session_id,
            len(traces),
            max_traces,
        )
        traces = traces[-max_traces:]
    samples: list[Any] = []
    for trace_index, trace in enumerate(traces):
        sample = _build_sample(
            Sample=Sample,
            result=result,
            trace=trace,
            trace_index=trace_index,
            group_index=group_index,
            index=trajectory_index,
            reward_key=reward_key,
            max_tokens=max_tokens,
        )
        if sample is not None:
            samples.append(sample)

    if samples:
        return samples

    logger.warning(
        "Session %s: no usable trace (traces=%d, max_tokens=%s); emitting dummy placeholder",
        result.session_id, len(traces), max_tokens,
    )
    return [_build_dummy_sample(
        Sample=Sample,
        result=result,
        group_index=group_index,
        index=trajectory_index,
        reward_key=reward_key,
    )]


def _build_sample(
    *,
    Sample: Any,
    result: SessionResult,
    trace: Trace,
    trace_index: int,
    group_index: int,
    index: int,
    reward_key: str,
    max_tokens: int | None = None,
) -> Any | None:
    prompt_ids = list(trace.prompt_ids)
    response_ids = list(trace.response_ids)

    if not prompt_ids or not response_ids:
        logger.warning(
            "Dropping trace %d from session %s: missing tokens (prompt=%d, response=%d)",
            trace_index, result.session_id, len(prompt_ids), len(response_ids),
        )
        return None

    total_len = len(prompt_ids) + len(response_ids)
    if max_tokens is not None and total_len > max_tokens:
        logger.warning(
            "Dropping trace %d from session %s: total_len=%d > max_tokens=%d",
            trace_index, result.session_id, total_len, max_tokens,
        )
        return None

    prompt_messages = deepcopy(trace.prompt_messages)
    response_messages = deepcopy(trace.response_messages)
    response_text = _messages_to_text(response_messages)

    status = _sample_status(Sample, result, trace)
    reward_value = _reward_value(trace)

    trainable = status not in (Sample.Status.ABORTED, Sample.Status.FAILED)
    loss_mask = _loss_mask_from_trace(
        trace,
        len(response_ids),
        require_loss_mask=trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )
    if status in (Sample.Status.ABORTED, Sample.Status.FAILED):
        loss_mask = [0] * len(response_ids)
    response_log_probs = _extract_rollout_log_probs(
        trace,
        response_len=len(response_ids),
        loss_mask=loss_mask,
        require_trainable_logprobs=trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )

    prompt_value = prompt_messages if prompt_messages else ""

    polar_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_metadata": deepcopy(getattr(result, "metadata", {}) or {}),
        "result_error": result.error,
        "session_id": result.session_id,
        "session_status": result.status,
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": trace_index,
        "trace_metadata": deepcopy(getattr(trace, "metadata", {}) or {}),
        "trajectory_error": result.trajectory.error,
        "trajectory_metadata": deepcopy(result.trajectory.metadata),
        "trajectory_status": result.trajectory.status,
        # Preserved for the longest-trace artifact dump; training reads
        # tokens+logprobs, not these.
        "trace_debug": {
            "finish_reason": trace.finish_reason,
            "response_messages": deepcopy(response_messages),
        },
    }
    polar_metadata.update(_scheduler_metadata(result, trace))

    return Sample(
        group_index=group_index,
        index=index,
        prompt=prompt_value,
        tokens=prompt_ids + response_ids,
        response=response_text,
        response_length=len(response_ids),
        rollout_id=index,
        reward={reward_key: reward_value},
        loss_mask=loss_mask,
        rollout_log_probs=response_log_probs,
        status=status,
        metadata={"polar": polar_metadata},
    )


def _build_dummy_sample(
    *,
    Sample: Any,
    result: SessionResult,
    group_index: int,
    index: int,
    reward_key: str,
) -> Any:
    """Fully masked placeholder for a session with no usable trace.

    This carries no policy, TIS, or KL contribution. It lets the scheduler
    accept a partially usable group while still surfacing empty sessions in
    Polar metrics.
    """
    polar_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_metadata": deepcopy(getattr(result, "metadata", {}) or {}),
        "result_error": result.error,
        "session_id": result.session_id,
        "session_status": result.status,
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": -1,
        "trajectory_error": result.trajectory.error,
        "trajectory_metadata": deepcopy(result.trajectory.metadata),
        "trajectory_status": result.trajectory.status,
        "placeholder": True,
    }
    polar_metadata.update(_scheduler_metadata(result, None))
    return Sample(
        group_index=group_index,
        index=index,
        prompt="",
        tokens=[0, 0],
        response="",
        response_length=1,
        rollout_id=index,
        reward={reward_key: 0.0},
        loss_mask=[0],
        rollout_log_probs=[0.0],
        status=Sample.Status.ABORTED,
        remove_sample=True,
        metadata={"polar": polar_metadata},
    )


def _reward_value(trace: Trace) -> float:
    """Read the reward the evaluator already placed on the trace.

    Reward assignment is the evaluator's job (including any broadcasting
    from session-level outcomes). The adapter just consumes what's there.
    """
    return float(trace.reward) if trace.reward is not None else 0.0


def _scheduler_metadata(result: SessionResult, trace: Trace | None) -> dict[str, Any]:
    keys = {"group_id", "policy_version", "rollout_step"}
    merged: dict[str, Any] = {}
    for source in (
        getattr(result, "metadata", None),
        getattr(result.trajectory, "metadata", None),
        getattr(trace, "metadata", None) if trace is not None else None,
    ):
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source:
                merged[key] = source[key]
    return merged


def _sample_status(Sample: Any, result: SessionResult, trace: Trace) -> Any:
    trajectory_status = result.trajectory.status
    if trajectory_status == "TIMEOUT" or result.status == "TIMEOUT":
        return Sample.Status.ABORTED
    if trajectory_status == "ERROR" or result.status == "ERROR" or result.error or result.trajectory.error:
        return Sample.Status.FAILED
    if trace.finish_reason == "length":
        return Sample.Status.TRUNCATED
    return Sample.Status.COMPLETED


def _extract_rollout_log_probs(
    trace: Trace,
    *,
    response_len: int,
    loss_mask: list[int],
    require_trainable_logprobs: bool,
    session_id: str,
    trace_index: int,
) -> list[float]:
    logprobs = trace.response_logprobs
    if not logprobs:
        if require_trainable_logprobs and any(loss_mask):
            raise RolloutLogprobError(
                f"Session {session_id} trace {trace_index}: missing rollout_log_probs "
                "for trainable response tokens"
            )
        return [0.0] * response_len

    if len(logprobs) != response_len:
        raise RolloutLogprobError(
            f"Session {session_id} trace {trace_index}: rollout_log_probs length "
            f"{len(logprobs)} != response length {response_len}"
        )

    # response_logprobs is one float per response token (interstitials are 0.0,
    # masked out by loss_mask); the builder guarantees trainable tokens carry
    # their real sampled logprob.
    return [float(value) for value in logprobs]


def _loss_mask_from_trace(
    trace: Trace,
    response_len: int,
    *,
    require_loss_mask: bool,
    session_id: str,
    trace_index: int,
) -> list[int]:
    """Read and validate the builder-assigned per-response-token loss mask."""
    mask = list(trace.loss_mask)
    if not mask:
        if require_loss_mask:
            raise RolloutLogprobError(
                f"Session {session_id} trace {trace_index}: missing loss_mask"
            )
        return [0] * response_len
    if len(mask) != response_len:
        raise RolloutLogprobError(
            f"Session {session_id} trace {trace_index}: loss_mask length "
            f"{len(mask)} != response length {response_len}"
        )
    return [1 if int(value) else 0 for value in mask]


def _load_sample_type() -> Any:
    """Return the miles-native ``Sample`` class used for RL training.

    Miles already ships the full training sample type in
    ``miles.utils.types``; no Polar dependency is involved. Imported lazily so
    a plain Miles environment never pays to import ``miles.utils.types`` at
    module load, and so callers can inject a substitute during tests.
    """
    try:
        from miles.utils.types import Sample
    except ImportError as exc:
        raise ImportError(
            "Miles is required to convert Polar rollouts into training samples. "
            "Ensure the Miles package is importable in the current environment."
        ) from exc
    return Sample


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    """Render a list of chat messages into a ``[role] content`` block string.

    Mirrors ``slime_bridge._messages.messages_to_text``. Known limitation:
    drops assistant ``tool_calls`` structure into the plain text view. Training
    consumes tokens + logprobs, so this is only a degraded human-readable
    representation of ``Sample.response``.
    """
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "assistant"))
        content = _flatten_content(message.get("content"))
        if content:
            parts.append(f"[{role}] {content}")
    return "\n\n".join(parts)


def _flatten_content(content: Any) -> str:
    """Render OpenAI-style message content into a plain string.

    Accepts the three shapes the Chat Completions API uses: ``None`` → ``""``,
    ``str`` returned as-is, ``list[dict]`` with each dict's ``"text"`` field
    concatenated. Anything else is coerced via ``str(content)``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif "text" in item:
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    if content is None:
        return ""
    return str(content)
