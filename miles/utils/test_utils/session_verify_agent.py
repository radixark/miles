"""Custom-generate / custom-agent driver for TITO session-server verification.

Wired through ``--custom-generate-function-path`` /
``--custom-agent-function-path``; consumed by
``tests/e2e/sglang/test_session_server_multi_role/`` (one test file per
model family) and ``scripts/tools/verify_session_tito_tokenizer.py``.
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum
from functools import wraps

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

import httpx

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.agentic_tool_call import generate as _base_generate
from miles.utils.chat_template_utils.tito_tokenizer import VALID_APPEND_ROLES, TITOTokenizerType
from miles.utils.test_utils.openai_stream_client import stream_chat_completions

logger = logging.getLogger(__name__)


def _append_session_verify_record(entry: dict) -> bool:
    metrics_path = os.environ.get("MILES_SESSION_VERIFY_METRICS_PATH")
    if not metrics_path:
        return False
    payload = (json.dumps(entry) + "\n").encode()
    fd = os.open(
        metrics_path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError(f"short metrics sidecar write: expected {len(payload)} bytes, wrote {written}")
    finally:
        os.close(fd)
    return True


def _journal_verifier_assertions(stage: str):
    def decorate(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except AssertionError as exc:
                _append_session_verify_record(
                    {
                        "verification_error": {
                            "stage": stage,
                            "type": type(exc).__name__,
                            "message": str(exc)[:1000],
                        }
                    }
                )
                raise

        return wrapped

    return decorate


class DriverAction(Enum):
    TOOL_RESULT = "tool_result"
    USER_FOLLOWUP = "user_followup"
    SYSTEM_REMINDER = "system_reminder"
    ASSISTANT_INPUT = "assistant_input"
    ROLLBACK = "rollback"
    FORCE_FINAL = "force_final"


_T = DriverAction.TOOL_RESULT
_U = DriverAction.USER_FOLLOWUP
_S = DriverAction.SYSTEM_REMINDER
_A = DriverAction.ASSISTANT_INPUT
_R = DriverAction.ROLLBACK
_F = DriverAction.FORCE_FINAL


class ToolCallFailureMode(StrEnum):
    """Recovery strategy when a TOOL_RESULT step finds the assistant emitted no tool_calls.

    APPEND_TOOL  : Splice a sentinel ``tool`` message and continue.  Works on
                   lenient templates; strict templates that hard-assert any
                   ``tool`` role must follow an assistant with ``tool_calls``
                   (e.g. MiniMax-M2.7) will reject the next request at server-side.
    APPEND_USER  : Splice a ``user`` message carrying the same failure text as
                   APPEND_TOOL.  Requires the selected family's fixed template
                   to support "user"; raises ValueError at agent start otherwise.
    ROLLBACK     : Pop the offending assistant and let the loop's chat call at
                   the bottom re-inference.  Universal — no role-surface
                   dependency — and the default.
    """

    APPEND_TOOL = "append_tool"
    APPEND_USER = "append_user"
    ROLLBACK = "rollback"


DEFAULT_TOOL_CALL_FAILURE_MODE = ToolCallFailureMode.ROLLBACK

# Cap consecutive ROLLBACK retries — same context every time, so a model that
# never tool-calls would loop forever.
MAX_CONSECUTIVE_TOOL_CALL_FAILURE_ROLLBACKS = 3

# Same body for both APPEND_TOOL and APPEND_USER fallbacks; only the role of
# the spliced message differs between the two modes.
TOOL_CALL_PARSE_FAILURE_TEXT = (
    "Tool call parsing failed: the previous assistant turn did not emit a "
    "parseable tool_call. Please retry with a valid tool invocation."
)

# Mismatch tiers reported by the session-server's per-sample comparator
# (sessions.py:83).  Any occurrence of these "hard" types in a sample's
# tito_session_mismatch indicates a TITO bug and fails the verifier run.  The
# soft `assistant_text` tier is excluded — it is aggregated across samples
# and gated by a ratio threshold instead.
_FORBIDDEN_MISMATCH_TYPES: frozenset[str] = frozenset(
    {"special_token_count", "special_token_type", "non_assistant_text"}
)

# Override per call: ``--session-verify-cycles N`` (CLI) or ``cycles=N``
# (pytest via ``run_session_verify``).  Smaller-context models with a 4K
# response budget should drop to 2 to avoid context overflow.
DEFAULT_CYCLES = 3
_MAX_INCOMPLETE_TURN_RETRIES = 2
_RETRY_SEED_STRIDE = 1_000_000


def fixed_template_append_roles(tito_model: TITOTokenizerType | str) -> tuple[str, ...]:
    """Return the selected family's fixed append capability in canonical order."""
    tokenizer_type = TITOTokenizerType(tito_model)
    supported = TITOTokenizerType.get_tokenizer_class(tokenizer_type).FIXED_TEMPLATE.allowed_append_roles
    return tuple(role for role in VALID_APPEND_ROLES if role in supported)


def _build_cycle(role_surface: frozenset[str]) -> list[DriverAction]:
    cycle: list[DriverAction] = [_T]
    if "user" in role_surface:
        cycle.append(_U)
        cycle.append(_T)
    if "system" in role_surface:
        cycle.append(_S)
    cycle.append(_R)
    return cycle


# English-only on purpose: matches the production agentic flows tokenization
# and tool-call parsing are tuned against.
USER_FOLLOWUP_TEXT = "Now check the weather in Shanghai."
SYSTEM_REMINDER_TEXT = "Note: from now on, answer in a single sentence; skip all pleasantries."
ASSISTANT_INPUT_TEXTS = (
    "The earlier Beijing weather result was 22 degrees Celsius and sunny.",
    "The earlier Shanghai weather result was 30 degrees Celsius and rainy.",
)
ASSISTANT_INPUT_FOLLOWUP_TEXT = "Summarize the two weather results above in one sentence without calling a tool."
FORCE_FINAL_TEXT = "Please summarize all results inside <final_answer>...</final_answer> tags."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. Beijing",
                    },
                },
                "required": ["location"],
            },
        },
    },
]

MOCK_TOOL_RESULTS = [
    '{"temperature_celsius": 22, "condition": "sunny"}',
    '{"temperature_celsius": 15, "condition": "cloudy"}',
    '{"temperature_celsius": 30, "condition": "rainy"}',
    '{"temperature_celsius": 8, "condition": "snowy"}',
]


INITIAL_SYSTEM_PROMPT = (
    "You are a weather assistant.  Use the get_weather tool when the user asks "
    "about a city's weather.  Answer one question at a time and wait for the "
    "next user message; do not summarize until the user explicitly asks you "
    "to.  When asked to summarize, wrap the final summary in "
    "<final_answer>...</final_answer> tags."
)
INITIAL_USER_PROMPT = "What's the weather in Beijing?"


def select_schedule(allowed_roles, *, cycles: int = DEFAULT_CYCLES) -> list[DriverAction]:
    """Build a schedule from the selected FixedTemplate capability."""
    key = frozenset(allowed_roles)
    invalid = key - set(VALID_APPEND_ROLES)
    if invalid:
        raise ValueError(f"Unknown append roles: {sorted(invalid)}")
    if "tool" not in key:
        raise ValueError(f"The session verifier requires 'tool' capability, got {sorted(key)}")
    if cycles < 1:
        raise ValueError(f"cycles must be >= 1, got {cycles}")
    cycle = _build_cycle(key)
    # Extra R after the first cycle exercises consecutive-rollback adjacency,
    # which cycle-repeat alone never produces.
    schedule = list(cycle) + [_R] + cycle * (cycles - 1)
    if "user" in key:
        schedule.append(_F)
    # Keep injected assistants after every rollback: they are prompt messages,
    # not generated checkpoints, so rollback across them is a separate contract.
    if "assistant" in key:
        schedule.append(_A)
    return schedule


def build_initial_messages() -> list[dict]:
    """The fixed (system, user) prompt all schedules start from."""
    return [
        {"role": "system", "content": INITIAL_SYSTEM_PROMPT},
        {"role": "user", "content": INITIAL_USER_PROMPT},
    ]


async def _chat(client, base_url, messages, request_kwargs, *, label):
    payload = {"messages": messages, "tools": TOOLS, **request_kwargs}
    # Streaming is the e2e default: black-box agent harnesses mostly consume
    # chat completions as SSE, so exercise the session server's fake-streaming
    # path unless the caller opts out with stream=False in request_kwargs.
    if payload.pop("stream", True):
        return await stream_chat_completions(client, f"{base_url}/v1/chat/completions", payload, label=label)
    resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
    assert resp.status_code == 200, f"{label} failed ({resp.status_code}): {resp.text}"
    return resp.json()


async def _chat_complete(client, base_url, messages, request_kwargs, *, label):
    for retry in range(_MAX_INCOMPLETE_TURN_RETRIES + 1):
        attempt_kwargs = request_kwargs
        if retry and request_kwargs.get("seed") is not None:
            attempt_kwargs = {
                **request_kwargs,
                "seed": request_kwargs["seed"] + retry * _RETRY_SEED_STRIDE,
            }
        response = await _chat(client, base_url, messages, attempt_kwargs, label=label)
        if response["choices"][0].get("finish_reason") != "length":
            return response
    raise AssertionError(f"{label} exhausted {_MAX_INCOMPLETE_TURN_RETRIES} retries after finish_reason='length'")


@_journal_verifier_assertions("session_verify_agent.run_agent")
async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    """Custom-agent entry point.  Returns ``{"driver_events": [...], **counters}``.

    ``tito_model`` must be present in ``metadata`` (the ``generate`` wrapper
    below injects it from ``args.tito_model``).  The driver schedule is derived
    from that family's ``FixedTemplate.allowed_append_roles``.
    ``prompt`` is ignored — the driver synthesizes its own initial conversation
    from ``build_initial_messages`` so runs are reproducible.
    """
    tito_model = metadata.get("tito_model")
    if tito_model is None:
        raise ValueError("session_verify_agent.run_agent requires tito_model in metadata")
    allowed_roles = fixed_template_append_roles(tito_model)
    cycles = metadata.get("session_verify_cycles", DEFAULT_CYCLES)
    schedule = select_schedule(allowed_roles, cycles=cycles)

    failure_mode = ToolCallFailureMode(metadata.get("tool_call_failure_mode", DEFAULT_TOOL_CALL_FAILURE_MODE))
    # APPEND_USER injects a user message, so the fixed template must support it.
    # Refuse up front instead of silently downgrading.
    if failure_mode is ToolCallFailureMode.APPEND_USER and "user" not in allowed_roles:
        raise ValueError(
            f"tool_call_failure_mode=APPEND_USER requires the {tito_model!r} fixed template "
            f"to support 'user', got {sorted(allowed_roles)}. Pick ROLLBACK (universal) or APPEND_TOOL "
            "(lenient-template) for tool-only surfaces."
        )

    rk = {k: v for k, v in request_kwargs.items() if k not in ("tools",)}
    messages = build_initial_messages()
    events: list[str] = []
    counters = {
        "rollback_count": 0,
        "user_count": 0,
        "system_count": 0,
        "assistant_input_count": 0,
        "tool_result_count": 0,
        "tool_call_count": 0,
    }
    # Streak of TOOL_RESULT steps that fell into the ROLLBACK fallback without
    # the model recovering to a real tool_call.  Reset on any successful
    # tool_call; gated by MAX_CONSECUTIVE_TOOL_CALL_FAILURE_ROLLBACKS to keep
    # silently-stuck samples from burning wall-time.
    consecutive_failure_rollbacks = 0

    async with httpx.AsyncClient(timeout=180) as client:
        # Initial completion — no driver action yet.
        resp = await _chat_complete(client, base_url, messages, rk, label="Initial")
        assistant = resp["choices"][0]["message"]
        messages.append(assistant)
        events.append("initial")
        counters["tool_call_count"] += len(assistant.get("tool_calls") or [])

        for step_idx, action in enumerate(schedule):
            label = f"Step {step_idx + 1} {action.value}"

            if action is DriverAction.TOOL_RESULT:
                tool_calls = assistant.get("tool_calls") or []
                if tool_calls:
                    consecutive_failure_rollbacks = 0
                    for i, tc in enumerate(tool_calls):
                        result_idx = (counters["tool_result_count"] + i) % len(MOCK_TOOL_RESULTS)
                        messages.append(
                            {
                                "role": "tool",
                                "content": MOCK_TOOL_RESULTS[result_idx],
                                "tool_call_id": tc["id"],
                            }
                        )
                    counters["tool_result_count"] += len(tool_calls)
                    events.append("append_tool")
                else:
                    # Model emitted no tool_calls — apply the configured fallback.
                    # Templates differ on what role may follow a "no tool_calls"
                    # assistant:
                    #   - GLM / Nemotron (lenient): a tool message is fine -> APPEND_TOOL.
                    #   - Kimi: a tool message must carry the id from a valid
                    #     tool_call, which we don't have -> APPEND_TOOL not OK.
                    #   - MiniMax: a tool message must follow an assistant with
                    #     non-empty tool_calls -> APPEND_TOOL not OK.
                    # If APPEND_TOOL not ok, use APPEND_USER as instead.
                    match failure_mode:
                        case ToolCallFailureMode.APPEND_TOOL:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": "none",
                                    "content": TOOL_CALL_PARSE_FAILURE_TEXT,
                                }
                            )
                            events.append("tool_call_failure_append_tool")
                        case ToolCallFailureMode.APPEND_USER:
                            messages.append({"role": "user", "content": TOOL_CALL_PARSE_FAILURE_TEXT})
                            counters["user_count"] += 1
                            events.append("tool_call_failure_append_user")
                        case ToolCallFailureMode.ROLLBACK:
                            # Same as the schedule's ROLLBACK
                            assert messages and messages[-1]["role"] == "assistant", (
                                f"tool_call_failure_mode=ROLLBACK: tail role is "
                                f"{messages[-1]['role'] if messages else 'EMPTY'}, expected assistant"
                            )
                            consecutive_failure_rollbacks += 1
                            if consecutive_failure_rollbacks > MAX_CONSECUTIVE_TOOL_CALL_FAILURE_ROLLBACKS:
                                raise AssertionError(
                                    f"ROLLBACK fallback hit {consecutive_failure_rollbacks} consecutive "
                                    f"tool_call failures (limit={MAX_CONSECUTIVE_TOOL_CALL_FAILURE_ROLLBACKS}). "
                                    "Model is not tool-calling on this prompt — check sampling temperature, "
                                    "the tool spec, or switch tool_call_failure_mode to APPEND_TOOL/APPEND_USER "
                                    "if sentinel-driven retry is preferred."
                                )
                            messages.pop()
                            counters["rollback_count"] += 1
                            events.append("tool_call_failure_rollback")
                        case _:
                            raise AssertionError(f"Unknown ToolCallFailureMode {failure_mode!r}")

            elif action is DriverAction.USER_FOLLOWUP:
                messages.append({"role": "user", "content": USER_FOLLOWUP_TEXT})
                counters["user_count"] += 1
                events.append("append_user")

            elif action is DriverAction.SYSTEM_REMINDER:
                messages.append({"role": "system", "content": SYSTEM_REMINDER_TEXT})
                counters["system_count"] += 1
                events.append("append_system")

            elif action is DriverAction.ASSISTANT_INPUT:
                messages.extend({"role": "assistant", "content": text} for text in ASSISTANT_INPUT_TEXTS)
                messages.append({"role": "user", "content": ASSISTANT_INPUT_FOLLOWUP_TEXT})
                counters["assistant_input_count"] += len(ASSISTANT_INPUT_TEXTS)
                counters["user_count"] += 1
                events.append("append_assistant")

            elif action is DriverAction.ROLLBACK:
                # Pop the last assistant from our local copy.  The next
                # request therefore has one fewer message than what the
                # server has stored, which is the trigger for its
                # ``_detect_and_rollback`` path — the server rewinds its
                # state, then re-inferences.
                if not messages or messages[-1]["role"] != "assistant":
                    raise AssertionError(
                        f"Cannot rollback at step {step_idx}: tail role is "
                        f"{messages[-1]['role'] if messages else 'EMPTY'}, expected assistant"
                    )
                messages.pop()
                counters["rollback_count"] += 1
                events.append("rollback")

            elif action is DriverAction.FORCE_FINAL:
                messages.append({"role": "user", "content": FORCE_FINAL_TEXT})
                events.append("force_final")

            else:
                raise AssertionError(f"Unknown DriverAction {action!r}")

            resp = await _chat_complete(client, base_url, messages, rk, label=label)
            assistant = resp["choices"][0]["message"]
            messages.append(assistant)
            counters["tool_call_count"] += len(assistant.get("tool_calls") or [])

    logger.info("Agent done: events=%s counters=%s", events, counters)

    return {"driver_events": events, **counters}


def _verify_tito_samples(samples, events_per_sample, *, allowed_roles) -> None:
    """Record every sample and defer hard failures to the run-level gate.

    The rollout loop treats custom-generate exceptions as retryable sample
    failures.  With a metrics sidecar, hard TITO mismatches must therefore be
    persisted instead of raised here and discarded with the sample.  Direct
    callers without a sidecar retain the immediate-failure behavior.
    """
    metrics_path = os.environ.get("MILES_SESSION_VERIFY_METRICS_PATH")
    for i, sample in enumerate(samples):
        mismatches = sample.metadata.get("tito_session_mismatch")
        if mismatches is None:
            raise AssertionError(
                f"Session multi-role e2e: sample {i} has no tito_session_mismatch "
                f"in metadata.  The session-server's compute_session_mismatch raised "
                f"TokenizationError (sessions.py:83 swallows it) — this always "
                f"indicates a TITO subclass / setup bug, not a real PASS."
            )
        forbidden = [m for m in mismatches if m.get("type") in _FORBIDDEN_MISMATCH_TYPES]
        assistant_mismatches = [m for m in mismatches if m.get("type") == "assistant_text"]
        if metrics_path:
            had_assistant_mismatch = bool(assistant_mismatches)
            assistant_example = None
            if assistant_mismatches:
                first = assistant_mismatches[0]
                assistant_example = {
                    "segment_index": first.get("segment_index"),
                    "expected_text": (first.get("expected_text") or "")[:300],
                    "actual_text": (first.get("actual_text") or "")[:300],
                }
            hard_example = None
            if forbidden:
                first = forbidden[0]
                hard_example = {
                    "type": first.get("type"),
                    "segment_index": first.get("segment_index"),
                    "detail": str(first.get("detail") or "")[:500],
                }
            entry = {
                "sample_index": i,
                "driver_events": events_per_sample[i],
                "had_assistant_mismatch": had_assistant_mismatch,
                "total_mismatches": len(mismatches),
                "assistant_mismatch_count": len(assistant_mismatches),
                "assistant_mismatch_example": assistant_example,
                "hard_mismatch_count": len(forbidden),
                "hard_mismatch_types": sorted({m.get("type") for m in forbidden}),
                "hard_mismatch_example": hard_example,
            }
            _append_session_verify_record(entry)
        elif forbidden:
            raise AssertionError(
                f"Session multi-role e2e: sample {i} has forbidden mismatches "
                f"{forbidden}. allowed_roles={allowed_roles}.  These types must be 0 "
                f"for any TITO-correct setup."
            )


@_journal_verifier_assertions("session_verify_agent.generate")
async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Custom-generate wrapper that asserts driver-action coverage.

    - Per-sample: every sample must contain ``rollback``, plus ``append_user``
      / ``append_system`` / ``append_assistant`` when those roles are allowed.
    - Cross-sample: at least one sample must contain ``append_tool``
      (model-dependent on emitting a tool_call).
    """
    tito_model = input.args.tito_model
    allowed_roles = list(fixed_template_append_roles(tito_model))
    cycles = getattr(input.args, "session_verify_cycles", DEFAULT_CYCLES)
    failure_mode = getattr(input.args, "tool_call_failure_mode", DEFAULT_TOOL_CALL_FAILURE_MODE)
    # Sample.metadata is mutable even when the outer dataclass is frozen.
    input.sample.metadata["tito_model"] = tito_model
    input.sample.metadata["session_verify_cycles"] = cycles
    input.sample.metadata["tool_call_failure_mode"] = failure_mode

    output = await _base_generate(input)

    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    events_per_sample = [s.metadata.get("driver_events", []) for s in samples]
    metrics_path = os.environ.get("MILES_SESSION_VERIFY_METRICS_PATH")

    if not samples:
        raise AssertionError("Session multi-role e2e: generate returned no samples")
    _verify_tito_samples(samples, events_per_sample, allowed_roles=allowed_roles)

    required_per_sample = ["rollback"]
    if "user" in allowed_roles:
        required_per_sample.append("append_user")
    if "system" in allowed_roles:
        required_per_sample.append("append_system")
    if "assistant" in allowed_roles:
        required_per_sample.append("append_assistant")

    for i, events in enumerate(events_per_sample):
        missing = [req for req in required_per_sample if req not in events]
        if missing:
            raise AssertionError(
                f"Session multi-role e2e: sample {i} missing required driver events "
                f"{missing}. allowed_roles={allowed_roles}, events={events}"
            )

    if not metrics_path and not any("append_tool" in events for events in events_per_sample):
        raise AssertionError(
            "Session multi-role e2e: no sample produced an append_tool action — "
            f"the model may not be tool-calling.  events_per_sample={events_per_sample}"
        )

    logger.info(
        "Multi-role coverage verified: per_sample=%s, samples=%d, events=%s",
        required_per_sample,
        len(samples),
        events_per_sample,
    )
    return output


def _add_arguments(parser):
    _base_generate.add_arguments(parser)
    parser.add_argument(
        "--session-verify-cycles",
        type=int,
        default=DEFAULT_CYCLES,
        help="Number of driver schedule cycles per sample for session-server "
        "TITO verification.  Each cycle exercises recurrent role actions plus "
        "a rollback; assistant input is exercised once as the terminal action. "
        "More cycles stress the TITO accumulator longer but expand context "
        "length.  Drop to 2 on tighter-context models (e.g. Qwen3 32K with 4K "
        "response budget).",
    )
    parser.add_argument(
        "--tool-call-failure-mode",
        type=str,
        default=DEFAULT_TOOL_CALL_FAILURE_MODE.value,
        choices=[m.value for m in ToolCallFailureMode],
        help="Recovery mode when a TOOL_RESULT step sees no tool_calls on the "
        "assistant.  'rollback' (default, universal) pops the assistant and "
        "re-inferences.  'append_tool' splices a sentinel tool message (only "
        "works on lenient templates).  'append_user' splices a user message "
        "with the same failure text — requires the fixed template to support 'user'.",
    )


generate.add_arguments = _add_arguments
