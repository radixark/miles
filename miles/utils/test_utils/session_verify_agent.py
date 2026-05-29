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
from enum import Enum, StrEnum

import httpx

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.agentic_tool_call import generate as _base_generate

logger = logging.getLogger(__name__)


class DriverAction(Enum):
    TOOL_RESULT = "tool_result"
    USER_FOLLOWUP = "user_followup"
    SYSTEM_REMINDER = "system_reminder"
    ROLLBACK = "rollback"
    FORCE_FINAL = "force_final"


_T = DriverAction.TOOL_RESULT
_U = DriverAction.USER_FOLLOWUP
_S = DriverAction.SYSTEM_REMINDER
_R = DriverAction.ROLLBACK
_F = DriverAction.FORCE_FINAL


class ToolCallFailureMode(StrEnum):
    """Recovery strategy when a TOOL_RESULT step finds the assistant emitted no tool_calls.

    APPEND_TOOL  : Splice a sentinel ``tool`` message and continue.  Works on
                   lenient templates; strict templates that hard-assert any
                   ``tool`` role must follow an assistant with ``tool_calls``
                   (e.g. MiniMax-M2.7) will reject the next request at server-side.
    APPEND_USER  : Splice a ``user`` message carrying the same failure text as
                   APPEND_TOOL.  Requires "user" in ``allowed_append_roles`` —
                   raises ValueError at agent start otherwise, so misconfig is
                   immediately visible instead of silently downgrading.
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
# tito_session_mismatch indicates a TITO bug and fails the sample.  The
# soft `assistant_text` tier is excluded — it is aggregated across samples
# and gated by a ratio threshold instead.
_FORBIDDEN_MISMATCH_TYPES: frozenset[str] = frozenset(
    {"special_token_count", "special_token_type", "non_assistant_text"}
)

# Override per call: ``--session-verify-cycles N`` (CLI) or ``cycles=N``
# (pytest via ``run_session_verify``).  Smaller-context models with a 4K
# response budget should drop to 2 to avoid context overflow.
DEFAULT_CYCLES = 3

_SUPPORTED_ROLE_SURFACES: tuple[frozenset[str], ...] = (
    frozenset({"tool"}),
    frozenset({"tool", "user"}),
    frozenset({"tool", "user", "system"}),
)


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
    """Pick the schedule for ``frozenset(allowed_roles)``; raises on unregistered."""
    key = frozenset(allowed_roles)
    if key not in _SUPPORTED_ROLE_SURFACES:
        registered = sorted(sorted(k) for k in _SUPPORTED_ROLE_SURFACES)
        raise ValueError(f"No schedule registered for allowed_roles={sorted(key)}. Registered: {registered}")
    if cycles < 1:
        raise ValueError(f"cycles must be >= 1, got {cycles}")
    cycle = _build_cycle(key)
    # Extra R after the first cycle exercises consecutive-rollback adjacency,
    # which cycle-repeat alone never produces.
    schedule = list(cycle) + [_R] + cycle * (cycles - 1)
    if "user" in key:
        schedule.append(_F)
    return schedule


def build_initial_messages() -> list[dict]:
    """The fixed (system, user) prompt all schedules start from."""
    return [
        {"role": "system", "content": INITIAL_SYSTEM_PROMPT},
        {"role": "user", "content": INITIAL_USER_PROMPT},
    ]


async def _chat(client, base_url, messages, request_kwargs, *, label):
    payload = {"messages": messages, "tools": TOOLS, **request_kwargs}
    resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
    assert resp.status_code == 200, f"{label} failed ({resp.status_code}): {resp.text}"
    return resp.json()


async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    """Custom-agent entry point.  Returns ``{"driver_events": [...], **counters}``.

    ``allowed_append_roles`` must be present in ``metadata`` (the ``generate``
    wrapper below injects it from ``args.tito_allowed_append_roles``).
    ``prompt`` is ignored — the driver synthesizes its own initial conversation
    from ``build_initial_messages`` so runs are reproducible.
    """
    allowed_roles = metadata.get("allowed_append_roles")
    if allowed_roles is None:
        raise ValueError(
            "session_verify_agent.run_agent requires allowed_append_roles in metadata; "
            "the generate wrapper should inject it from args.tito_allowed_append_roles."
        )
    cycles = metadata.get("session_verify_cycles", DEFAULT_CYCLES)
    schedule = select_schedule(allowed_roles, cycles=cycles)

    failure_mode = ToolCallFailureMode(metadata.get("tool_call_failure_mode", DEFAULT_TOOL_CALL_FAILURE_MODE))
    # APPEND_USER injects a user message — only valid if 'user' is in
    # allowed_append_roles.  Refuse up front instead of silently downgrading.
    if failure_mode is ToolCallFailureMode.APPEND_USER and "user" not in allowed_roles:
        raise ValueError(
            f"tool_call_failure_mode=APPEND_USER requires 'user' in allowed_append_roles, "
            f"got {sorted(allowed_roles)}.  Pick ROLLBACK (universal) or APPEND_TOOL "
            "(lenient-template) for tool-only surfaces."
        )

    rk = {k: v for k, v in request_kwargs.items() if k not in ("tools",)}
    messages = build_initial_messages()
    events: list[str] = []
    counters = {
        "rollback_count": 0,
        "user_count": 0,
        "system_count": 0,
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
        resp = await _chat(client, base_url, messages, rk, label="Initial")
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

            resp = await _chat(client, base_url, messages, rk, label=label)
            assistant = resp["choices"][0]["message"]
            messages.append(assistant)
            counters["tool_call_count"] += len(assistant.get("tool_calls") or [])

    logger.info("Agent done: events=%s counters=%s", events, counters)

    return {"driver_events": events, **counters}


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Custom-generate wrapper that asserts driver-action coverage.

    - Per-sample: every sample must contain ``rollback``, plus ``append_user``
      / ``append_system`` when those roles are allowed.
    - Cross-sample: at least one sample must contain ``append_tool``
      (model-dependent on emitting a tool_call).
    """
    allowed_roles = list(input.args.tito_allowed_append_roles)
    cycles = getattr(input.args, "session_verify_cycles", DEFAULT_CYCLES)
    failure_mode = getattr(input.args, "tool_call_failure_mode", DEFAULT_TOOL_CALL_FAILURE_MODE)
    # Sample.metadata is mutable even when the outer dataclass is frozen.
    input.sample.metadata["allowed_append_roles"] = allowed_roles
    input.sample.metadata["session_verify_cycles"] = cycles
    input.sample.metadata["tool_call_failure_mode"] = failure_mode

    output = await _base_generate(input)

    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    events_per_sample = [s.metadata.get("driver_events", []) for s in samples]
    metrics_path = os.environ.get("MILES_SESSION_VERIFY_METRICS_PATH")

    required_per_sample = ["rollback"]
    if "user" in allowed_roles:
        required_per_sample.append("append_user")
    if "system" in allowed_roles:
        required_per_sample.append("append_system")

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
        if forbidden:
            raise AssertionError(
                f"Session multi-role e2e: sample {i} has forbidden mismatches "
                f"{forbidden}. allowed_roles={allowed_roles}.  These types must be 0 "
                f"for any TITO-correct setup."
            )
        if metrics_path:
            assistant_mismatches = [m for m in mismatches if m.get("type") == "assistant_text"]
            had_assistant_mismatch = bool(assistant_mismatches)
            example = None
            if assistant_mismatches:
                first = assistant_mismatches[0]
                example = {
                    "segment_index": first.get("segment_index"),
                    "expected_text": (first.get("expected_text") or "")[:300],
                    "actual_text": (first.get("actual_text") or "")[:300],
                }
            with open(metrics_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "sample_index": i,
                            "driver_events": events_per_sample[i],
                            "had_assistant_mismatch": had_assistant_mismatch,
                            "total_mismatches": len(mismatches),
                            "assistant_mismatch_count": len(assistant_mismatches),
                            "assistant_mismatch_example": example,
                        }
                    )
                    + "\n"
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
        "TITO verification.  Each cycle exercises every action in the role "
        "surface plus a rollback; more cycles stress the TITO accumulator "
        "longer but expand context length.  Drop to 2 on tighter-context "
        "models (e.g. Qwen3 32K with 4K response budget).",
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
        "with the same failure text — requires 'user' in allowed_append_roles.",
    )


generate.add_arguments = _add_arguments
