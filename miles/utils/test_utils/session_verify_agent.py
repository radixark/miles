"""Custom-generate / custom-agent driver for TITO session-server verification.

Wired through ``--custom-generate-function-path`` /
``--custom-agent-function-path``; consumed by
``tests/e2e/sglang/test_session_server_multi_role.py`` and
``scripts/tools/verify_session_tito_tokenizer.py``.
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum

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
                    # Model didn't tool-call — append a sentinel tool message
                    # so a ``{tool}``-only session is not rejected by the
                    # server's append-role check.  Coverage still flags this
                    # via the missing ``append_tool`` event.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": "none",
                            "content": (
                                "Driver note: the previous assistant turn did not "
                                "emit a tool_call, so no tool was actually invoked."
                            ),
                        }
                    )
                    events.append("tool_result_skipped_no_tool_call")

            elif action is DriverAction.USER_FOLLOWUP:
                messages.append({"role": "user", "content": USER_FOLLOWUP_TEXT})
                counters["user_count"] += 1
                events.append("append_user")

            elif action is DriverAction.SYSTEM_REMINDER:
                messages.append({"role": "system", "content": SYSTEM_REMINDER_TEXT})
                counters["system_count"] += 1
                events.append("append_system")

            elif action is DriverAction.ROLLBACK:
                # Drop the last assistant; the next request's messages list
                # is shorter than the session accumulator, so the server walks
                # _detect_and_rollback before re-inferencing the assistant.
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
    # Sample.metadata is mutable even when the outer dataclass is frozen.
    input.sample.metadata["allowed_append_roles"] = allowed_roles
    input.sample.metadata["session_verify_cycles"] = cycles

    output = await _base_generate(input)

    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    events_per_sample = [s.metadata.get("driver_events", []) for s in samples]

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

    if not any("append_tool" in events for events in events_per_sample):
        raise AssertionError(
            "Session multi-role e2e: no sample produced an append_tool action — "
            f"the model may not be tool-calling.  events_per_sample={events_per_sample}"
        )

    # Token-seq comparator metrics (server-computed in sessions.py:83).  Any of
    # special_token_count / special_token_type / non_assistant_text mismatches
    # is a hard TITO bug and must be 0 per-sample; assistant_text mismatches
    # are softer (assistant tokens inherited from the pretokenized prefix may
    # not match the chat template's canonical tokenization) and are aggregated
    # across samples via a metrics file.
    forbidden_types = {"special_token_count", "special_token_type", "non_assistant_text"}
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
        forbidden = [m for m in mismatches if m.get("type") in forbidden_types]
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


generate.add_arguments = _add_arguments
