"""Custom-generate / custom-agent driver for TITO session-server verification.

Wired through ``--custom-generate-function-path`` /
``--custom-agent-function-path``; consumed by
``tests/e2e/sglang/test_session_server_multi_role/`` (one test file per
model family) and ``scripts/tools/verify_session_tito_tokenizer.py``.

Each sample runs one of two rollouts, named by the ``session_verify_kind`` of
its prompt record:

- ``normal`` answers a tool call with a ``tool`` message and an answer with
  the next ``user`` question, the two continuations production harnesses send
  most.  CI monitors its mismatch rates.
- ``spec`` walks the family's continuation roles
  (``FixedTemplate.roles_after_tool_call`` / ``roles_after_no_tool_call``):
  every continuation the spec refuses must get HTTP 400 and every one it
  accepts must succeed.  It also rolls back, appends a system message and
  injects assistant messages, which the session server must handle too.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import wraps

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

import httpx

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.agentic_tool_call import generate as _base_generate
from miles.utils.chat_template_utils.tito_tokenizer import VALID_APPEND_ROLES, FixedTemplate, TITOTokenizerType
from miles.utils.test_utils.openai_stream_client import stream_chat_completions

logger = logging.getLogger(__name__)


def _append_session_verify_record(entry: dict) -> bool:
    metrics_path = os.environ.get("MILES_SESSION_VERIFY_METRICS_PATH")
    if not metrics_path:
        return False
    payload = (json.dumps(entry) + "\n").encode()
    fd = os.open(
        metrics_path,
        os.O_WRONLY
        | os.O_APPEND
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0),  # config-access-exempt: os.O_CLOEXEC is platform-dependent
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


class SessionVerifyKind(StrEnum):
    NORMAL = "normal"
    SPEC = "spec"


# Prompt-record metadata key that names a sample's rollout.
SESSION_VERIFY_KIND_KEY = "session_verify_kind"
# Agent metadata key carrying a spec rollout's expected and actual statuses.
EXPECTATIONS_KEY = "session_verify_expectations"

# Mismatch tiers reported by the session-server's per-sample comparator.  Any
# occurrence of these "hard" types in a sample's tito_session_mismatch
# indicates a TITO bug and fails the verifier run.  The soft `assistant_text`
# tier is excluded: the normal rollout gates it by a ratio threshold, and the
# spec rollout does not gate it.
_FORBIDDEN_MISMATCH_TYPES: frozenset[str] = frozenset(
    {"special_token_count", "special_token_type", "non_assistant_text"}
)

# Override per call: ``--session-verify-cycles N`` (CLI) or ``cycles=N``
# (pytest via ``run_session_verify``).  Smaller-context models with a 4K
# response budget should drop to 2 to avoid context overflow.
DEFAULT_CYCLES = 3
# Resamples of a turn cut by finish_reason='length', or of a turn that must call a tool and did not.
_MAX_RESAMPLES = 2
_RETRY_SEED_STRIDE = 1_000_000
# Tool rounds the agent answers before it needs a reply without tool calls.
_MAX_TOOL_ROUNDS = 3
_ACCEPTED = 200
_REFUSED = 400


def fixed_template(tito_model: TITOTokenizerType | str) -> FixedTemplate:
    """The selected family's fixed-template contract."""
    return TITOTokenizerType.get_tokenizer_class(TITOTokenizerType(tito_model)).FIXED_TEMPLATE


def fixed_template_append_roles(tito_model: TITOTokenizerType | str) -> tuple[str, ...]:
    """Return the selected family's fixed append capability in canonical order."""
    supported = fixed_template(tito_model).allowed_append_roles
    return tuple(role for role in VALID_APPEND_ROLES if role in supported)


# English-only on purpose: matches the production agentic flows tokenization
# and tool-call parsing are tuned against.
USER_FOLLOWUP_TEXT = "Now check the weather in Shanghai."
USER_FOLLOWUP_TEXTS = (
    USER_FOLLOWUP_TEXT,
    "Now check the weather in Hangzhou.",
    "Now check the weather in London.",
)
# A prompt that asks for an answer without a tool, so the reply carries no tool call.
NO_TOOL_CALL_USER_TEXT = "Without using any tool, tell me one sentence about Beijing."
# tool_choice "none" keeps that reply free of parsed tool calls whatever the model writes,
# and kept special tokens let the server re-render a call the model writes into content.
NO_TOOL_CALL_REQUEST_ARGS = {"tool_choice": "none", "skip_special_tokens": False}
# A neutral result for a tool message that answers no call; a retry instruction makes some models loop.
SENTINEL_TOOL_RESULT = '{"result": "unavailable"}'
SYSTEM_REMINDER_TEXT = "Note: from now on, answer in a single sentence; skip all pleasantries."
ASSISTANT_INPUT_TEXTS = (
    "The earlier Beijing weather result was 22 degrees Celsius and sunny.",
    "The earlier Shanghai weather result was 30 degrees Celsius and rainy.",
)
ASSISTANT_INPUT_FOLLOWUP_TEXT = "Summarize the two weather results above in one sentence without calling a tool."

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

# Roles the spec rollout appends directly after a tool call, in this order.
_PROBE_ROLES_AFTER_TOOL_CALL = ("user", "system", "assistant")
_PROBE_MESSAGES = {
    "user": {"role": "user", "content": USER_FOLLOWUP_TEXT},
    "system": {"role": "system", "content": SYSTEM_REMINDER_TEXT},
    "assistant": {"role": "assistant", "content": ASSISTANT_INPUT_TEXTS[0]},
}
_SENTINEL_TOOL_MESSAGE = {"role": "tool", "content": SENTINEL_TOOL_RESULT, "tool_call_id": "none"}


def spec_probe_steps(fixed: FixedTemplate) -> list[str]:
    """The probes a spec rollout sends for *fixed*, in order; each records one expectation."""
    roles = [role for role in _PROBE_ROLES_AFTER_TOOL_CALL if role in fixed.allowed_append_roles]
    return [f"{role} after tool call" for role in roles] + ["tool after no tool call"]


def build_initial_messages() -> list[dict]:
    """The fixed (system, user) prompt both rollouts start from."""
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


async def _chat_status(client, base_url, messages, request_kwargs, *, label) -> tuple[int, dict | None]:
    """Send a request the server may refuse; return its status and, on 200, the response.

    Non-streaming, so a 400 reaches the caller as a status instead of a broken stream.
    """
    payload = {"messages": messages, "tools": TOOLS, **request_kwargs, "stream": False}
    resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
    if resp.status_code == _ACCEPTED:
        return _ACCEPTED, resp.json()
    assert resp.status_code == _REFUSED, f"{label} failed ({resp.status_code}): {resp.text}"
    logger.info("%s refused (%d): %s", label, resp.status_code, resp.text[:300])
    return _REFUSED, None


def _with_retry_seed(request_kwargs: dict, retry: int) -> dict:
    if not retry or request_kwargs.get("seed") is None:
        return request_kwargs
    return {**request_kwargs, "seed": request_kwargs["seed"] + retry * _RETRY_SEED_STRIDE}


@dataclass
class _Driver:
    """One trajectory's client, events, counters and probe expectations.

    A resample re-sends the same messages, so the server rolls back the reply it replaces.
    """

    client: httpx.AsyncClient
    base_url: str
    request_kwargs: dict
    events: list[str] = field(default_factory=list)
    expectations: list[dict] = field(default_factory=list)
    counters: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(
            (
                "rollback_count",
                "user_count",
                "system_count",
                "assistant_input_count",
                "tool_result_count",
                "tool_call_count",
            ),
            0,
        )
    )

    async def chat(self, messages, *, label, extra_kwargs=None, need_tool_call=False) -> dict:
        """Generate the next assistant message, resampling a length cut or, if asked, a reply without calls."""
        kwargs = {**self.request_kwargs, **(extra_kwargs or {})}
        for retry in range(_MAX_RESAMPLES + 1):
            response = await _chat(self.client, self.base_url, messages, _with_retry_seed(kwargs, retry), label=label)
            choice = response["choices"][0]
            if choice.get("finish_reason") == "length":
                continue
            if need_tool_call and not choice["message"].get("tool_calls"):
                continue
            return self._accept(choice["message"])
        raise AssertionError(
            f"{label} exhausted {_MAX_RESAMPLES} resamples (finish_reason='length' or no tool call when one is needed)"
        )

    async def probe(self, messages, *, label, expected_status) -> dict | None:
        """Send a continuation the spec decides; record its status and return the reply if it was accepted."""
        for retry in range(_MAX_RESAMPLES + 1):
            kwargs = _with_retry_seed(self.request_kwargs, retry)
            status, response = await _chat_status(self.client, self.base_url, messages, kwargs, label=label)
            if status == _REFUSED or response["choices"][0].get("finish_reason") != "length":
                break
        else:
            raise AssertionError(f"{label} exhausted {_MAX_RESAMPLES} resamples after finish_reason='length'")
        self.expectations.append({"step": label, "expected_status": expected_status, "actual_status": status})
        return None if status == _REFUSED else self._accept(response["choices"][0]["message"])

    def _accept(self, message: dict) -> dict:
        self.counters["tool_call_count"] += len(message.get("tool_calls") or [])
        return message

    def append_tool_results(self, messages: list[dict], assistant: dict) -> None:
        """Append *assistant* and answer each of its tool calls with the next mock tool result."""
        messages.append(assistant)
        for i, call in enumerate(assistant["tool_calls"]):
            result = MOCK_TOOL_RESULTS[(self.counters["tool_result_count"] + i) % len(MOCK_TOOL_RESULTS)]
            messages.append({"role": "tool", "content": result, "tool_call_id": call["id"]})
        self.counters["tool_result_count"] += len(assistant["tool_calls"])
        self.events.append("append_tool")

    def append_user(self, messages: list[dict], assistant: dict, text: str) -> None:
        messages.extend([assistant, {"role": "user", "content": text}])
        self.counters["user_count"] += 1
        self.events.append("append_user")

    def next_city_question(self) -> str:
        return USER_FOLLOWUP_TEXTS[self.counters["user_count"] % len(USER_FOLLOWUP_TEXTS)]

    async def settle(self, messages: list[dict], reply: dict) -> dict:
        """Answer *reply*'s tool calls until the model answers without one."""
        for round_index in range(_MAX_TOOL_ROUNDS):
            if not reply.get("tool_calls"):
                return reply
            self.append_tool_results(messages, reply)
            reply = await self.chat(messages, label=f"tool round {round_index + 1}")
        assert not reply.get("tool_calls"), f"the model still calls a tool after {_MAX_TOOL_ROUNDS} answered rounds"
        return reply


async def _run_normal(driver: _Driver, *, cycles: int) -> None:
    """Tool answer after a tool call, next user question after an answer; nothing else."""
    messages = build_initial_messages()
    reply = await driver.chat(messages, label="Initial")
    driver.events.append("initial")
    # Each cycle is two continuations: with a tool-calling model, one tool answer and one question.
    for turn in range(2 * cycles):
        if reply.get("tool_calls"):
            driver.append_tool_results(messages, reply)
        else:
            driver.append_user(messages, reply, driver.next_city_question())
        reply = await driver.chat(messages, label=f"Turn {turn + 1}")


async def _run_spec(driver: _Driver, fixed: FixedTemplate) -> None:
    """Probe every continuation the spec decides, then roll back, append system and inject assistants."""
    messages = build_initial_messages()
    call = await driver.chat(messages, label="Initial", need_tool_call=True)
    driver.events.append("initial")
    steps = iter(spec_probe_steps(fixed))

    for role in _PROBE_ROLES_AFTER_TOOL_CALL:
        if role not in fixed.allowed_append_roles:
            continue
        probe = _PROBE_MESSAGES[role]
        expected = _ACCEPTED if role in fixed.roles_after_tool_call else _REFUSED
        reply = await driver.probe([*messages, call, probe], label=next(steps), expected_status=expected)
        driver.events.append(f"probe_{role}_after_tool_call")
        if reply is None:
            continue  # refused: the session still ends at `call`
        messages.extend([call, probe])
        if not reply.get("tool_calls"):
            driver.append_user(messages, reply, driver.next_city_question())
            reply = await driver.chat(messages, label=f"Tool call after {role}", need_tool_call=True)
        call = reply

    driver.append_tool_results(messages, call)
    reply = await driver.settle(messages, await driver.chat(messages, label="Tool result"))

    driver.append_user(messages, reply, NO_TOOL_CALL_USER_TEXT)
    answer = await driver.chat(messages, label="No tool call", extra_kwargs=NO_TOOL_CALL_REQUEST_ARGS)
    assert not answer.get("tool_calls"), f"tool_choice='none' still returned tool_calls={answer['tool_calls']}"
    driver.events.append("no_tool_call")
    expected = _ACCEPTED if "tool" in fixed.roles_after_no_tool_call else _REFUSED
    reply = await driver.probe(
        [*messages, answer, _SENTINEL_TOOL_MESSAGE], label=next(steps), expected_status=expected
    )
    driver.events.append("probe_tool_after_no_tool_call")
    if reply is None:
        driver.append_user(messages, answer, driver.next_city_question())
        reply = await driver.chat(messages, label="User after refused tool")
    else:
        messages.extend([answer, _SENTINEL_TOOL_MESSAGE])

    # Re-sending the request that produced `reply` makes the server roll it back and
    # regenerate; doing it twice covers a rollback right after a rollback.
    for _ in range(2):
        reply = await driver.chat(messages, label="Rollback")
        driver.counters["rollback_count"] += 1
        driver.events.append("rollback")

    if "system" in fixed.roles_after_no_tool_call:
        reply = await driver.settle(messages, reply)
        messages.extend([reply, {"role": "system", "content": SYSTEM_REMINDER_TEXT}])
        driver.counters["system_count"] += 1
        driver.events.append("append_system")
        reply = await driver.chat(messages, label="System message")

    # Injected assistants come last: they are prompt messages, not generated
    # checkpoints, so rolling back across them is a separate contract.
    if "assistant" in fixed.roles_after_no_tool_call:
        reply = await driver.settle(messages, reply)
        messages.extend([reply, *({"role": "assistant", "content": text} for text in ASSISTANT_INPUT_TEXTS)])
        messages.append({"role": "user", "content": ASSISTANT_INPUT_FOLLOWUP_TEXT})
        driver.counters["assistant_input_count"] += len(ASSISTANT_INPUT_TEXTS)
        driver.counters["user_count"] += 1
        driver.events.append("append_assistant")
        await driver.chat(messages, label="Assistant input")


def _assert_spec_rollout_supported(tito_model: str, fixed: FixedTemplate) -> None:
    # The spec rollout returns to a tool call with a user question after an answer.
    if "user" not in fixed.roles_after_no_tool_call:
        raise ValueError(
            f"the spec rollout needs 'user' after an assistant without tool calls; "
            f"{tito_model!r} accepts only {sorted(fixed.roles_after_no_tool_call)} there"
        )


def _session_verify_kind(metadata: dict) -> SessionVerifyKind:
    if SESSION_VERIFY_KIND_KEY not in metadata:
        raise ValueError(f"session verification needs {SESSION_VERIFY_KIND_KEY!r} in the sample metadata")
    return SessionVerifyKind(metadata[SESSION_VERIFY_KIND_KEY])


@_journal_verifier_assertions("session_verify_agent.run_agent")
async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    """Custom-agent entry point.  Returns ``{"driver_events": [...], EXPECTATIONS_KEY: [...], **counters}``.

    ``metadata`` must carry ``tito_model`` (the ``generate`` wrapper below
    injects it from ``args.tito_model``) and ``session_verify_kind`` (from the
    prompt record).  ``prompt`` is ignored: both rollouts start from
    ``build_initial_messages`` so runs are reproducible.
    """
    tito_model = metadata.get("tito_model")
    if tito_model is None:
        raise ValueError("session_verify_agent.run_agent requires tito_model in metadata")
    kind = _session_verify_kind(metadata)
    fixed = fixed_template(tito_model)
    if kind is SessionVerifyKind.SPEC:
        _assert_spec_rollout_supported(tito_model, fixed)

    rk = {k: v for k, v in request_kwargs.items() if k not in ("tools",)}
    async with httpx.AsyncClient(timeout=180) as client:
        driver = _Driver(client=client, base_url=base_url, request_kwargs=rk)
        if kind is SessionVerifyKind.SPEC:
            await _run_spec(driver, fixed)
        else:
            await _run_normal(driver, cycles=metadata.get("session_verify_cycles", DEFAULT_CYCLES))

    logger.info("Agent done: kind=%s events=%s expectations=%s", kind, driver.events, driver.expectations)
    return {"driver_events": driver.events, EXPECTATIONS_KEY: driver.expectations, **driver.counters}


def _verify_tito_samples(samples, *, kind: SessionVerifyKind) -> None:
    """Record every sample and defer hard failures to the run-level gate.

    The rollout loop treats custom-generate exceptions as retryable sample
    failures.  With a metrics sidecar, hard TITO mismatches and unmet spec
    expectations must therefore be persisted instead of raised here and
    discarded with the sample.  Direct callers without a sidecar retain the
    immediate-failure behavior.
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
        if forbidden:
            logger.error("Hard TITO mismatch: leaf=%r mismatches=%r", sample.metadata.get("leaf"), forbidden)
        expectations = sample.metadata.get(EXPECTATIONS_KEY) or []
        unmet = [e for e in expectations if e["actual_status"] != e["expected_status"]]
        if metrics_path:
            _append_session_verify_record(_sample_record(i, sample, kind, mismatches, forbidden, expectations))
        elif forbidden or unmet:
            raise AssertionError(
                f"Session multi-role e2e: {kind} sample {i} has forbidden mismatches {forbidden} "
                f"and unmet spec expectations {unmet}.  Both must be empty for any TITO-correct setup."
            )


def _sample_record(index, sample, kind, mismatches, forbidden, expectations) -> dict:
    assistant_mismatches = [m for m in mismatches if m.get("type") == "assistant_text"]
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
    return {
        "sample_index": index,
        "kind": str(kind),
        "driver_events": sample.metadata.get("driver_events", []),
        "expectations": expectations,
        "had_assistant_mismatch": bool(assistant_mismatches),
        "total_mismatches": len(mismatches),
        "assistant_mismatch_count": len(assistant_mismatches),
        "assistant_mismatch_example": assistant_example,
        "hard_mismatch_count": len(forbidden),
        "hard_mismatch_types": sorted({m.get("type") for m in forbidden}),
        "hard_mismatch_example": hard_example,
    }


@_journal_verifier_assertions("session_verify_agent.generate")
async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Custom-generate wrapper that records each sample and checks rollout coverage.

    - Spec samples must carry one expectation per probe of the family, in order.
    - Without a metrics sidecar, at least one normal sample must have answered
      a tool call (with one, the runner checks this across the run).
    """
    tito_model = input.args.tito_model
    kind = _session_verify_kind(input.sample.metadata)
    cycles = getattr(
        input.args, "session_verify_cycles", DEFAULT_CYCLES
    )  # config-access-exempt: optional custom session-verification hook argument
    # Sample.metadata is mutable even when the outer dataclass is frozen.
    input.sample.metadata["tito_model"] = tito_model
    input.sample.metadata["session_verify_cycles"] = cycles

    output = await _base_generate(input)

    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    if not samples:
        raise AssertionError("Session multi-role e2e: generate returned no samples")
    _verify_tito_samples(samples, kind=kind)

    if kind is SessionVerifyKind.SPEC:
        steps = spec_probe_steps(fixed_template(tito_model))
        for i, sample in enumerate(samples):
            recorded = [e["step"] for e in sample.metadata.get(EXPECTATIONS_KEY) or []]
            if recorded != steps:
                raise AssertionError(
                    f"Session multi-role e2e: spec sample {i} recorded probes {recorded}, expected {steps}"
                )
    elif not os.environ.get("MILES_SESSION_VERIFY_METRICS_PATH") and not any(
        "append_tool" in sample.metadata.get("driver_events", []) for sample in samples
    ):
        raise AssertionError(
            "Session multi-role e2e: no sample produced an append_tool action — "
            f"the model may not be tool-calling.  events={[s.metadata.get('driver_events') for s in samples]}"
        )

    logger.info("Session verification %s rollout recorded %d samples", kind, len(samples))
    return output


def _add_arguments(parser):
    _base_generate.add_arguments(parser)
    parser.add_argument(
        "--session-verify-cycles",
        type=int,
        default=DEFAULT_CYCLES,
        help="Length of the normal session-verification rollout: each cycle is two "
        "continuations (a tool answer after a tool call, a user question after an "
        "answer).  More cycles stress the TITO accumulator longer but expand context "
        "length.  Drop to 2 on tighter-context models (e.g. Qwen3 32K with 4K "
        "response budget).",
    )


generate.add_arguments = _add_arguments
