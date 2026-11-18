"""Custom-generate / custom-agent driver for TITO session-server verification.

Wired through ``--custom-generate-function-path`` /
``--custom-agent-function-path``; consumed by
``tests/e2e/sglang/test_session_server_multi_role/`` (one test file per
model family) and ``scripts/tools/verify_session_tito_tokenizer.py``.

Each sample runs one of three rollouts, named by the ``session_verify_kind`` of
its prompt record:

- ``normal`` answers a tool call with a ``tool`` message and an answer with
  the next ``user`` question, the two continuations production harnesses send
  most.  CI monitors its mismatch rates.
- ``spec`` walks the family's continuation roles
  (``FixedTemplate.roles_after_tool_call`` / ``roles_after_no_tool_call``):
  every continuation the spec refuses must get HTTP 400 and every one it
  accepts must succeed.  It also rolls back, appends a system message and
  injects assistant messages, which the session server must handle too.

- ``truncate`` generates one reply in full, regenerates it with a ``max_tokens``
  that would stop it inside its reasoning, a tool call or its answer
  (``session_verify_cut_kind`` of the prompt record), and continues the cut
  turn.  The continuation must succeed exactly when the run continues
  truncated turns (``--allow-truncated-continuation``), and then the next
  prompt must close the cut turn with the family's closer.

Some spec and truncate steps need something of the model: a tool call, no tool
call, or a reply that has the structure to cut.  A sample whose model does not
deliver is recorded as unfinished and stops at that step; the run fails only
when too many samples of a rollout are unfinished (the runner's
``UNFINISHED_RATIO_THRESHOLD``).
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
from miles.utils.test_utils.session_truncation import (
    check_continued_cuts,
    cut_length,
    load_tito_tokenizer,
    record_cut_kind,
)

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


def _journal_verification_error(stage: str, exc: AssertionError) -> bool:
    """Record a failure the run-level gate must see; False without a metrics sidecar."""
    return _append_session_verify_record(
        {"verification_error": {"stage": stage, "type": type(exc).__name__, "message": str(exc)[:1000]}}
    )


def _journal_verifier_assertions(stage: str):
    def decorate(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except AssertionError as exc:
                _journal_verification_error(stage, exc)
                raise

        return wrapped

    return decorate


class SessionVerifyKind(StrEnum):
    NORMAL = "normal"
    SPEC = "spec"
    TRUNCATE = "truncate"


# Stage name of ``generate`` in a journaled verification error.
_GENERATE_STAGE = "session_verify_agent.generate"
# Prompt-record metadata key that names a sample's rollout.
SESSION_VERIFY_KIND_KEY = "session_verify_kind"
# Prompt-record metadata key that names where a truncate sample cuts its reply (``truncation.CUT_KINDS``).
SESSION_VERIFY_CUT_KIND_KEY = "session_verify_cut_kind"
# Agent metadata key carrying a spec or truncate rollout's expected and actual statuses.
EXPECTATIONS_KEY = "session_verify_expectations"
# Agent metadata key carrying why a spec or truncate rollout stopped before its last step, or None.
UNFINISHED_KEY = "session_verify_unfinished"
# Agent metadata key carrying a truncate rollout's continued cuts (``session_truncation.check_continued_cuts``).
CUTS_KEY = "session_verify_cuts"

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
_MAX_RESAMPLES = 4
_RETRY_SEED_STRIDE = 1_000_000
# Tool rounds the agent answers before it needs a reply without tool calls.
_MAX_TOOL_ROUNDS = 3
_ACCEPTED = 200
_REFUSED = 400
# A run that does not continue truncated turns refuses the continuation with this status.
_TRUNCATED_REFUSED = 409
# The one probe of a truncate rollout.
CONTINUE_CUT_STEP = "continue cut turn"
# Where a truncate rollout cuts when the family declares no closer to place the cut by.
_BLIND_CUT_TOKENS = 8
# Regenerations a truncate rollout tries until one stops in the planned place.  Each is a new sample:
# a training session pins the temperature and refuses top_k, and SGLang honours a seed only under
# deterministic inference, so nothing makes the regeneration repeat the probed reply.
_MAX_CUT_ATTEMPTS = 4


def fixed_template(tito_model: TITOTokenizerType | str) -> FixedTemplate:
    """The selected family's fixed-template contract."""
    return TITOTokenizerType.get_tokenizer_class(TITOTokenizerType(tito_model)).FIXED_TEMPLATE


def fixed_template_append_roles(tito_model: TITOTokenizerType | str) -> tuple[str, ...]:
    """Return the selected family's fixed append capability in canonical order."""
    supported = fixed_template(tito_model).allowed_append_roles
    return tuple(role for role in VALID_APPEND_ROLES if role in supported)


# English-only on purpose: matches the production agentic flows tokenization
# and tool-call parsing are tuned against.
# Each question names the tool: the spec rollout needs a tool call back after a probe, and a
# model that has just answered without one keeps doing so when the question leaves it open.
USER_FOLLOWUP_TEXT = "Now check the weather in Shanghai with the get_weather tool."
USER_FOLLOWUP_TEXTS = (
    USER_FOLLOWUP_TEXT,
    "Now check the weather in Hangzhou with the get_weather tool.",
    "Now check the weather in London with the get_weather tool.",
)
# A prompt that asks for an answer without a tool, so the reply carries no tool call.
NO_TOOL_CALL_USER_TEXT = "Without using any tool, tell me one sentence about Beijing."
# tool_choice "none" keeps that reply free of parsed tool calls whatever the model writes,
# and kept special tokens let the server re-render a call the model writes into content.
NO_TOOL_CALL_REQUEST_ARGS = {"tool_choice": "none", "skip_special_tokens": False}
# A neutral result for a tool message that answers no call; a retry instruction makes some models loop.
SENTINEL_TOOL_RESULT = '{"result": "unavailable"}'
# Neutral about tools and answer length: the spec rollout needs a tool call after one use of it
# and a reply without a call after the other.
SYSTEM_REMINDER_TEXT = "Note: from now on, skip all pleasantries."
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
    """The fixed (system, user) prompt every rollout starts from."""
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


async def _chat_status(
    client, base_url, messages, request_kwargs, *, label, refused_status=_REFUSED
) -> tuple[int, dict | None]:
    """Send a request the server may refuse with *refused_status*; return its status and, on 200, the response.

    Non-streaming, so a refusal reaches the caller as a status instead of a broken stream.
    """
    payload = {"messages": messages, "tools": TOOLS, **request_kwargs, "stream": False}
    resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
    if resp.status_code == _ACCEPTED:
        return _ACCEPTED, resp.json()
    assert resp.status_code == refused_status, f"{label} failed ({resp.status_code}): {resp.text}"
    logger.info("%s refused (%d): %s", label, resp.status_code, resp.text[:300])
    return refused_status, None


async def _session_records(client, base_url) -> list[dict]:
    """The session's records along its latest generation's path, oldest first."""
    resp = await client.get(base_url)
    assert resp.status_code == 200, f"reading the session failed ({resp.status_code}): {resp.text}"
    return resp.json()["records"]


class _ModelBehaviorError(Exception):
    """The model did not do what a rollout step needs: call the tool, stop calling it, or reply with what is to be cut.

    The server did nothing wrong, so the sample stops at that step and stays valid.
    """


def _reply_head(message: dict) -> str:
    """The start of a reply's text, for the error that reports a rejected resample."""
    return (message.get("content") or message.get("reasoning_content") or "")[:100]


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
    cuts: list[dict] = field(default_factory=list)
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
        rejected = []
        complete_without_call = False
        for retry in range(_MAX_RESAMPLES + 1):
            response = await _chat(self.client, self.base_url, messages, _with_retry_seed(kwargs, retry), label=label)
            choice = response["choices"][0]
            message = choice["message"]
            if choice.get("finish_reason") != "length":
                if message.get("tool_calls") or not need_tool_call:
                    return self._accept(message)
                complete_without_call = True
            rejected.append(f"{choice.get('finish_reason')}: {_reply_head(message)!r}")
        # A turn cut on every resample is no model choice; it fails the run like any other turn.
        error = _ModelBehaviorError if complete_without_call else AssertionError
        raise error(
            f"{label} exhausted {_MAX_RESAMPLES} resamples (finish_reason='length' or no tool call when one is "
            f"needed); replies: {rejected}"
        )

    async def probe(self, messages, *, label, expected_status, refused_status=_REFUSED) -> dict | None:
        """Send a continuation the family decides; record its status and return the reply if it was accepted."""
        for retry in range(_MAX_RESAMPLES + 1):
            kwargs = _with_retry_seed(self.request_kwargs, retry)
            status, response = await _chat_status(
                self.client, self.base_url, messages, kwargs, label=label, refused_status=refused_status
            )
            if status != _ACCEPTED or response["choices"][0].get("finish_reason") != "length":
                break
        else:
            raise AssertionError(f"{label} exhausted {_MAX_RESAMPLES} resamples after finish_reason='length'")
        self.expectations.append({"step": label, "expected_status": expected_status, "actual_status": status})
        return None if status != _ACCEPTED else self._accept(response["choices"][0]["message"])

    async def cut(self, messages, *, max_tokens: int, label, retry: int) -> dict | None:
        """Regenerate the last reply with at most *max_tokens* tokens; ``None`` if it ended by itself before that."""
        kwargs = _with_retry_seed({**self.request_kwargs, "max_tokens": max_tokens}, retry)
        response = await _chat(self.client, self.base_url, messages, kwargs, label=label)
        choice = response["choices"][0]
        return choice["message"] if choice.get("finish_reason") == "length" else None

    async def records(self) -> list[dict]:
        return await _session_records(self.client, self.base_url)

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
        if reply.get("tool_calls"):
            raise _ModelBehaviorError(f"the model still calls a tool after {_MAX_TOOL_ROUNDS} answered rounds")
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
        if role == "user":
            # An accepted user probe is a user turn: the next question must name another city,
            # or the model is asked again what it has just been asked.
            driver.counters["user_count"] += 1
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


async def _run_truncate(driver: _Driver, tito, *, cut_kind: str, expected_status: int) -> None:
    """Cut one reply inside *cut_kind*, continue the cut turn, and check what the server did with it.

    *tito* is the family's TITO tokenizer, or ``None`` for a family without a closer.
    """
    messages = build_initial_messages()
    if cut_kind == "answer":
        # The first reply is a tool call without prose; the answer follows its result.
        call = await driver.chat(messages, label="Initial", need_tool_call=True)
        driver.events.append("initial")
        driver.append_tool_results(messages, call)
    await driver.chat(messages, label="Probe", need_tool_call=cut_kind == "tool_call")
    driver.events.append("probe")
    # A full reply tells where such a reply has its reasoning, call and answer: the probe places the
    # first cut.  A regeneration is another sample, so try a few until one stops in the planned place;
    # one that ends by itself is a full reply too and places the next cut.  Each attempt replaces the
    # one before it in the session, so only the last one is left to continue, wherever it stopped.
    max_tokens = _BLIND_CUT_TOKENS
    cut = None
    for attempt in range(_MAX_CUT_ATTEMPTS):
        if tito is not None and cut is None:
            placed = cut_length(tito, (await driver.records())[-1], cut_kind)
            if placed is None and attempt == 0:
                raise _ModelBehaviorError(f"the probed reply has no {cut_kind} to cut")
            max_tokens = placed or max_tokens
        cut = await driver.cut(messages, max_tokens=max_tokens, label=f"Cut in {cut_kind}", retry=attempt)
        driver.events.append("cut")
        if cut is not None and (tito is None or record_cut_kind(tito, (await driver.records())[-1]) == cut_kind):
            break
    if cut is None:
        raise _ModelBehaviorError(f"the last regenerated reply ended by itself within {max_tokens} tokens")
    driver.counters["tool_call_count"] += len(cut.get("tool_calls") or [])

    probed_messages = list(messages)
    # The continuation follows the normal rollout: answer the cut turn's complete calls, else ask on.
    if cut.get("tool_calls"):
        driver.append_tool_results(messages, cut)
    else:
        driver.append_user(messages, cut, driver.next_city_question())
    reply = await driver.probe(
        messages, label=CONTINUE_CUT_STEP, expected_status=expected_status, refused_status=_TRUNCATED_REFUSED
    )
    if reply is None:
        # The refusal leaves the cut turn as the session's last one, and a turn nothing closed fails
        # the mismatch check.  Re-sending the probed request replaces it with a complete turn.
        await driver.chat(probed_messages, label="After refused cut")
        driver.events.append("regenerate_refused_cut")
    elif tito is not None:
        driver.events.append("continue_cut")
        driver.cuts.extend(check_continued_cuts(tito, await driver.records()))


def _assert_rollout_supported(kind: SessionVerifyKind, tito_model: str, fixed: FixedTemplate) -> None:
    # The spec rollout returns to a tool call with a user question after an answer,
    # and the truncate rollout continues a cut turn without calls with one.
    if "user" not in fixed.roles_after_no_tool_call:
        raise ValueError(
            f"the {kind} rollout needs 'user' after an assistant without tool calls; "
            f"{tito_model!r} accepts only {sorted(fixed.roles_after_no_tool_call)} there"
        )


def _session_verify_kind(metadata: dict) -> SessionVerifyKind:
    if SESSION_VERIFY_KIND_KEY not in metadata:
        raise ValueError(f"session verification needs {SESSION_VERIFY_KIND_KEY!r} in the sample metadata")
    return SessionVerifyKind(metadata[SESSION_VERIFY_KIND_KEY])


@_journal_verifier_assertions("session_verify_agent.run_agent")
async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    """Custom-agent entry point.

    Returns ``{"driver_events": [...], EXPECTATIONS_KEY: [...], CUTS_KEY: [...], UNFINISHED_KEY: str | None,
    **counters}``.

    ``metadata`` must carry ``tito_model`` (the ``generate`` wrapper below
    injects it from ``args.tito_model``) and ``session_verify_kind`` (from the
    prompt record).  A truncate rollout also needs ``session_verify_cut_kind``
    (from the prompt record) and the run's ``hf_checkpoint`` and resolved
    ``allow_truncated_continuation`` (injected by ``generate``).  ``prompt`` is
    ignored: every rollout starts from ``build_initial_messages`` so runs are
    reproducible.
    """
    tito_model = metadata.get("tito_model")
    if tito_model is None:
        raise ValueError("session_verify_agent.run_agent requires tito_model in metadata")
    kind = _session_verify_kind(metadata)
    fixed = fixed_template(tito_model)
    if kind is not SessionVerifyKind.NORMAL:
        _assert_rollout_supported(kind, tito_model, fixed)

    rk = {k: v for k, v in request_kwargs.items() if k not in ("tools",)}
    async with httpx.AsyncClient(timeout=180) as client:
        driver = _Driver(client=client, base_url=base_url, request_kwargs=rk)
        unfinished = None
        try:
            if kind is SessionVerifyKind.SPEC:
                await _run_spec(driver, fixed)
            elif kind is SessionVerifyKind.TRUNCATE:
                await _run_truncate(
                    driver,
                    (
                        None
                        if fixed.truncation_closer is None
                        else load_tito_tokenizer(metadata["hf_checkpoint"], tito_model)
                    ),
                    cut_kind=metadata[SESSION_VERIFY_CUT_KIND_KEY],
                    expected_status=_ACCEPTED if metadata["allow_truncated_continuation"] else _TRUNCATED_REFUSED,
                )
            else:
                await _run_normal(driver, cycles=metadata.get("session_verify_cycles", DEFAULT_CYCLES))
        except _ModelBehaviorError as exc:
            unfinished = str(exc)
            logger.warning("%s rollout unfinished after events=%s: %s", kind, driver.events, unfinished)

    logger.info("Agent done: kind=%s events=%s expectations=%s", kind, driver.events, driver.expectations)
    return {
        "driver_events": driver.events,
        EXPECTATIONS_KEY: driver.expectations,
        CUTS_KEY: driver.cuts,
        UNFINISHED_KEY: unfinished,
        **driver.counters,
    }


def _verify_tito_samples(samples, *, kind: SessionVerifyKind, cut_kind: str | None = None) -> None:
    """Record every sample and defer hard failures to the run-level gate.

    The rollout loop treats custom-generate exceptions as retryable sample
    failures.  With a metrics sidecar, hard TITO mismatches, unmet
    expectations and cut turns the closer did not close must therefore be
    persisted instead of raised here and discarded with the sample.  *cut_kind*
    is where a truncate sample was told to cut.  Direct callers without a sidecar retain the
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
        cut_problems = [problem for cut in sample.metadata.get(CUTS_KEY) or [] for problem in cut["problems"]]
        if metrics_path:
            record = _sample_record(i, sample, kind, mismatches, forbidden, expectations)
            _append_session_verify_record({**record, "cut_kind": cut_kind})
        elif forbidden or unmet or cut_problems:
            raise AssertionError(
                f"Session multi-role e2e: {kind} sample {i} has forbidden mismatches {forbidden}, "
                f"unmet expectations {unmet} and cut turns the closer did not close {cut_problems}.  "
                "All must be empty for any TITO-correct setup."
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
        "cuts": sample.metadata.get(CUTS_KEY) or [],
        "unfinished": sample.metadata.get(UNFINISHED_KEY),
        "had_assistant_mismatch": bool(assistant_mismatches),
        "total_mismatches": len(mismatches),
        "assistant_mismatch_count": len(assistant_mismatches),
        "assistant_mismatch_example": assistant_example,
        "hard_mismatch_count": len(forbidden),
        "hard_mismatch_types": sorted({m.get("type") for m in forbidden}),
        "hard_mismatch_example": hard_example,
    }


def _fail_sample(message: str) -> None:
    """Fail the run for one sample: journal with a metrics sidecar, raise without one.

    Raising drops the sample's group, and the rollout loop refills the step with
    prompts of the next rollout's kind; the journaled error fails the run at the
    runner's gate instead.
    """
    error = AssertionError(message)
    if not _journal_verification_error(_GENERATE_STAGE, error):
        raise error


def _journal_unplanned_prompt(input: GenerateFnInput) -> None:
    """Fail the run when the rollout loop reads past the prompts planned for it.

    The rollout kind comes from prompt records read in file order,
    ``rollout_batch_size`` per step. When a sample group fails, the loop refills
    the step from the next records, which belong to the next kind, and the last
    step wraps around to the first kind. Every such run reads a prompt group
    past the planned ones, and the step that reads it cannot finish without it.

    Journal only: raising would drop this group too, and the loop would refill
    again. Without a metrics sidecar there is no run gate to tell. A run of one
    step has one kind, so a refill there mixes nothing.
    """
    group_index, num_rollout = input.sample.group_index, input.args.num_rollout
    if group_index is None or num_rollout is None or num_rollout < 2:
        return
    planned = input.args.rollout_batch_size * num_rollout
    if group_index >= planned:
        error = AssertionError(
            f"Session multi-role e2e: prompt group {group_index} lies past the {planned} planned ones.  A sample "
            "group failed and the rollout loop read more prompts to refill its step, so the steps no longer run "
            "one rollout each.  Check rollout logs for 'Task raised exception'."
        )
        logger.error("%s", error)
        _journal_verification_error(_GENERATE_STAGE, error)


@_journal_verifier_assertions(_GENERATE_STAGE)
async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Custom-generate wrapper that records each sample and checks rollout coverage.

    - The sample's prompt must be one of the planned ones (``_journal_unplanned_prompt``).
    - Every sample must carry the agent's metadata: the base generate still
      collects the samples of an agent that raised, and those carry none.
    - Spec and truncate samples must carry one expectation per probe of their
      rollout, in order; an unfinished one carries the probes up to the step it
      stopped at.  Without a metrics sidecar an unfinished sample fails here
      (with one, the runner gates how many there are across the run).
    - Without a metrics sidecar, at least one normal sample must have answered
      a tool call (with one, the runner checks this across the run).
    """
    tito_model = input.args.tito_model
    kind = _session_verify_kind(input.sample.metadata)
    _journal_unplanned_prompt(input)
    cycles = getattr(
        input.args, "session_verify_cycles", DEFAULT_CYCLES
    )  # config-access-exempt: optional custom session-verification hook argument
    # Sample.metadata is mutable even when the outer dataclass is frozen.
    input.sample.metadata["tito_model"] = tito_model
    input.sample.metadata["session_verify_cycles"] = cycles
    if kind is SessionVerifyKind.TRUNCATE:
        input.sample.metadata["hf_checkpoint"] = input.args.hf_checkpoint
        input.sample.metadata["allow_truncated_continuation"] = input.args.allow_truncated_continuation

    output = await _base_generate(input)

    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    if not samples:
        raise AssertionError("Session multi-role e2e: generate returned no samples")
    _verify_tito_samples(samples, kind=kind, cut_kind=input.sample.metadata.get(SESSION_VERIFY_CUT_KIND_KEY))

    steps = {
        SessionVerifyKind.SPEC: spec_probe_steps(fixed_template(tito_model)),
        SessionVerifyKind.TRUNCATE: [CONTINUE_CUT_STEP],
    }.get(kind)
    for i, sample in enumerate(samples):
        if "driver_events" not in sample.metadata:
            _fail_sample(
                f"Session multi-role e2e: {kind} sample {i} carries no agent metadata: "
                "the agent raised before it returned"
            )
        elif steps is not None:
            recorded = [e["step"] for e in sample.metadata.get(EXPECTATIONS_KEY) or []]
            unfinished = sample.metadata.get(UNFINISHED_KEY)
            if recorded != (steps if unfinished is None else steps[: len(recorded)]):
                _fail_sample(f"Session multi-role e2e: {kind} sample {i} recorded probes {recorded}, expected {steps}")
            elif unfinished is not None and not os.environ.get("MILES_SESSION_VERIFY_METRICS_PATH"):
                raise AssertionError(f"Session multi-role e2e: {kind} sample {i} is unfinished: {unfinished}")
    if (
        kind is SessionVerifyKind.NORMAL
        and not os.environ.get("MILES_SESSION_VERIFY_METRICS_PATH")
        and not any("append_tool" in sample.metadata.get("driver_events", []) for sample in samples)
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
