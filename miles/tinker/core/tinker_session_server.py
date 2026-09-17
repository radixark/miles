"""Recorded-session collector: render messages, sample via TinkerService, record exact ids; optional TITO; core."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from miles.tinker.core.future import FAILED
from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import OwnershipError, UserInputError
from miles.tinker.core.utils import resolve_sampler_checkpoint

logger = logging.getLogger(__name__)

TINKER_PATH_PREFIX = "tinker://"
# what Harbor's harness bindings hand the agent as its OpenAI key (harbor_agent_function.build_trial_config)
PLACEHOLDER_KEYS = frozenset({"dummy"})
_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class UnknownSessionError(Exception):
    """No recorded session with this id (HTTP 404)."""


class SamplingBackendError(Exception):
    """The engine failed the sample; the turn is not recorded (HTTP 502)."""


class SessionLimitError(Exception):
    """The tenant's open sessions or the session's turns hit the collector's cap (HTTP 429)."""


def validate_session_id(session_id: str) -> None:
    """Session ids come from the URL path: one to 128 chars of [A-Za-z0-9._:-], starting alphanumeric."""
    if not isinstance(session_id, str) or _SESSION_ID.fullmatch(session_id) is None:
        raise UserInputError(f"invalid session id {session_id!r}: use 1-128 chars of A-Z a-z 0-9 . _ : -")


@dataclass
class Turn:
    """One recorded generation: the ids the engine consumed and produced plus logprobs (a cookbook Transition)."""

    input_ids: Sequence[int]
    output_ids: Sequence[int]
    logprobs: Sequence[float]
    finish_reason: str  # "stop" | "length"
    created_at: float = field(default_factory=time.time)
    inherits: bool = False  # TITO: input_ids extend the previous turn's input_ids + output_ids

    def as_json(self) -> dict[str, Any]:
        """Plain lists for the GET export (what the client's turns_to_trajectory reads)."""
        return {
            "input_ids": list(self.input_ids),
            "output_ids": list(self.output_ids),
            "logprobs": list(self.logprobs),
            "finish_reason": self.finish_reason,
            "created_at": self.created_at,
            "inherits": self.inherits,
        }


@dataclass
class TrajectorySession:
    """Per-trajectory state: owning tenant, pinned sampler path, recorded turns, and the TITO prefix state."""

    session_id: str
    tenant: str
    model_path: str | None = None  # tinker://M/sampler_weights/V; None samples the frozen base
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    in_flight: int = 0  # samples running right now; the TTL sweep leaves such a session alone
    max_datum_tokens: int | None = None  # the client's per-datum cap from bind; a TITO chain never grows past it
    tito_messages: list[dict[str, Any]] | None = None  # history the last recorded turn answered, its reply appended
    tito_token_ids: Sequence[int] | None = None  # the last turn's input_ids + output_ids, inherited by the next turn


# --- TITO: prefix inheritance over the injected TITOTokenizer (miles/utils/chat_template_utils) ------------


def tito_render_prompt(
    session: TrajectorySession,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tito_tokenizer,
    *,
    max_new_tokens: int,
    budget: int | None,
) -> list[int] | None:
    """TITO: previous turn's ids + tokens of the appended messages (merge_tokens); None = full render, new segment."""
    if session.tito_messages is None or session.tito_token_ids is None:
        return None
    if not isinstance(messages, list) or len(messages) <= len(session.tito_messages):
        return None
    try:
        prompt = tito_tokenizer.merge_tokens(
            old_messages=session.tito_messages,
            new_messages=messages,
            pretokenized_token_ids=session.tito_token_ids,
            tools=tools,
        )
    except ValueError:  # the harness edited, reordered or summarized the history; a fresh render is still exact
        return None
    prompt_ids = [int(token) for token in prompt]
    if budget is not None and len(prompt_ids) + max_new_tokens > budget:
        return None
    return prompt_ids


def on_turn_committed(
    session: TrajectorySession, turn: Turn, messages: list[dict[str, Any]], reply: dict[str, Any]
) -> None:
    """TITO: remember the answered history + reply and this turn's ids as the prefix the next turn inherits."""
    session.tito_messages = [*messages, reply]
    session.tito_token_ids = array("i", [*turn.input_ids, *turn.output_ids])


# --- request / response shaping (no existing implementation for the OpenAI shapes) -----------


def _token_list(rendered) -> list[int]:
    """Flatten apply_chat_template(tokenize=True) output (list, BatchEncoding, or batch of one) into ids."""
    if hasattr(rendered, "input_ids"):
        rendered = rendered["input_ids"]
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(token) for token in rendered]


def render_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any],
    tokenizer,
) -> list[int]:
    """Validate the messages and render them with apply_chat_template(add_generation_prompt=True, tokenize=True)."""
    if not isinstance(messages, list) or not messages:
        raise UserInputError("messages must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or "role" not in message:
            raise UserInputError(f"messages[{index}] must be an object with a role")
    kwargs = dict(chat_template_kwargs)
    if tools:
        kwargs["tools"] = tools
    try:
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kwargs)
    except (TypeError, ValueError, KeyError) as error:
        raise UserInputError(f"cannot render messages with the chat template: {error}") from error
    ids = _token_list(rendered)
    if not ids:
        raise UserInputError("the chat template rendered an empty prompt")
    return ids


def _number(request: dict[str, Any], key: str, default: float) -> float:
    value = request.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UserInputError(f"{key} must be a number")
    return value


def max_new_tokens_of(request: dict[str, Any]) -> int:
    """The request's max_tokens (or max_completion_tokens): a positive int, required as for Tinker sample."""
    max_tokens = request.get("max_tokens", request.get("max_completion_tokens"))
    if type(max_tokens) is not int or max_tokens < 1:
        raise UserInputError("max_tokens must be a positive integer (required, as for Tinker sample)")
    return max_tokens


def to_sample_payload(prompt_ids: list[int], request: dict[str, Any], model_path: str | None) -> dict[str, Any]:
    """OpenAI request → TinkerService.submit_sample payload (n=1, max_tokens required; empty stop list dropped)."""
    max_tokens = max_new_tokens_of(request)
    if request.get("n", 1) != 1:
        raise UserInputError("a recorded session samples one completion per turn; use n=1")
    if request.get("stream"):
        raise UserInputError("stream=true is not supported on the recorded session route")
    sampling_params: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": _number(request, "temperature", 1.0),
        "top_p": _number(request, "top_p", 1.0),
    }
    for key in ("top_k", "seed"):
        if request.get(key) is not None:
            sampling_params[key] = request[key]
    stop = request.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    if stop:
        if not isinstance(stop, list) or not all(isinstance(item, str) for item in stop):
            raise UserInputError("stop must be a string or a list of strings")
        sampling_params["stop"] = list(stop)
    return {
        "model_path": model_path,
        "num_samples": 1,
        "prompt_tokens": list(prompt_ids),
        "sampling_params": sampling_params,
        "prompt_logprobs": False,
        "topk_prompt_logprobs": 0,
    }


def build_chat_response(
    request: dict[str, Any], output_ids: list[int], text: str, finish_reason: str, prompt_len: int
) -> dict[str, Any]:
    """Assemble an OpenAI ChatCompletion JSON: one choice with message/finish_reason plus usage."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.get("model") or "",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": len(output_ids),
            "total_tokens": prompt_len + len(output_ids),
        },
    }


# --- collector -------------------------------------------------------------------


class TrajectoryCollector:
    """Owns the recorded sessions; everything about sampling is delegated to TinkerService."""

    def __init__(
        self,
        service: TinkerService,
        tokenizer,
        session_ttl_s: float,
        chat_template_kwargs: dict | None,
        clock: Callable[[], float] = time.time,
        max_sessions_per_tenant: int = 1024,
        max_turns_per_session: int = 1024,
        tito_tokenizer=None,
        sweep_interval_s: float = 60.0,
    ) -> None:
        """Keep the service, HF tokenizer, settings (TTL, sweep period), caps and the optional TITOTokenizer."""
        self.service = service
        self.tokenizer = tokenizer
        self.session_ttl_s = session_ttl_s
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.clock = clock
        self.max_sessions_per_tenant = max_sessions_per_tenant
        self.max_turns_per_session = max_turns_per_session
        self.tito_tokenizer = tito_tokenizer
        self.sweep_interval_s = sweep_interval_s
        self.sessions: dict[str, TrajectorySession] = {}

    def bind(
        self,
        session_id: str,
        tenant: str,
        model: str | None = None,
        sampling_session_id: str | None = None,
        max_datum_tokens: int | None = None,
    ) -> TrajectorySession:
        """Create or re-bind a session pinned to a tinker:// path or sampling_session_id; caps and ownership apply."""
        validate_session_id(session_id)
        if not tenant or tenant in PLACEHOLDER_KEYS:
            raise UserInputError("binding a session needs the tenant's API key")
        if max_datum_tokens is not None and (type(max_datum_tokens) is not int or max_datum_tokens < 1):
            raise UserInputError("max_datum_tokens must be a positive integer")
        session = self.sessions.get(session_id)
        if session is not None:
            if session.tenant != tenant:
                raise OwnershipError("session does not belong to this tenant")
            self._check_same_version(session, model)
            if max_datum_tokens is not None:
                session.max_datum_tokens = max_datum_tokens
            session.last_seen = self.clock()
            return session
        open_sessions = sum(1 for existing in self.sessions.values() if existing.tenant == tenant)
        if open_sessions >= self.max_sessions_per_tenant:
            raise SessionLimitError(
                f"tenant has {open_sessions} open recorded sessions (cap {self.max_sessions_per_tenant}); "
                "DELETE finished sessions or wait for the TTL sweep"
            )
        now = self.clock()
        session = TrajectorySession(
            session_id=session_id,
            tenant=tenant,
            model_path=self._resolve_model_path(tenant, model, sampling_session_id),
            created_at=now,
            last_seen=now,
            max_datum_tokens=max_datum_tokens,
        )
        self.sessions[session_id] = session
        return session

    def get(self, session_id: str, tenant: str | None = None) -> TrajectorySession:
        """Return the session; the tenant, when given, must own it; unknown ids raise UnknownSessionError."""
        session = self.sessions.get(session_id)
        if session is None:
            raise UnknownSessionError(f"unknown session {session_id!r}")
        if tenant is not None and session.tenant != tenant:
            raise OwnershipError("session does not belong to this tenant")
        return session

    def delete(self, session_id: str, tenant: str) -> None:
        """Drop a session and its turns; owner only."""
        self.get(session_id, tenant)
        del self.sessions[session_id]

    def trajectory(self, session_id: str, tenant: str) -> dict[str, Any]:
        """Export {session_id, model_path, turns} for the client's turns_to_trajectory; owner only."""
        session = self.get(session_id, tenant)
        return {
            "session_id": session.session_id,
            "model_path": session.model_path,
            "turns": [turn.as_json() for turn in session.turns],
        }

    def sweep(self, now: float | None = None) -> int:
        """Drop idle sessions past the TTL, and idle sessions whose tenant lost its Tinker lease; returns how many."""
        now = self.clock() if now is None else now
        lease_grace = self.service.config.lease_timeout_s
        expired = [
            sid
            for sid, session in self.sessions.items()
            if session.in_flight == 0
            and (
                now - session.last_seen >= self.session_ttl_s
                or (now - session.last_seen >= lease_grace and not self._tenant_alive(session.tenant))
            )
        ]
        for sid in expired:
            del self.sessions[sid]
        return len(expired)

    async def run_sweeper(self) -> None:
        """Every sweep_interval_s drop what sweep() considers dead; runs for the life of the gateway."""
        while True:
            await asyncio.sleep(self.sweep_interval_s)
            if dropped := self.sweep():
                logger.info(f"swept {dropped} recorded session(s)")

    def _tenant_alive(self, tenant: str) -> bool:
        """True while the tenant still holds a Tinker session lease (service.sessions records know their tenant)."""
        return any(record.tenant == tenant for record in self.service.sessions.values())

    def _tito_budget(self, session: TrajectorySession) -> int:
        """TITO chain budget: the gateway per-datum cap, lowered to the client's bind-time max_datum_tokens."""
        cap = self.service.config.max_tokens_per_datum
        return cap if session.max_datum_tokens is None else min(cap, session.max_datum_tokens)

    def _tito_tokenizer_for(self, request: dict[str, Any]):
        """The injected TITOTokenizer, re-scoped with the request's chat_template_kwargs when present."""
        override = request.get("chat_template_kwargs")
        if not override:
            return self.tito_tokenizer
        if not isinstance(override, dict):
            raise UserInputError("chat_template_kwargs must be an object")
        try:
            return self.tito_tokenizer.clone_with_chat_template_kwargs(override)
        except ValueError as error:
            raise UserInputError(f"chat_template_kwargs conflict with the TITO template: {error}") from error

    async def chat(self, request: dict[str, Any], *, session_id: str, tenant: str | None = None) -> dict[str, Any]:
        """Recorded chat completion: TITO prompt else full render (off the loop), sample, record the Turn, respond."""
        session = self._session_for_request(session_id, tenant, request.get("model"))
        if len(session.turns) >= self.max_turns_per_session:
            raise SessionLimitError(
                f"session {session_id!r} already holds {len(session.turns)} turns (cap {self.max_turns_per_session})"
            )
        messages = request.get("messages")
        tools = request.get("tools") or None
        template_kwargs = self._template_kwargs(request)
        max_new_tokens = max_new_tokens_of(request)
        prompt_ids = None
        if self.tito_tokenizer is not None:
            # tokenizing is CPU work; keep it off the loop that serves every tenant's Tinker traffic
            prompt_ids = await asyncio.to_thread(
                tito_render_prompt,
                session,
                messages,
                tools,
                self._tito_tokenizer_for(request),
                max_new_tokens=max_new_tokens,
                budget=self._tito_budget(session),
            )
        inherits = prompt_ids is not None
        if prompt_ids is None:
            prompt_ids = await asyncio.to_thread(render_prompt, messages, tools, template_kwargs, self.tokenizer)
        payload = to_sample_payload(prompt_ids, request, session.model_path)
        session.in_flight += 1
        try:
            sequence = await self._sample(session.tenant, payload)
        finally:
            session.in_flight -= 1
        turn = Turn(
            input_ids=array("i", prompt_ids),
            output_ids=array("i", (int(token) for token in sequence["tokens"])),
            logprobs=array("d", (float(value) for value in sequence["logprobs"])),
            finish_reason="length" if sequence.get("stop_reason") == "length" else "stop",
            created_at=self.clock(),
            inherits=inherits,
        )
        session.turns.append(turn)
        session.last_seen = turn.created_at
        output_ids = list(turn.output_ids)
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        if self.tito_tokenizer is not None:
            on_turn_committed(session, turn, messages, {"role": "assistant", "content": text})
        return build_chat_response(request, output_ids, text, turn.finish_reason, len(prompt_ids))

    async def _sample(self, tenant: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Sample through the gateway: submit_sample → retrieve_future → settled → sequences[0], or raise."""
        request_id, _ = self.service.submit_sample(tenant, payload)
        future = self.service.retrieve_future(tenant, request_id)
        assert future is not None, f"sampling future {request_id} vanished before it settled"
        await future.settled.wait()
        if future.state == FAILED:
            if future.error_category == "user":
                raise UserInputError(future.error or "sampling rejected")
            raise SamplingBackendError(future.error or "sampling failed")
        return future.result["sequences"][0]

    def _session_for_request(self, session_id: str, tenant: str | None, model: str | None) -> TrajectorySession:
        """The session for this request: a known id (owner or placeholder key) or a new id under a real bearer."""
        validate_session_id(session_id)
        placeholder = not tenant or tenant in PLACEHOLDER_KEYS
        session = self.sessions.get(session_id)
        if session is None:
            if placeholder:
                raise UnknownSessionError(
                    f"unknown session {session_id!r}: bind it with POST /oai/sessions/{session_id} "
                    "or send the tenant's bearer token"
                )
            return self.bind(session_id, tenant, model)
        if not placeholder and tenant != session.tenant:
            raise OwnershipError("session does not belong to this tenant")
        self._check_same_version(session, model)
        session.last_seen = self.clock()
        return session

    def _template_kwargs(self, request: dict[str, Any]) -> dict[str, Any]:
        """The gateway's chat_template_kwargs, overridden by the request's chat_template_kwargs object."""
        kwargs = dict(self.chat_template_kwargs)
        override = request.get("chat_template_kwargs")
        if override is not None:
            if not isinstance(override, dict):
                raise UserInputError("chat_template_kwargs must be an object")
            kwargs.update(override)
        return kwargs

    def _resolve_model_path(self, tenant: str, model: str | None, sampling_session_id: str | None) -> str | None:
        """None for the frozen base, else the tinker:// sampler path once resolve_sampler_checkpoint accepted it."""
        if sampling_session_id is not None:
            model = self.service.get_sampler(tenant, sampling_session_id)["model_path"]
        if model is None or model == self.service.config.base_model:
            return None
        if not model.startswith(TINKER_PATH_PREFIX):
            raise UserInputError(
                f"model must be {self.service.config.base_model!r} or a tinker://…/sampler_weights/… path, got {model!r}"
            )
        resolve_sampler_checkpoint(self.service.config.checkpoint_root, tenant, model, self.service.config.base_model)
        return model

    @staticmethod
    def _check_same_version(session: TrajectorySession, model: str | None) -> None:
        """A bound session serves one sampler version; only tinker:// spellings are compared."""
        if model and model.startswith(TINKER_PATH_PREFIX) and model != session.model_path:
            raise UserInputError(
                f"session {session.session_id!r} is bound to {session.model_path!r}; start a new session for {model!r}"
            )
