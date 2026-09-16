"""Token trajectory collector behind the four recorded-session routes: renders OpenAI messages with the base model's chat template, samples via TinkerService.submit_sample (adapter M@V, ownership, vocab checks), records each turn's exact ids + logprobs in tenant-owned sessions; TITO hooks empty; core layer."""

from __future__ import annotations

import asyncio
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
    """One recorded generation: exactly the ids the engine consumed and produced, plus their logprobs (= a cookbook Transition); stored as arrays (4 bytes a token) because every turn keeps its full prompt."""

    input_ids: Sequence[int]
    output_ids: Sequence[int]
    logprobs: Sequence[float]
    finish_reason: str  # "stop" | "length"
    created_at: float = field(default_factory=time.time)

    def as_json(self) -> dict[str, Any]:
        """Plain lists for the GET export (what the client's turns_to_trajectory reads)."""
        return {
            "input_ids": list(self.input_ids),
            "output_ids": list(self.output_ids),
            "logprobs": list(self.logprobs),
            "finish_reason": self.finish_reason,
            "created_at": self.created_at,
        }


@dataclass
class TrajectorySession:
    """Per-trajectory state: the owning tenant (whose key resolves the adapter on every turn), the pinned sampler path, and the turns."""

    session_id: str
    tenant: str
    model_path: str | None = None  # tinker://M/sampler_weights/V; None samples the frozen base
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    in_flight: int = 0  # samples running right now; the TTL sweep leaves such a session alone


# --- TITO hooks (empty for now) ------------------------------------------------


def tito_render_prompt(
    session: TrajectorySession,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any],
    tokenizer,
) -> list[int] | None:
    """TITO hook: return prompt ids that inherit the previous turns' tokens, or None to re-render the full history (mirrors LinearTrajectory.prepare_pretokenized + TITOTokenizer.merge_tokens)."""
    return None


def on_turn_committed(session: TrajectorySession, turn: Turn) -> None:
    """TITO hook: update session-side TITO state after a turn is recorded (mirrors LinearTrajectory.update_pretokenized_state); no-op for now."""
    return None


# --- request / response shaping (no existing implementation for the OpenAI shapes) -----------


def _token_list(rendered) -> list[int]:
    """Flatten what apply_chat_template(tokenize=True) returns (a list, a BatchEncoding, or a batch of one) into ids."""
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
    """Validate the messages list and call tokenizer.apply_chat_template(messages, tools=..., add_generation_prompt=True, tokenize=True, **kwargs); nothing else."""
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


def to_sample_payload(prompt_ids: list[int], request: dict[str, Any], model_path: str | None) -> dict[str, Any]:
    """OpenAI request → the internal payload TinkerService.submit_sample takes (prompt_tokens, num_samples=1, sampling_params{max_tokens (required, like Tinker sample), temperature, top_p, top_k, seed, stop}, model_path); an empty stop list is dropped because Tinker reads it as "ignore EOS"."""
    max_tokens = request.get("max_tokens", request.get("max_completion_tokens"))
    if type(max_tokens) is not int or max_tokens < 1:
        raise UserInputError("max_tokens must be a positive integer (required, as for Tinker sample)")
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
    """Assemble an OpenAI ChatCompletion JSON (id, created, model echoed, one choice with message/finish_reason, usage); the gateway's render_result renders Tinker JSON, not this shape."""
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
    ) -> None:
        """Keep the TinkerService (submit_sample / retrieve_future / get_sampler), the injected HF tokenizer, the collector's settings and the two per-tenant caps."""
        self.service = service
        self.tokenizer = tokenizer
        self.session_ttl_s = session_ttl_s
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.clock = clock
        self.max_sessions_per_tenant = max_sessions_per_tenant
        self.max_turns_per_session = max_turns_per_session
        self.sessions: dict[str, TrajectorySession] = {}

    def bind(
        self, session_id: str, tenant: str, model: str | None = None, sampling_session_id: str | None = None
    ) -> TrajectorySession:
        """Create or re-bind a session: model is a tinker:// path (checked eagerly with resolve_sampler_checkpoint) or a Tinker sampling_session_id (resolved with service.get_sampler); base model when neither is given; a different tinker:// path on a bound session is a UserInputError; a placeholder key cannot bind; the tenant's open-session cap is enforced."""
        validate_session_id(session_id)
        if not tenant or tenant in PLACEHOLDER_KEYS:
            raise UserInputError("binding a session needs the tenant's API key")
        session = self.sessions.get(session_id)
        if session is not None:
            if session.tenant != tenant:
                raise OwnershipError("session does not belong to this tenant")
            self._check_same_version(session, model)
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
        )
        self.sessions[session_id] = session
        return session

    def get(self, session_id: str, tenant: str | None = None) -> TrajectorySession:
        """Return the session; with a tenant given it must be the owner (OwnershipError), unknown ids raise UnknownSessionError."""
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
        """Export {session_id, model_path, turns: [turn.as_json()]} for the client's turns_to_trajectory; owner only."""
        session = self.get(session_id, tenant)
        return {
            "session_id": session.session_id,
            "model_path": session.model_path,
            "turns": [turn.as_json() for turn in session.turns],
        }

    def sweep(self, now: float | None = None) -> int:
        """Safety net for trials that died before DELETE: drop sessions idle longer than session_ttl_s and with no sample in flight; returns how many."""
        now = self.clock() if now is None else now
        expired = [
            sid
            for sid, session in self.sessions.items()
            if session.in_flight == 0 and now - session.last_seen >= self.session_ttl_s
        ]
        for sid in expired:
            del self.sessions[sid]
        return len(expired)

    async def chat(self, request: dict[str, Any], *, session_id: str, tenant: str | None = None) -> dict[str, Any]:
        """The recorded chat completion: session lookup or auto-register, turn cap, tito_render_prompt then render_prompt (off the event loop), _sample, record a Turn, build_chat_response."""
        session = self._session_for_request(session_id, tenant, request.get("model"))
        if len(session.turns) >= self.max_turns_per_session:
            raise SessionLimitError(
                f"session {session_id!r} already holds {len(session.turns)} turns (cap {self.max_turns_per_session})"
            )
        messages = request.get("messages")
        tools = request.get("tools") or None
        template_kwargs = self._template_kwargs(request)
        prompt_ids = tito_render_prompt(session, messages, tools, template_kwargs, self.tokenizer)
        if prompt_ids is None:
            # tokenizing a long history is CPU work; keep it off the loop that serves every tenant's Tinker traffic
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
        )
        session.turns.append(turn)
        session.last_seen = turn.created_at
        on_turn_committed(session, turn)
        output_ids = list(turn.output_ids)
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return build_chat_response(request, output_ids, text, turn.finish_reason, len(prompt_ids))

    async def _sample(self, tenant: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Reuse the gateway's sampling pipeline end to end: service.submit_sample(tenant, payload) → service.retrieve_future(tenant, request_id) → await future.settled.wait() → future.result["sequences"][0] (tokens, logprobs, stop_reason) or raise on future.error."""
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
        """The recorded session this request samples in: a known id (same version; no key, the placeholder key or the owner's key) or a new id auto-registered under a real bearer; another tenant's key is refused."""
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
        """The gateway's chat_template_kwargs, overridden by the request's chat_template_kwargs object (e.g. enable_thinking)."""
        kwargs = dict(self.chat_template_kwargs)
        override = request.get("chat_template_kwargs")
        if override is not None:
            if not isinstance(override, dict):
                raise UserInputError("chat_template_kwargs must be an object")
            kwargs.update(override)
        return kwargs

    def _resolve_model_path(self, tenant: str, model: str | None, sampling_session_id: str | None) -> str | None:
        """None for the frozen base, else the tinker:// sampler path after resolve_sampler_checkpoint proved it exists and belongs to the tenant."""
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
        """A bound session serves one sampler version; only tinker:// spellings are compared (harness model names are ignored)."""
        if model and model.startswith(TINKER_PATH_PREFIX) and model != session.model_path:
            raise UserInputError(
                f"session {session.session_id!r} is bound to {session.model_path!r}; start a new session for {model!r}"
            )
