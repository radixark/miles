"""Recorded-session collector: sessions, ownership, caps and turns; PromptRenderer renders, TinkerService samples."""

from __future__ import annotations

import asyncio
import re
import time
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from miles.tinker.core.future import FAILED
from miles.tinker.core.prompt_renderer import PromptRenderer
from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import OwnershipError, UserInputError

TINKER_PATH_PREFIX = "tinker://"
_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class UnknownSessionError(Exception):
    """No recorded session with this id."""


class SamplingBackendError(Exception):
    """The engine failed the sample; nothing was recorded for the turn."""


class SessionLimitError(Exception):
    """The tenant's open sessions or the session's turns hit the collector's cap."""


def validate_session_id(session_id: str) -> None:
    """A session id is one to 128 chars of [A-Za-z0-9._:-], starting alphanumeric."""
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
        """Plain lists for the trajectory export (what the client's turns_to_trajectory reads)."""
        return {
            "input_ids": list(self.input_ids),
            "output_ids": list(self.output_ids),
            "logprobs": list(self.logprobs),
            "finish_reason": self.finish_reason,
            "created_at": self.created_at,
            "inherits": self.inherits,
        }


@dataclass(frozen=True)
class TurnRequest:
    """One turn to record, independent of the wire protocol; the server's OpenAI adapter builds it."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    sampling_params: dict[str, Any]  # max_tokens (required), temperature, top_p, optional top_k / seed / stop
    chat_template_kwargs: dict[str, Any] | None = None  # per-request override of the gateway's template kwargs
    model: str | None = None  # the tinker:// spelling the client sent, checked against the bound version


@dataclass(frozen=True)
class TurnResult:
    """The recorded Turn plus its decoded text; the server shapes this into the wire response."""

    turn: Turn
    text: str


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


def max_new_tokens_of(sampling_params: dict[str, Any]) -> int:
    """The turn's max_tokens: a positive int, required as for Tinker sample."""
    max_tokens = sampling_params.get("max_tokens")
    if type(max_tokens) is not int or max_tokens < 1:
        raise UserInputError("sampling_params.max_tokens must be a positive integer")
    return max_tokens


# --- collector -------------------------------------------------------------------


class TrajectoryCollector:
    """Owns the recorded sessions; everything about sampling is delegated to TinkerService."""

    def __init__(
        self,
        service: TinkerService,
        renderer: PromptRenderer,
        session_ttl_s: float,
        clock: Callable[[], float] = time.time,
        max_sessions_per_tenant: int = 1024,
        max_turns_per_session: int = 1024,
    ) -> None:
        """Keep the service, the prompt renderer, the TTL, the clock and the per-tenant / per-session caps."""
        self.service = service
        self.renderer = renderer
        self.session_ttl_s = session_ttl_s
        self.clock = clock
        self.max_sessions_per_tenant = max_sessions_per_tenant
        self.max_turns_per_session = max_turns_per_session
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
        if not tenant:
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
                "delete finished sessions or wait for the TTL sweep"
            )
        now = self.clock()
        session = TrajectorySession(
            session_id=session_id,
            tenant=tenant,
            model_path=self.service.resolve_sampler_path(tenant, model, sampling_session_id),
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

    def _tenant_alive(self, tenant: str) -> bool:
        """True while the tenant still holds a Tinker session lease; asked of the service, not read off its state."""
        return self.service.tenant_alive(tenant)

    def _tito_budget(self, session: TrajectorySession) -> int:
        """TITO chain budget: the gateway per-datum cap, lowered to the client's bind-time max_datum_tokens."""
        cap = self.service.config.max_tokens_per_datum
        return cap if session.max_datum_tokens is None else min(cap, session.max_datum_tokens)

    async def complete(self, session_id: str, tenant: str | None, request: TurnRequest) -> TurnResult:
        """Record one turn: render the prompt off the loop, sample via the service, append and remember the Turn."""
        session = self._session_for_request(session_id, tenant, request.model)
        if len(session.turns) >= self.max_turns_per_session:
            raise SessionLimitError(
                f"session {session_id!r} already holds {len(session.turns)} turns (cap {self.max_turns_per_session})"
            )
        messages = request.messages
        max_new_tokens = max_new_tokens_of(request.sampling_params)
        # tokenizing is CPU work; keep it off the loop that serves every tenant's Tinker traffic
        prompt_ids, inherits = await asyncio.to_thread(
            self.renderer.render,
            session,
            messages,
            request.tools,
            request.chat_template_kwargs,
            max_new_tokens=max_new_tokens,
            budget=self._tito_budget(session),
        )
        payload = {
            "model_path": session.model_path,
            "num_samples": 1,
            "prompt_tokens": list(prompt_ids),
            "sampling_params": dict(request.sampling_params),
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        }
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
        text = self.renderer.decode(turn.output_ids)
        self.renderer.committed(session, turn, messages, text)
        return TurnResult(turn=turn, text=text)

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
        """The session for this request: a known id (its owner, or anonymous) or a new id under a real tenant key."""
        validate_session_id(session_id)
        anonymous = not tenant
        session = self.sessions.get(session_id)
        if session is None:
            if anonymous:
                raise UnknownSessionError(f"unknown session {session_id!r}")
            return self.bind(session_id, tenant, model)
        if not anonymous and tenant != session.tenant:
            raise OwnershipError("session does not belong to this tenant")
        self._check_same_version(session, model)
        session.last_seen = self.clock()
        return session

    @staticmethod
    def _check_same_version(session: TrajectorySession, model: str | None) -> None:
        """A bound session serves one sampler version; only tinker:// spellings are compared."""
        if model and model.startswith(TINKER_PATH_PREFIX) and model != session.model_path:
            raise UserInputError(
                f"session {session.session_id!r} is bound to {session.model_path!r}; start a new session for {model!r}"
            )
