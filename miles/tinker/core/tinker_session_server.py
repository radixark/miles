"""Recorded-session collector: one turn at a time per session; every turn is kept, reset_reason marks segments."""

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


class SessionError(Exception):
    """Base of the recorded-session errors; status_code is what the session routes answer with."""

    status_code = 500


class SessionNotFoundError(SessionError):
    """No recorded session with this id."""

    status_code = 404


class TruncatedGenerationError(SessionError):
    """The harness continued past a reply cut at max_tokens and strict truncation refuses to extend it."""

    status_code = 409


class SessionLimitError(SessionError):
    """The tenant's open sessions or the session's turns hit the collector's cap."""

    status_code = 429


class SamplingBackendError(SessionError):
    """The engine failed the sample; nothing was recorded for the turn."""

    status_code = 502


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
    reset_reason: str | None = None  # why a full render opened a segment: first/retry/rewrite/budget/mismatch/no_tito
    after_truncation: bool = False  # the harness continued past a reply that ended with finish_reason="length"

    def as_json(self) -> dict[str, Any]:
        """Plain lists for the trajectory export (what the client's turns_to_trajectory reads)."""
        return {
            "input_ids": list(self.input_ids),
            "output_ids": list(self.output_ids),
            "logprobs": list(self.logprobs),
            "finish_reason": self.finish_reason,
            "created_at": self.created_at,
            "inherits": self.inherits,
            "reset_reason": self.reset_reason,
            "after_truncation": self.after_truncation,
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
    """The recorded Turn plus the unified assistant message; an API adapter shapes both into its wire response."""

    turn: Turn
    assistant_message: dict[str, Any]  # OpenAI-style {role, content[, tool_calls]}: TITO stores it, adapters render it


@dataclass
class TrajectorySession:
    """Per-trajectory state: owning tenant, pinned sampler path, recorded turns, and the TITO prefix state."""

    session_id: str
    tenant: str
    model_path: str | None = None  # tinker://M/sampler_weights/V; None samples the frozen base
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # held for a whole turn: render, sample and commit as one
    pending_request_id: str | None = None  # the sample running under the lock; DELETE cancels it
    max_datum_tokens: int | None = None  # the client's per-datum cap from bind; a TITO chain never grows past it
    sampling_session_id: str | None = None  # the Tinker sampling session bound at create: sampler version + lease
    messages: list[dict[str, Any]] | None = None  # history the last recorded turn answered, its reply appended
    token_ids: Sequence[int] | None = None  # the last turn's input_ids + output_ids, inherited by the next turn
    request_args: dict[str, Any] | None = None  # the last turn's resolved TITO args (kwargs, tools) it may inherit


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
        strict_truncation: bool = False,
    ) -> None:
        """Keep the service, renderer, TTL, clock, caps and whether a truncated reply may be extended."""
        self.service = service
        self.renderer = renderer
        self.session_ttl_s = session_ttl_s
        self.clock = clock
        self.max_sessions_per_tenant = max_sessions_per_tenant
        self.max_turns_per_session = max_turns_per_session
        self.strict_truncation = strict_truncation
        self.sessions: dict[str, TrajectorySession] = {}

    def create_session(
        self,
        session_id: str,
        tenant: str,
        sampling_session_id: str | None = None,
        max_datum_tokens: int | None = None,
    ) -> TrajectorySession:
        """Create or re-bind a session to a Tinker sampling session (its sampler version and its lease); caps apply."""
        validate_session_id(session_id)
        if not tenant:
            raise UserInputError("binding a session needs the tenant's API key")
        if not sampling_session_id:
            raise UserInputError("bind needs sampling_session_id: create a Tinker sampling session and pass its id")
        if max_datum_tokens is not None and (type(max_datum_tokens) is not int or max_datum_tokens < 1):
            raise UserInputError("max_datum_tokens must be a positive integer")
        session = self.sessions.get(session_id)
        if session is not None:
            if session.tenant != tenant:
                raise OwnershipError("session does not belong to this tenant")
            bound_path = self.service.resolve_sampler_path(tenant, sampling_session_id)
            if bound_path != session.model_path:
                raise UserInputError(f"session {session_id!r} is bound to {session.model_path!r}, not {bound_path!r}")
            session.sampling_session_id = sampling_session_id  # same version, possibly a fresh lease
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
            model_path=self.service.resolve_sampler_path(tenant, sampling_session_id),
            created_at=now,
            last_seen=now,
            max_datum_tokens=max_datum_tokens,
            sampling_session_id=sampling_session_id,
        )
        self.sessions[session_id] = session
        return session

    def _get_session(self, session_id: str, tenant: str | None = None) -> TrajectorySession:
        """Return the session; the tenant, when given, must own it; unknown ids raise SessionNotFoundError."""
        session = self.sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"unknown session {session_id!r}")
        if tenant is not None and session.tenant != tenant:
            raise OwnershipError("session does not belong to this tenant")
        return session

    def delete_session(self, session_id: str, tenant: str) -> None:
        """Drop a session and its turns; owner only; a sample still running under its lock is cancelled."""
        session = self._get_session(session_id, tenant)
        del self.sessions[session_id]
        if session.pending_request_id is not None:
            self.service.cancel(tenant, session.pending_request_id)

    def get_session(self, session_id: str, tenant: str) -> dict[str, Any]:
        """Export {session_id, model_path, max_trim_tokens, turns} for the client's turns_to_trajectory; owner only."""
        session = self._get_session(session_id, tenant)
        return {
            "session_id": session.session_id,
            "model_path": session.model_path,
            "max_trim_tokens": self.renderer.max_trim_tokens,
            "turns": [turn.as_json() for turn in session.turns],
        }

    def sweep(self, now: float | None = None) -> int:
        """Drop idle sessions past the TTL, and idle sessions whose tenant lost its Tinker lease; returns how many."""
        now = self.clock() if now is None else now
        lease_grace = self.service.config.lease_timeout_s
        expired = [
            sid
            for sid, session in self.sessions.items()
            if not session.lock.locked()
            and (
                now - session.last_seen >= self.session_ttl_s
                or (now - session.last_seen >= lease_grace and not self.service.tenant_alive(session.tenant))
            )
        ]
        for sid in expired:
            del self.sessions[sid]
        return len(expired)

    def _tito_budget(self, session: TrajectorySession) -> int:
        """TITO chain budget: the gateway per-datum cap, lowered to the client's bind-time max_datum_tokens."""
        cap = self.service.config.max_tokens_per_datum
        return cap if session.max_datum_tokens is None else min(cap, session.max_datum_tokens)

    async def complete(self, session_id: str, request: TurnRequest) -> TurnResult:
        """Record one turn on a bound session under its lock: render off the loop, sample under the lease, commit."""
        session = self._session_for_request(session_id, request.model)
        max_new_tokens = max_new_tokens_of(request.sampling_params)
        async with session.lock:  # a retry that overlaps its first attempt waits here and is then seen as a resend
            if self.sessions.get(session_id) is not session:
                raise SessionNotFoundError(f"session {session_id!r} was deleted")
            if len(session.turns) >= self.max_turns_per_session:
                cap = self.max_turns_per_session
                raise SessionLimitError(f"session {session_id!r} already holds {len(session.turns)} turns (cap {cap})")
            # tokenizing is CPU work; keep it off the loop that serves every tenant's Tinker traffic
            prompt_token_ids, inherits, reset_reason, request_args = await asyncio.to_thread(
                self.renderer.prepare_pretokenized,
                session,
                request.messages,
                request.tools,
                request.chat_template_kwargs,
                max_new_tokens=max_new_tokens,
                budget=self._tito_budget(session),
            )
            # the harness continued past a reply cut at max_tokens; miles session server v2 refuses, v1 desyncs
            last = session.turns[-1] if session.turns else None
            after_truncation = inherits and last is not None and last.finish_reason == "length"
            if after_truncation and self.strict_truncation:
                raise TruncatedGenerationError("cannot extend a reply that ended at max_tokens; resample it or rebind")
            sequence = await self._sample(session, self._payload(session, prompt_token_ids, request.sampling_params))
            return self._commit_generation(
                session,
                request.messages,
                prompt_token_ids,
                sequence,
                inherits,
                reset_reason,
                after_truncation,
                request_args,
            )

    @staticmethod
    def _payload(session: TrajectorySession, prompt_token_ids: list[int], sampling_params: dict) -> dict[str, Any]:
        """The Tinker sample body for one turn: the pinned sampler path, this prompt, the turn's sampling params."""
        return {
            "model_path": session.model_path,
            "num_samples": 1,
            "prompt_tokens": list(prompt_token_ids),
            "sampling_params": dict(sampling_params),
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        }

    async def _sample(self, session: TrajectorySession, payload: dict[str, Any]) -> dict[str, Any]:
        """Sample through the gateway under the session's lease; the request id stays on the session for DELETE."""
        # refused once the lease behind the sampling session is gone; else filed under it so expiry cancels the task
        request_id, _ = self.service.submit_sample(
            session.tenant, payload, lease_sampling_session_id=session.sampling_session_id
        )
        future = self.service.retrieve_future(session.tenant, request_id)
        assert future is not None, f"sampling future {request_id} vanished before it settled"
        session.pending_request_id = request_id
        try:
            await future.settled.wait()
        finally:
            session.pending_request_id = None
        if future.state == FAILED:
            if self.sessions.get(session.session_id) is not session:
                raise SessionNotFoundError(f"session {session.session_id!r} was deleted while sampling")
            if future.error_category == "user":
                raise UserInputError(future.error or "sampling rejected")
            raise SamplingBackendError(future.error or "sampling failed")
        return future.result["sequences"][0]

    def _commit_generation(
        self,
        session: TrajectorySession,
        request_messages: list[dict[str, Any]],
        prompt_token_ids: list[int],
        sequence: dict[str, Any],
        inherits: bool,
        reset_reason: str | None,
        after_truncation: bool,
        request_args: dict[str, Any] | None,
    ) -> TurnResult:
        """Build the Turn and its assistant message first, then write turns, last_seen and the TITO state together."""
        turn = Turn(
            input_ids=array("i", prompt_token_ids),
            output_ids=array("i", (int(token) for token in sequence["tokens"])),
            logprobs=array("d", (float(value) for value in sequence["logprobs"])),
            finish_reason="length" if sequence.get("stop_reason") == "length" else "stop",
            created_at=self.clock(),
            inherits=inherits,
            reset_reason=reset_reason,
            after_truncation=after_truncation,
        )
        message = self.renderer.assistant_message(turn)
        session.turns.append(turn)
        session.last_seen = turn.created_at
        self.renderer.update_pretokenized_state(session, turn, request_messages, message, request_args)  # same dict
        return TurnResult(turn=turn, assistant_message=message)

    def _session_for_request(self, session_id: str, model: str | None) -> TrajectorySession:
        """The bound session for a chat turn: the unguessable id is the credential; a tinker:// model must match."""
        validate_session_id(session_id)
        session = self._get_session(session_id)
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
