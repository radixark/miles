"""Recorded-session collector: one turn at a time per session; turns form a tree by parent, the client prunes it."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from miles.tinker.core.future import FAILED
from miles.tinker.core.prompt_renderer import PromptRenderer, Rendered
from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import GatewayError, OwnershipError, UserInputError

logger = logging.getLogger(__name__)

TINKER_PATH_PREFIX = "tinker://"
SWEEP_INTERVAL_S = 60.0
_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{31,127}")  # the chat route's only credential: unguessable


class SessionError(GatewayError):
    """Base of the recorded-session errors; each subclass carries the status the routes answer with."""


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
    """A session id is 32 to 128 chars of [A-Za-z0-9._:-], starting alphanumeric (e.g. a prefix + uuid4 hex)."""
    if not isinstance(session_id, str) or _SESSION_ID.fullmatch(session_id) is None:
        raise UserInputError(f"invalid session id {session_id!r}: use 32-128 chars of A-Z a-z 0-9 . _ : -")


@dataclass
class Turn:
    """One recorded generation: the ids the engine consumed and produced plus logprobs (a cookbook Transition)."""

    input_ids: Sequence[int]
    output_ids: Sequence[int]
    logprobs: Sequence[float]
    finish_reason: str  # "stop" | "length"
    created_at: float = field(default_factory=time.time)
    inherits: bool = False  # TITO: input_ids extend the parent turn's input_ids + output_ids (up to max_trim_tokens)
    reset_reason: str | None = None  # why a full render: first/retry/rewrite/budget/mismatch/stop_string/no_tito
    after_truncation: bool = False  # an ancestor's reply was cut at max_tokens and the harness continued past it
    parent: int | None = None  # the turn this prompt continues (the tree edge the client prunes by); None: a root
    messages: list[dict[str, Any]] | None = field(default=None, repr=False)  # request + reply, for attach points
    request_args: dict[str, Any] | None = field(default=None, repr=False)  # resolved TITO args a child inherits
    ended_on_stop: bool = False  # the reply ended on a request stop string, so its ids lack the end-of-turn token

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
            "parent": self.parent,
        }


@dataclass(frozen=True)
class TurnRequest:
    """One turn to record, independent of the wire protocol; the server's OpenAI adapter builds it."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    sampling_params: dict[str, Any]  # max_tokens (required); optional temperature / top_p / top_k / seed / stop
    chat_template_kwargs: dict[str, Any] | None = None  # per-request override of the gateway's template kwargs
    model: str | None = None  # the tinker:// spelling the client sent, checked against the bound version


@dataclass(frozen=True)
class TurnResult:
    """The recorded Turn plus the unified assistant message; an API adapter shapes both into its wire response."""

    turn: Turn
    assistant_message: dict[str, Any]  # OpenAI-style {role, content[, tool_calls]}: TITO stores it, adapters render it
    model: str = ""  # what actually sampled: the bound tinker:// sampler path, or the frozen base model


@dataclass
class TrajectorySession:
    """Per-trajectory state: owning tenant, pinned sampler path, and the recorded turns (a tree by Turn.parent)."""

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


def lineage_truncated(turns: list[Turn], parent: int | None) -> bool:
    """True when any turn on the path from the root to `parent` ended at max_tokens (finish_reason "length")."""
    while parent is not None:
        if turns[parent].finish_reason == "length":
            return True
        parent = turns[parent].parent
    return False


def max_new_tokens_of(sampling_params: dict[str, Any]) -> int:
    """The turn's max_tokens: a positive int, required as for Tinker sample."""
    max_tokens = sampling_params.get("max_tokens")
    if type(max_tokens) is not int or max_tokens < 1:
        raise UserInputError("max_tokens must be a positive integer (required, as for Tinker sample)")
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
        if not sampling_session_id or not isinstance(sampling_session_id, str):
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
            raise SessionNotFoundError(
                f"unknown session {session_id!r}; bind it first (tenant key + sampling_session_id)"
            )
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

    async def run_sweeper(self, interval_s: float = SWEEP_INTERVAL_S) -> None:
        """Every interval_s run sweep(); the gateway starts it beside service.run() and cancels it with it."""
        while True:
            await asyncio.sleep(interval_s)
            if dropped := self.sweep():
                logger.info(f"swept {dropped} idle recorded session(s)")

    def datum_budget(self, session: TrajectorySession) -> int:
        """The session's per-datum cap: the gateway's, lowered to the client's bind-time max_datum_tokens."""
        cap = self.service.config.max_tokens_per_datum
        return cap if session.max_datum_tokens is None else min(cap, session.max_datum_tokens)

    async def complete(self, session_id: str, request: TurnRequest) -> TurnResult:
        """Record one turn on a bound session under its lock: attach + render off the loop, sample, commit."""
        session = self._session_for_request(session_id, request.model)
        max_new_tokens = max_new_tokens_of(request.sampling_params)
        async with session.lock:  # a retry that overlaps its first attempt waits here and is then seen as a resend
            if self.sessions.get(session_id) is not session:
                raise SessionNotFoundError(f"session {session_id!r} was deleted")
            if len(session.turns) >= self.max_turns_per_session:
                cap = self.max_turns_per_session
                raise SessionLimitError(f"session {session_id!r} already holds {len(session.turns)} turns (cap {cap})")
            # tokenizing is CPU work; keep it off the loop that serves every tenant's Tinker traffic
            rendered = await asyncio.to_thread(
                self.renderer.prepare_pretokenized,
                session,
                request.messages,
                request.tools,
                request.chat_template_kwargs,
                max_new_tokens=max_new_tokens,
                budget=self.datum_budget(session),
            )
            if self.sessions.get(session_id) is not session:  # a DELETE landed while rendering: sample nothing
                raise SessionNotFoundError(f"session {session_id!r} was deleted")
            # the harness continued past a reply cut at max_tokens; miles session server v2 refuses, v1 desyncs
            after_truncation = lineage_truncated(session.turns, rendered.parent)
            if after_truncation and self.strict_truncation:
                raise TruncatedGenerationError("cannot extend a reply that ended at max_tokens; resample it or rebind")
            payload = self._payload(session, rendered.prompt_token_ids, request.sampling_params)
            sequence = await self._sample(session, payload)
            stop = request.sampling_params.get("stop")
            return self._commit_generation(session, request.messages, rendered, sequence, after_truncation, stop)

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
        rendered: Rendered,
        sequence: dict[str, Any],
        after_truncation: bool,
        stop: list[str] | None = None,
    ) -> TurnResult:
        """Build the Turn with its history and assistant message, then append it: the tree grows by one node."""
        turn = Turn(
            input_ids=array("i", rendered.prompt_token_ids),
            output_ids=array("i", (int(token) for token in sequence["tokens"])),
            logprobs=array("d", (float(value) for value in sequence["logprobs"])),
            finish_reason="length" if sequence.get("stop_reason") == "length" else "stop",
            created_at=self.clock(),
            inherits=rendered.inherits,
            reset_reason=rendered.reset_reason,
            after_truncation=after_truncation,
            parent=rendered.parent,
            request_args=rendered.request_args,
        )
        message = self.renderer.assistant_message(turn, stop)
        turn.messages = [*request_messages, message]  # the same dict the adapter renders: a child matches it later
        session.turns.append(turn)
        session.last_seen = turn.created_at
        return TurnResult(
            turn=turn, assistant_message=message, model=session.model_path or self.service.config.base_model
        )

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
