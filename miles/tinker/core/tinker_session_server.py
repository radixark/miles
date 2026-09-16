"""Token trajectory collector behind the gateway's OpenAI-compatible chat routes.

Skeleton: only what has no existing implementation is declared here; each docstring names what is reused.

The stateless routes mirror Tinker's OpenAI-compatible API (``…/oai/api/v1/chat/completions`` and
``…/oai/api/v1/completions``, ``model=tinker://M/sampler_weights/V``, bearer = Tinker API key, prompts
rendered with the base model's default HF chat template). Recorded sessions under
``/oai/sessions/{sid}/v1/…`` are the one extension: every turn keeps exactly the ids the engine consumed
and produced, plus their logprobs. A ``Turn`` is tinker-cookbook's ``Transition`` (``ob`` = input_ids,
``ac`` = output_ids + logprobs), so ``tinker_cookbook.rl.data_processing.trajectory_to_data`` decides on the
client whether turns chain into one Datum or split per turn; no merge logic lives here.

Reused, not reimplemented:
- sampling: ``TinkerService.submit_sample(tenant, payload)`` — validates prompt ids against the vocab
  (``validate_sample_payload``), resolves ``model_path`` through ``resolve_sampler_checkpoint`` (ownership),
  runs ``MilesBackend.sample`` (router ``/generate`` with ``lora_path`` + ``lora_backfill_paths``), and
  settles a ``RequestFuture`` (``TinkerService.retrieve_future`` + ``RequestFuture.settled``).
- sampling-session binding: ``TinkerService.get_sampler(tenant, sampling_session_id)["model_path"]``.
- rendering: the HF tokenizer's ``apply_chat_template`` / ``encode`` / ``decode``, injected by ``serve_tinker.py``.

Core layer: stdlib plus ``miles.tinker.core`` only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from miles.tinker.core.service import TinkerService


class UnknownSessionError(Exception):
    """No recorded session with this id (HTTP 404)."""


@dataclass
class Turn:
    """One recorded generation: exactly the ids the engine consumed and produced, plus their logprobs (= a cookbook Transition)."""

    input_ids: list[int]
    output_ids: list[int]
    logprobs: list[float]
    finish_reason: str  # "stop" | "length"
    created_at: float = field(default_factory=time.time)


@dataclass
class TrajectorySession:
    """Per-trajectory state: the owning tenant (whose key resolves the adapter on every turn), the pinned sampler path, and the turns."""

    session_id: str
    tenant: str
    model_path: str | None = None  # tinker://M/sampler_weights/V; None samples the frozen base
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


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


def render_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any],
    tokenizer,
) -> list[int]:
    """Validate the messages list and call tokenizer.apply_chat_template(messages, tools=..., add_generation_prompt=True, tokenize=True, **kwargs); nothing else."""
    raise NotImplementedError


def to_sample_payload(prompt_ids: list[int], request: dict[str, Any], model_path: str | None) -> dict[str, Any]:
    """OpenAI request → the internal payload TinkerService.submit_sample takes (prompt_tokens, num_samples, sampling_params{max_tokens (required, like Tinker sample), temperature, top_p, stop, seed}, model_path); an empty stop list is dropped because Tinker reads it as "ignore EOS"."""
    raise NotImplementedError


def build_chat_response(request: dict[str, Any], choices: list[dict[str, Any]], prompt_len: int) -> dict[str, Any]:
    """Assemble an OpenAI ChatCompletion JSON (id, created, model echoed, choices[].message, finish_reason, usage); the gateway's render_result renders Tinker JSON, not this shape."""
    raise NotImplementedError


def build_completion_response(
    request: dict[str, Any], choices: list[dict[str, Any]], prompt_len: int
) -> dict[str, Any]:
    """Assemble an OpenAI Completion JSON with choices[].text and usage."""
    raise NotImplementedError


def to_sse(response: dict[str, Any]) -> bytes:
    """Fake streaming: one SSE chunk carrying the whole message, then data: [DONE] (same trick as the miles session server)."""
    raise NotImplementedError


# --- collector -------------------------------------------------------------------


class TrajectoryCollector:
    """Owns the recorded sessions; everything about sampling is delegated to TinkerService."""

    def __init__(
        self, service: TinkerService, tokenizer, session_ttl_s: float, chat_template_kwargs: dict | None
    ) -> None:
        """Keep the TinkerService (submit_sample / retrieve_future / get_sampler), the injected HF tokenizer, and the collector's two settings."""
        self.service = service
        self.tokenizer = tokenizer
        self.session_ttl_s = session_ttl_s
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.sessions: dict[str, TrajectorySession] = {}

    def bind(
        self, session_id: str, tenant: str, model: str | None = None, sampling_session_id: str | None = None
    ) -> TrajectorySession:
        """Create or re-bind a session: model is a tinker:// path (checked eagerly with resolve_sampler_checkpoint) or a Tinker sampling_session_id (resolved with service.get_sampler); base model when neither is given; a different tinker:// path on a bound session is a UserInputError."""
        raise NotImplementedError

    def get(self, session_id: str, tenant: str | None = None) -> TrajectorySession:
        """Return the session; with a tenant given it must be the owner (OwnershipError), unknown ids raise UnknownSessionError."""
        raise NotImplementedError

    def delete(self, session_id: str, tenant: str) -> None:
        """Drop a session and its turns; owner only."""
        raise NotImplementedError

    def trajectory(self, session_id: str, tenant: str) -> dict[str, Any]:
        """Export {session_id, model_path, turns: [asdict(turn)]} for the client's turns_to_trajectory; owner only."""
        raise NotImplementedError

    def sweep(self, now: float | None = None) -> int:
        """Drop sessions idle longer than session_ttl_s; returns how many (TinkerService's lease sweeper does not know these sessions)."""
        raise NotImplementedError

    async def chat(
        self, request: dict[str, Any], *, session_id: str | None = None, tenant: str | None = None
    ) -> dict[str, Any] | bytes:
        """/chat/completions: session lookup or auto-register, tito_render_prompt then render_prompt, _sample, record a Turn, build_chat_response (to_sse when stream=true)."""
        raise NotImplementedError

    async def completions(
        self, request: dict[str, Any], *, session_id: str | None = None, tenant: str | None = None
    ) -> dict[str, Any] | bytes:
        """/completions: tokenizer.encode(prompt) (or token ids as-is), then the same _sample / record / build_completion_response path."""
        raise NotImplementedError

    async def _sample(self, tenant: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Reuse the gateway's sampling pipeline end to end: service.submit_sample(tenant, payload) → service.retrieve_future(tenant, request_id) → await future.settled.wait() → future.result["sequences"] (tokens, logprobs, stop_reason) or raise on future.error."""
        raise NotImplementedError
