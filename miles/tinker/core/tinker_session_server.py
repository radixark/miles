"""Token trajectory collector behind the gateway's OpenAI-compatible chat routes.

Skeleton: every class and function documents what it will do; bodies land in follow-up commits.

The stateless routes mirror Tinker's OpenAI-compatible API (``…/oai/api/v1/chat/completions`` and
``…/oai/api/v1/completions``, ``model=tinker://M/sampler_weights/V``, bearer = Tinker API key, prompts
rendered with the base model's default HF chat template). Recorded sessions under
``/oai/sessions/{sid}/v1/…`` are the one extension: every turn keeps exactly the ids the engine consumed
and produced, plus their logprobs, so the client can build Datums without any decode/re-encode.
A ``Turn`` maps onto tinker-cookbook's ``Transition`` (``ob`` = input_ids, ``ac`` = output_ids + logprobs), so
``tinker_cookbook.rl.data_processing.trajectory_to_data`` turns a session into Datums unchanged.
Sampling reuses ``MilesBackend.sample`` (router ``/generate`` with ``lora_path`` + ``lora_backfill_paths``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    """One recorded generation: exactly the ids the engine consumed and produced, plus their logprobs."""

    input_ids: list[int]
    output_ids: list[int]
    logprobs: list[float]
    finish_reason: str  # "stop" | "length"
    prefix_ok: bool  # input_ids starts with the previous turn's input_ids + output_ids
    sampling_params: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class TrajectorySession:
    """Per-trajectory state: the tenant that owns it, the sampler version it is pinned to, and its turns."""

    session_id: str
    tenant: str
    model_path: str | None = None  # tinker://M/sampler_weights/V; None samples the frozen base
    lora_name: str | None = None  # M@V, the adapter name engines know
    lora_dir: str | None = None  # snapshot directory engines backfill from
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


# --- rendering and request shaping ----------------------------------------------


def render_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any],
    tokenizer,
) -> list[int]:
    """Render OpenAI messages (+tools) to prompt ids with the base model's HF chat template, add_generation_prompt=True."""
    raise NotImplementedError


def encode_completion_prompt(prompt: str, tokenizer) -> list[int]:
    """Tokenize a raw /completions prompt string the way Tinker's completions endpoint does."""
    raise NotImplementedError


def reasoning_kwargs(reasoning_effort: str | float | None) -> dict[str, Any]:
    """Map Tinker's reasoning_effort ("none" | "minimal"…"xhigh" | float) to chat-template kwargs such as enable_thinking."""
    raise NotImplementedError


def build_sample_payload(prompt_ids: list[int], request: dict[str, Any]) -> dict[str, Any]:
    """Build the payload Tinker `sample` sends: prompt_tokens, num_samples, sampling_params (max_tokens, temperature, top_p, top_k, stop, seed), no prompt logprobs."""
    raise NotImplementedError


def compute_prefix_ok(previous: Turn | None, input_ids: list[int]) -> bool:
    """True when input_ids starts with previous.input_ids + previous.output_ids, or when there is no previous turn."""
    raise NotImplementedError


# --- response shaping ------------------------------------------------------------


def split_reasoning(text: str) -> tuple[str | None, str]:
    """Split a leading <think>…</think> block into (reasoning_content, content) for separate_reasoning=true."""
    raise NotImplementedError


def build_chat_response(
    request: dict[str, Any],
    output_ids: list[int],
    text: str,
    reasoning: str | None,
    finish_reason: str,
    prompt_len: int,
) -> dict[str, Any]:
    """Assemble an OpenAI ChatCompletion JSON: id, created, model echoed, choices[0].message, finish_reason, usage."""
    raise NotImplementedError


def build_completion_response(
    request: dict[str, Any], output_ids: list[int], text: str, finish_reason: str, prompt_len: int
) -> dict[str, Any]:
    """Assemble an OpenAI Completion JSON with choices[0].text and usage."""
    raise NotImplementedError


def to_sse(response: dict[str, Any]) -> bytes:
    """Fake streaming: one SSE chunk carrying the whole message, then data: [DONE]."""
    raise NotImplementedError


# --- collector -------------------------------------------------------------------


class TrajectoryCollector:
    """Owns the sessions; binds each one to a sampler version, samples through the gateway backend, records turns."""

    def __init__(self, backend, config, tokenizer) -> None:
        """Keep the MilesBackend (sample), the GatewayConfig (base_model, checkpoint_root, chat_template_kwargs, session_ttl_s) and the HF tokenizer."""
        self.backend = backend
        self.config = config
        self.tokenizer = tokenizer
        self.sessions: dict[str, TrajectorySession] = {}

    def bind(self, session_id: str, tenant: str, model: str | None) -> TrajectorySession:
        """Create or re-bind a session: resolve tinker://… with resolve_sampler_checkpoint (ownership included); base model when model is None or the base name; a bound session rejects a different model."""
        raise NotImplementedError

    def get(self, session_id: str, tenant: str | None) -> TrajectorySession:
        """Return the session when it exists and belongs to tenant (unknown id → 404-class error, other tenant → OwnershipError)."""
        raise NotImplementedError

    def delete(self, session_id: str, tenant: str) -> None:
        """Drop a session and its turns."""
        raise NotImplementedError

    async def chat(
        self, request: dict[str, Any], *, session_id: str | None = None, tenant: str | None = None
    ) -> dict[str, Any]:
        """Serve /chat/completions: bind or auto-register when session_id is given, render (TITO hook first), sample via backend.sample, record a Turn, return the OpenAI response."""
        raise NotImplementedError

    async def completions(
        self, request: dict[str, Any], *, session_id: str | None = None, tenant: str | None = None
    ) -> dict[str, Any]:
        """Serve /completions on the same sampling path, recording a Turn when session_id is given."""
        raise NotImplementedError

    def trajectory(self, session_id: str, tenant: str) -> dict[str, Any]:
        """Export {session_id, model_path, lora_name, turns: [...]} for the client to turn into Datums."""
        raise NotImplementedError

    def sweep(self, now: float | None = None) -> int:
        """Expire sessions idle longer than config.session_ttl_s and return how many were dropped; called from the gateway's lease sweeper."""
        raise NotImplementedError
