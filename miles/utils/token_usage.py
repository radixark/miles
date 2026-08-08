"""Per-adapter token metering: counting only — no rates, no cost computation.

Every adapter registration carries one monotonic :class:`TokenUsage` meter,
split by *class* (prefill / cached_prefill / sample / scoring_prefill / train
/ train_forward — the axes an external rate card would price) and reported
from three *sources* (rollout worker, trainer commit, external inference).
Pricing and invoicing live in an external billing backend that consumes the
usage API / journal; nothing in miles stores rates.

Counting policy (see the multi-LoRA API design doc, §4.2):

- Count compute where it burns: generation tokens are counted at engine
  response receipt, so stale-dropped / filter-dropped / aborted-partial
  groups are all counted (the GPU work happened); the ``sample_tokens_*``
  detail counters itemize those outcomes as informational subsets.
- Train tokens are full-sequence tokens per *committed* train call (banked
  via the registry's exactly-once ``mark_batch_trained``), never at optimizer
  step time.
- Meters key on ``(adapter name, registration_id)`` so a re-registered name
  never inherits a previous tenant's counters.

Cross-process reporting uses cumulative-gauge snapshots keyed by a reporter
*incarnation* (minted per accumulator instance): the receiver element-wise
``max()``-merges per incarnation and sums across incarnations, which makes
at-least-once RPC delivery idempotent and reporter restarts lossless.

This module must stay import-light (no torch / ray / fastapi) so the
accounting logic is unit-testable anywhere.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable

logger = logging.getLogger(__name__)

__all__ = [
    "METER_VERSION",
    "ROLLOUT_FIELDS",
    "TRAIN_FIELDS",
    "RolloutTokenMeter",
    "TokenUsage",
    "train_forward_pass_count",
]

METER_VERSION = 1


@dataclass
class TokenUsage:
    """Monotonic token counters for one adapter registration."""

    # Rollout side (engine compute, reported by the rollout worker).
    prefill_tokens: int = 0  # uncached prompt tokens (prompt - cached) through generation prefill
    cached_prefill_tokens: int = 0  # radix-cache hits (adapter+version-scoped via extra_key)
    sample_tokens: int = 0  # engine-reported decode tokens (meta_info completion_tokens)
    scoring_prefill_tokens: int = 0  # prefill-logprob recompute passes (clean prefill, cache flushed)
    # Informational subsets of sample_tokens (already counted above; never price these).
    sample_tokens_trained: int = 0
    sample_tokens_dropped_stale: int = 0
    sample_tokens_dropped_filter: int = 0
    sample_tokens_aborted: int = 0
    sample_tokens_dropped_retired: int = 0  # buffered-but-untrained groups discarded at retire
    # Train side (trainer compute, banked only on committed train batches).
    train_tokens: int = 0  # full-sequence tokens through forward+backward
    train_forward_tokens: int = 0  # forward-only extra passes (ref KL / OPD teacher / logprob recompute)
    optimizer_steps: int = 0  # informational; O(params), not a token count

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TokenUsage":
        names = {f.name for f in fields(TokenUsage)}
        return TokenUsage(**{k: int(v) for k, v in data.items() if k in names})

    def add_inplace(self, other: dict[str, int]) -> None:
        for key, value in other.items():
            if hasattr(self, key):
                setattr(self, key, getattr(self, key) + int(value))


ROLLOUT_FIELDS = (
    "prefill_tokens",
    "cached_prefill_tokens",
    "sample_tokens",
    "scoring_prefill_tokens",
    "sample_tokens_trained",
    "sample_tokens_dropped_stale",
    "sample_tokens_dropped_filter",
    "sample_tokens_aborted",
    "sample_tokens_dropped_retired",
)

TRAIN_FIELDS = ("train_tokens", "train_forward_tokens", "optimizer_steps")


def max_merge_counters(previous: dict[str, int] | None, incoming: dict[str, int]) -> dict[str, int]:
    """Element-wise max of two cumulative counter dicts (idempotent under
    replayed or reordered snapshots)."""
    if previous is None:
        return {key: int(incoming.get(key, 0)) for key in ROLLOUT_FIELDS}
    return {key: max(int(previous.get(key, 0)), int(incoming.get(key, 0))) for key in ROLLOUT_FIELDS}


class RolloutTokenMeter:
    """Cumulative per-(name, registration_id) rollout counters.

    Lives on the rollout worker; the *incarnation* is minted per meter
    instance (not per process) because a dead producer's worker can be
    re-created inside the same process — a process-scoped incarnation would
    ship a regressed cumulative snapshot that max-merge silently absorbs.

    Thread-safety: the producer thread records generation usage while the
    rollout-manager loop records scoring usage and exports snapshots, so
    every access takes the lock.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.incarnation = uuid.uuid4().hex
        self._usage: dict[tuple[str, str], TokenUsage] = {}
        # Adapters already warned about for non-reporting generate fns.
        self._zero_prompt_warned: set[str] = set()

    def _get_locked(self, name: str, registration_id: str) -> TokenUsage:
        key = (name, registration_id)
        usage = self._usage.get(key)
        if usage is None:
            usage = self._usage[key] = TokenUsage()
        return usage

    def record_generation(self, name: str, registration_id: str | None, samples: Iterable[Any]) -> None:
        """Count one generated group at engine-consumption time.

        ``samples`` are miles Samples (duck-typed): ``prefix_cache_info``
        carries per-call prompt/cached sums and ``engine_completion_tokens``
        the engine-reported decode total. Must run BEFORE ``reset_for_retry``
        wipes those counters. Groups without a registration stamp burned no
        engine compute (they are aborted before POST) and are skipped.
        """
        if registration_id is None:
            return
        prompt = cached = completion = 0
        completed_with_zero_prompt = False
        for sample in samples:
            info = sample.prefix_cache_info
            prompt += info.total_prompt_tokens
            cached += info.cached_tokens
            completion += sample.engine_completion_tokens
            if info.total_prompt_tokens == 0 and getattr(getattr(sample, "status", None), "value", None) == "completed":
                completed_with_zero_prompt = True
        if completed_with_zero_prompt and name not in self._zero_prompt_warned:
            # Defensive check for custom generate fns that skip the
            # Sample.update_from_meta_info fold-in: they zero their own meter.
            self._zero_prompt_warned.add(name)
            logger.warning(
                f"Adapter '{name}': COMPLETED sample arrived with zero engine-reported prompt tokens; "
                "its generate fn likely skips Sample.update_from_meta_info, under-reporting token usage"
            )
        with self.lock:
            usage = self._get_locked(name, registration_id)
            usage.prefill_tokens += max(0, prompt - cached)
            usage.cached_prefill_tokens += cached
            usage.sample_tokens += completion

    def record_detail(self, name: str, registration_id: str | None, detail_field: str, completion_tokens: int) -> None:
        """Attribute already-counted sample tokens to an outcome bucket
        (``sample_tokens_trained`` / ``_dropped_stale`` / ``_dropped_filter``
        / ``_aborted``). Informational: parents were counted at generation."""
        assert detail_field in ROLLOUT_FIELDS and detail_field.startswith("sample_tokens_"), detail_field
        if registration_id is None or completion_tokens <= 0:
            return
        with self.lock:
            usage = self._get_locked(name, registration_id)
            setattr(usage, detail_field, getattr(usage, detail_field) + int(completion_tokens))

    def record_scoring(self, name: str, registration_id: str | None, prefill_tokens: int) -> None:
        """Count a prefill-logprob scoring pass (full clean prefill of the
        sample's prompt+response tokens)."""
        if registration_id is None or prefill_tokens <= 0:
            return
        with self.lock:
            self._get_locked(name, registration_id).scoring_prefill_tokens += int(prefill_tokens)

    def snapshot_entries(self) -> list[dict[str, Any]]:
        """Cumulative snapshot for the controller: a list of
        ``{"name", "registration_id", "counters"}`` entries. Safe to send
        at-least-once; the receiver max-merges per incarnation."""
        with self.lock:
            return [
                {
                    "name": name,
                    "registration_id": registration_id,
                    "counters": {key: getattr(usage, key) for key in ROLLOUT_FIELDS},
                }
                for (name, registration_id), usage in self._usage.items()
            ]

    def prune(self, registration_ids: Iterable[str]) -> None:
        """Drop meters the controller acknowledged as finalized: their gauges
        can never be merged again, and re-shipping them every flush would spam
        late-snapshot audit events forever."""
        finalized = set(registration_ids)
        if not finalized:
            return
        with self.lock:
            for key in [k for k in self._usage if k[1] in finalized]:
                self._usage.pop(key)


def train_forward_pass_count(args: Any) -> int:
    """Forward-only trainer passes per train call, from run flags.

    Approximates the actor's pass structure (actor.py train): the actor
    log-prob recompute runs unless --use-rollout-logprobs (and no mismatch
    metrics); a ref-model forward runs under KL; a teacher forward under OPD.
    Multiplied by the batch's train tokens to fill ``train_forward_tokens``.
    """
    if not getattr(args, "compute_advantages_and_returns", True):
        return 0
    passes = 0
    if not getattr(args, "use_rollout_logprobs", False) or getattr(args, "get_mismatch_metrics", False):
        passes += 1
    if getattr(args, "kl_coef", 0) or getattr(args, "use_kl_loss", False):
        passes += 1
    if getattr(args, "use_opd", False):
        passes += 1
    return passes
