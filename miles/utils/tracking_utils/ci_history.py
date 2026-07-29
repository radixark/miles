"""CI metric-history collection backend.

* Captures the fixed :data:`TARGET_METRIC_KEYS` whitelist of training/rollout
  metrics from the live process into one JSONL record file; keys outside the
  whitelist are never recorded.
* Collection is on only when the harness sets :data:`RECORD_DIR_ENV`
  (`MILES_CI_GATE_RECORD_DIR`); without it `init()` leaves the backend a
  no-op.
* The record is a pure process-to-harness handoff: raw unreduced
  `{"metric": key, "series": [[step, value], ...]}` lines, no identity (no
  test path), nothing read from wandb, nothing written to any cloud. Reduction
  and gating happen in a later step that consumes these records.
* Every `log()` atomically rewrites the whole file as a fresh snapshot (temp
  file + rename), not an append; a process that never calls `finish()` still
  leaves a complete record.
* Each backend instance owns a distinct file keyed by pid + a fresh uuid, so
  concurrent processes never clobber each other's records.

Caveats:

* Capture never blocks the run on metric content: non-finite values (NaN/±Inf)
  are recorded faithfully, serialized as the string markers `"NaN"` /
  `"Infinity"` / `"-Infinity"` so every line stays strict JSON (the
  gate-side reader decodes them).
* A non-numeric value (bool/str/...) at a whitelisted key raises `TypeError`
  in the logging process -- an authoring error fails loudly instead of being
  dropped.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import uuid
from typing import Any

from .base import TrackingBackend

logger = logging.getLogger(__name__)

# Metric keys captured for the history gate, plus the step key carried alongside
# each. The training keys are logged from the Ray training actor with step_key
# "train/step"; rollout/raw_reward is logged with step_key "rollout/step". Keys
# are the actor (role="actor") form with no role prefix.
TARGET_METRIC_KEYS: tuple[str, ...] = (
    "train/grad_norm",
    "train/ppo_kl",
    "train/train_rollout_logprob_abs_diff",
    "train/train_rollout_kl",
    "rollout/raw_reward",
)

# Env var naming the directory the harness assigns for this run's records.
RECORD_DIR_ENV = "MILES_CI_GATE_RECORD_DIR"


def _encode_value(value: float) -> float | str:
    # Bare NaN/Infinity from json.dumps is not strict JSON; markers keep the
    # record parseable by any reader.
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


class CiHistoryBackend(TrackingBackend):
    """Accumulate target metrics in-process and persist the raw series to disk.

    Each initialized backend instance owns a distinct JSONL file keyed by a
    fresh process-local id, so separate instances do not clobber each other.

    Some logging processes may not call `finish()`, so every `log()` persists
    a fresh snapshot of the full accumulated series. The file is the latest
    snapshot regardless of whether `finish()` ever fires.
    """

    def __init__(self) -> None:
        self._series: dict[str, list[tuple[int | None, float]]] = {}
        self._lock = threading.Lock()
        self._record_dir: str | None = None
        self._record_path: str | None = None

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        record_dir = os.environ.get(RECORD_DIR_ENV)
        if not record_dir:
            # No harness-assigned directory: nothing to collect into. Leaving
            # _record_dir None makes log()/finish() no-ops.
            logger.info("%s not set; CI history collection disabled.", RECORD_DIR_ENV)
            return
        os.makedirs(record_dir, exist_ok=True)
        self._record_dir = record_dir
        process_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        self._record_path = os.path.join(record_dir, f"{process_id}.jsonl")

    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs) -> None:
        if self._record_dir is None:
            return
        with self._lock:
            captured: list[tuple[str, float]] = []
            for key in TARGET_METRIC_KEYS:
                if key not in metrics:
                    continue
                value = metrics[key]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    message = f"CI history metric {key!r} must be int or float, got {type(value).__name__}"
                    logger.error("%s", message)
                    raise TypeError(message)
                captured.append((key, float(value)))
            for key, value in captured:
                self._series.setdefault(key, []).append((step, value))
            if captured:
                self._write_snapshot_locked()

    def finish(self) -> None:
        if self._record_dir is None:
            return
        with self._lock:
            self._write_snapshot_locked()

    def _write_snapshot_locked(self) -> None:
        # Rewrite the whole record file with the current series. Writing to a
        # temp file and renaming makes each snapshot atomic, so a concurrent reader
        # never sees a half-written record.
        assert self._record_path is not None
        tmp_path = f"{self._record_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for key, points in self._series.items():
                line = {
                    "metric": key,
                    "series": [[step, _encode_value(value)] for step, value in points],
                }
                f.write(json.dumps(line, allow_nan=False) + "\n")
        os.replace(tmp_path, self._record_path)
