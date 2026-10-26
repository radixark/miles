"""Latest health of each agent sandbox, for the dashboard's Sandboxes view.

The collector keeps one row per sandbox rather than an event log: finished
sandboxes are evicted after a while and the row count is capped, so memory
stays bounded however long the run. The rows are periodically written to a
small JSON snapshot, which the dashboard server reads.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

SNAPSHOT_FILENAME = "sandboxes.json"
SandboxStatus = Literal["healthy", "error", "finished"]
STATUSES: tuple[str, ...] = ("healthy", "error", "finished")


@dataclass
class SandboxReport:
    """One ``report_sandbox`` call."""

    ts: float
    sandbox_key: str
    status: str
    task: str | None = None
    provider: str | None = None
    sandbox_id: str | None = None
    error: str | None = None


@dataclass
class SandboxRow:
    """What is known about one sandbox: its labels, status and last report time."""

    sandbox_key: str
    status: str
    first_seen_ts: float
    last_exec_ts: float
    task: str | None = None
    provider: str | None = None
    sandbox_id: str | None = None
    last_error: str | None = None
    finished: bool = False


class SandboxHealth:
    """Bounded table of sandbox rows, written as a snapshot to ``path``."""

    def __init__(self, path: Path, *, finished_ttl_s: float = 600.0, max_rows: int = 10_000):
        self.path = path
        self.finished_ttl_s = finished_ttl_s
        self.max_rows = max_rows
        self.rows: dict[str, SandboxRow] = {}
        self._dirty = False

    def update(self, report: SandboxReport) -> None:
        row = self.rows.get(report.sandbox_key)
        if row is None:
            row = self.rows[report.sandbox_key] = SandboxRow(
                sandbox_key=report.sandbox_key, status=report.status, first_seen_ts=report.ts, last_exec_ts=report.ts
            )
        for label in ("task", "provider", "sandbox_id"):
            if getattr(report, label) is not None:  # config-access-exempt: fixed report labels
                setattr(row, label, getattr(report, label))  # config-access-exempt: fixed report labels
        if report.error is not None:
            row.last_error = report.error
        # reports relayed from a subproc agent call can arrive late; an older one only fills labels
        if report.ts >= row.last_exec_ts:
            row.last_exec_ts = report.ts
            row.finished = report.status == "finished"
            # a failed sandbox is torn down right after its error; keep it counted as an error
            if not (row.finished and row.status == "error"):
                row.status = report.status
        row.first_seen_ts = min(row.first_seen_ts, report.ts)
        self._dirty = True
        if len(self.rows) > self.max_rows:
            self._evict(now=report.ts)

    def _evict(self, *, now: float) -> None:
        for key, row in list(self.rows.items()):
            if row.finished and now - row.last_exec_ts > self.finished_ttl_s:
                del self.rows[key]
                self._dirty = True
        excess = len(self.rows) - self.max_rows
        if excess > 0:
            # finished rows go first, then the longest-silent ones
            by_priority = sorted(self.rows.values(), key=lambda r: (not r.finished, r.last_exec_ts))
            for row in by_priority[:excess]:
                del self.rows[row.sandbox_key]
            self._dirty = True

    def write_snapshot(self, *, now: float) -> None:
        """Atomically overwrite the snapshot file; its ``ts`` is when the run was last seen alive."""
        self._evict(now=now)
        if not (self._dirty or self.rows):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(dict(ts=now, sandboxes=[asdict(row) for row in self.rows.values()])))
        os.replace(tmp, self.path)
        self._dirty = False


def read_snapshot(path: Path) -> tuple[float, list[dict[str, Any]]] | None:
    """The snapshot's write time and rows, or None if no sandbox has reported yet."""
    if not path.is_file():
        return None
    snapshot = json.loads(path.read_text())
    return snapshot["ts"], snapshot["sandboxes"]


def sandbox_summary(rows: list[dict[str, Any]], *, now: float, stale_after_s: float) -> dict[str, Any]:
    """Counts per health and the rows, errors first, then stale ones.

    A healthy sandbox is stale when it has not reported for over ``stale_after_s``."""
    counts = dict.fromkeys((*STATUSES, "stale"), 0)
    out = []
    for row in rows:
        age_s = now - row["last_exec_ts"]
        stale = row["status"] == "healthy" and age_s > stale_after_s
        counts["stale" if stale else row["status"]] += 1
        out.append(dict(row, age_s=age_s, stale=stale))
    rank = {"error": 0, "stale": 1, "healthy": 2, "finished": 3}
    out.sort(key=lambda r: (rank["stale" if r["stale"] else r["status"]], -r["age_s"]))
    return dict(counts=counts, sandboxes=out, now=now)
