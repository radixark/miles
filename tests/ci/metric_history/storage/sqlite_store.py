# doc-dev: docs/ci/03-metric-history-gate.md
"""SQLite-backed :class:`MetricHistoryStore` for offline use and tests.

* No network dependency; the store runs entirely in-process. An in-memory
  database (`:memory:`) makes it a drop-in fixture for unit tests.
* Query and write semantics mirror the hosted Postgres backend
  (:class:`NeonMetricHistoryStore` in neon_store.py), so tests exercising this
  implementation validate the contract the gate relies on in production —
  including the authoritative baseline query's plain-equality match on the
  `(metric_key, steps_key, constraint_key, step)` coordinate.
* This module owns a small local schema literal for tests; production Postgres
  schema setup is out-of-band (provisioned outside the repo).
* Schema is applied at construction (`apply_schema`), never on the
  read/write path.
"""

from __future__ import annotations

import sqlite3
import uuid

from tests.ci.metric_history.storage.store import (
    MetricHistoryStore,
    MetricSample,
    RunIdentity,
    RunProvenance,
    validate_finite_values,
)

# The local/offline schema: the two tables and the composite baseline index.
# SQLite stores the same logical columns with its dynamic typing.
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    test_path           TEXT NOT NULL,
    backend             TEXT NOT NULL,
    suite               TEXT NOT NULL,
    commit_sha          TEXT NOT NULL,
    pr_number           INTEGER,
    github_run_id       INTEGER,
    github_run_attempt  INTEGER,
    event_name          TEXT,
    ref                 TEXT,
    created_at          TEXT NOT NULL,
    trusted             INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS metric_values (
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    metric_key      TEXT NOT NULL,
    steps_key       TEXT NOT NULL,
    constraint_key  TEXT NOT NULL,
    step            INTEGER NOT NULL,
    value           REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS runs_baseline_idx
    ON runs (test_path, backend, suite, trusted, created_at DESC);
"""

# Mirrors the authoritative baseline query: every coordinate column is
# NOT NULL, so the whole match is plain equality.
_BASELINE_SQL = """
SELECT mv.value
FROM metric_values mv
JOIN runs r USING (run_id)
WHERE r.test_path = ?
  AND r.backend = ?
  AND r.suite = ?
  AND mv.metric_key = ?
  AND mv.steps_key = ?
  AND mv.constraint_key = ?
  AND mv.step = ?
  AND r.trusted = 1
ORDER BY r.created_at DESC
LIMIT ?
"""


class SQLiteMetricHistoryStore(MetricHistoryStore):
    def __init__(self, database: str = ":memory:"):
        # check_same_thread is left at its default; the store is meant for the
        # single-threaded CI gate process and the test suite.
        self._conn = sqlite3.connect(database)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.apply_schema()

    def apply_schema(self) -> None:
        """Create the tables and index if absent.

        This runs once at construction (or when a caller resets a database),
        never on the read/write path. This method keeps the SQLite fixture
        self-contained.
        """
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def write_run(
        self,
        identity: RunIdentity,
        provenance: RunProvenance,
        created_at: str,
        trusted: bool,
        values: list[MetricSample],
    ) -> str:
        validate_finite_values(values)
        run_id = uuid.uuid4().hex
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO runs (
                    run_id, test_path, backend, suite,
                    commit_sha, pr_number, github_run_id, github_run_attempt,
                    event_name, ref, created_at, trusted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    identity.test_path,
                    identity.backend,
                    identity.suite,
                    provenance.commit_sha,
                    provenance.pr_number,
                    provenance.github_run_id,
                    provenance.github_run_attempt,
                    provenance.event_name,
                    provenance.ref,
                    created_at,
                    1 if trusted else 0,
                ),
            )
            self._conn.executemany(
                "INSERT INTO metric_values (run_id, metric_key, steps_key, constraint_key, step, value)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(run_id, s.metric_key, s.steps_key, s.constraint_key, s.step, s.value) for s in values],
            )
        return run_id

    def recent_trusted_values(
        self,
        test_path: str,
        backend: str,
        suite: str,
        metric_key: str,
        steps_key: str,
        constraint_key: str,
        step: int,
        limit: int,
    ) -> list[float]:
        rows = self._conn.execute(
            _BASELINE_SQL,
            (test_path, backend, suite, metric_key, steps_key, constraint_key, step, limit),
        ).fetchall()
        return [row[0] for row in rows]

    def mark_untrusted(
        self,
        *,
        run_id: str | None = None,
        github_run_id: int | None = None,
        commit_sha: str | None = None,
    ) -> int:
        keys = {"run_id": run_id, "github_run_id": github_run_id, "commit_sha": commit_sha}
        provided = {name: value for name, value in keys.items() if value is not None}
        if len(provided) != 1:
            raise ValueError(
                "mark_untrusted requires exactly one of run_id / github_run_id / commit_sha; "
                f"got {sorted(provided)}"
            )
        column, value = next(iter(provided.items()))
        with self._conn:
            cursor = self._conn.execute(
                f"UPDATE runs SET trusted = 0 WHERE {column} = ? AND trusted = 1",
                (value,),
            )
            return cursor.rowcount

    def close(self) -> None:
        self._conn.close()
