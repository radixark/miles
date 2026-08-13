# doc-dev: docs/ci/04-runtime-est-time.md
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime

from tests.ci.ci_register import HWBackend
from tests.ci.metric_history import NEON_DATABASE_URL_ENV


@dataclass(frozen=True)
class RuntimeAttempt:
    test_path: str
    backend: str
    suite: str
    test_attempt: int
    status: str
    elapsed_seconds: float
    estimated_seconds: float


@dataclass(frozen=True)
class RuntimeProvenance:
    commit_sha: str
    github_run_id: int
    github_run_attempt: int
    event_name: str
    ref: str


@dataclass(frozen=True)
class RuntimeSample:
    test_path: str
    backend: str
    suite: str
    elapsed_seconds: float
    github_run_id: int
    github_run_attempt: int
    recorded_at: datetime


_VALID_STATUSES = frozenset({"PASS", "FAIL", "TIMEOUT"})

_INSERT_SQL = """
INSERT INTO ci_test_runtime_attempts (
    test_path, backend, suite, test_attempt,
    status, elapsed_seconds, estimated_seconds,
    commit_sha, github_run_id, github_run_attempt, event_name, git_ref
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (
    github_run_id, github_run_attempt,
    test_path, backend, suite, test_attempt
) DO UPDATE SET elapsed_seconds = CASE
    WHEN ci_test_runtime_attempts.status IS NOT DISTINCT FROM EXCLUDED.status
      AND ci_test_runtime_attempts.elapsed_seconds IS NOT DISTINCT FROM EXCLUDED.elapsed_seconds
      AND ci_test_runtime_attempts.estimated_seconds IS NOT DISTINCT FROM EXCLUDED.estimated_seconds
      AND ci_test_runtime_attempts.commit_sha IS NOT DISTINCT FROM EXCLUDED.commit_sha
      AND ci_test_runtime_attempts.event_name IS NOT DISTINCT FROM EXCLUDED.event_name
      AND ci_test_runtime_attempts.git_ref IS NOT DISTINCT FROM EXCLUDED.git_ref
    THEN ci_test_runtime_attempts.elapsed_seconds
    ELSE NULL
END
"""

_RECENT_PASS_SQL = """
WITH ranked AS (
    SELECT
        test_path,
        backend,
        suite,
        elapsed_seconds,
        github_run_id,
        github_run_attempt,
        recorded_at,
        ROW_NUMBER() OVER (
            PARTITION BY test_path, backend, suite
            ORDER BY recorded_at DESC, github_run_id DESC, github_run_attempt DESC, test_attempt DESC
        ) AS sample_rank
    FROM ci_test_runtime_attempts
    WHERE status = 'PASS'
      AND event_name = 'schedule'
      AND git_ref = 'refs/heads/main'
      AND recorded_at >= %s
      AND recorded_at < %s
)
SELECT
    test_path,
    backend,
    suite,
    elapsed_seconds,
    github_run_id,
    github_run_attempt,
    recorded_at
FROM ranked
WHERE sample_rank <= %s
ORDER BY test_path, backend, suite, sample_rank
"""


def _validate_attempt(attempt: RuntimeAttempt) -> None:
    if attempt.status not in _VALID_STATUSES:
        raise ValueError(f"unsupported runtime status: {attempt.status!r}")
    if not math.isfinite(attempt.elapsed_seconds) or attempt.elapsed_seconds < 0:
        raise ValueError(f"invalid elapsed_seconds: {attempt.elapsed_seconds!r}")
    if not math.isfinite(attempt.estimated_seconds) or attempt.estimated_seconds <= 0:
        raise ValueError(f"invalid estimated_seconds: {attempt.estimated_seconds!r}")
    if attempt.test_attempt < 1:
        raise ValueError(f"invalid test_attempt: {attempt.test_attempt!r}")


class NeonRuntimeHistoryStore:
    def __init__(self, dsn: str | None = None):
        self._dsn = dsn or os.environ.get(NEON_DATABASE_URL_ENV)
        if not self._dsn:
            raise RuntimeError(f"{NEON_DATABASE_URL_ENV} is required for CI runtime history")

    def _connect(self):
        import psycopg

        return psycopg.connect(self._dsn)

    def write_attempts(self, provenance: RuntimeProvenance, attempts: list[RuntimeAttempt]) -> None:
        unique_attempts: dict[tuple[int, int, str, str, str, int], RuntimeAttempt] = {}
        for attempt in attempts:
            _validate_attempt(attempt)
            key = (
                provenance.github_run_id,
                provenance.github_run_attempt,
                attempt.test_path,
                attempt.backend,
                attempt.suite,
                attempt.test_attempt,
            )
            existing = unique_attempts.get(key)
            if existing is not None and existing != attempt:
                raise ValueError(f"conflicting runtime attempts for idempotency key {key!r}")
            unique_attempts[key] = attempt
        if not unique_attempts:
            return

        rows = [
            (
                attempt.test_path,
                attempt.backend,
                attempt.suite,
                attempt.test_attempt,
                attempt.status,
                attempt.elapsed_seconds,
                attempt.estimated_seconds,
                provenance.commit_sha,
                provenance.github_run_id,
                provenance.github_run_attempt,
                provenance.event_name,
                provenance.ref,
            )
            for attempt in unique_attempts.values()
        ]
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.executemany(_INSERT_SQL, rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def recent_successful_attempts(self, cutoff: datetime, before: datetime, limit: int) -> list[RuntimeSample]:
        if limit < 1:
            raise ValueError(f"limit must be positive, got {limit}")
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(_RECENT_PASS_SQL, (cutoff, before, limit))
                rows = cur.fetchall()
        finally:
            conn.close()
        return [RuntimeSample(*row) for row in rows]


def build_runtime_store_from_env(hw: HWBackend) -> NeonRuntimeHistoryStore | None:
    if hw != HWBackend.CUDA:
        return None
    if os.environ.get("GITHUB_EVENT_NAME") != "schedule":
        return None
    if os.environ.get("GITHUB_REF") != "refs/heads/main":
        return None
    if not os.environ.get(NEON_DATABASE_URL_ENV):
        return None
    return NeonRuntimeHistoryStore()


def runtime_provenance_from_env() -> RuntimeProvenance:
    return RuntimeProvenance(
        commit_sha=os.environ["GITHUB_SHA"],
        github_run_id=int(os.environ["GITHUB_RUN_ID"]),
        github_run_attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]),
        event_name=os.environ["GITHUB_EVENT_NAME"],
        ref=os.environ["GITHUB_REF"],
    )
