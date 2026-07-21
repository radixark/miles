# doc-dev: docs/ci/03-metric-history-gate.md
"""Neon (hosted Postgres) backend for the metric-history store.

* The production backend: the CI regression gate writes runs here and reads its
  baselines from here.
* Mirrors :class:`~tests.ci.metric_history.storage.sqlite_store.SQLiteMetricHistoryStore`
  semantics exactly -- same write surface, same authoritative baseline query --
  so swapping backends changes no gate logic.
* `recent_trusted_values` uses the authoritative baseline query verbatim
  (plain equality on `metric_key` / `steps_key` / `constraint_key` / `step`),
  so its results match the SQLite backend's.
* The DSN comes from the `NEON_DATABASE_URL` environment variable (a CI
  secret, provisioned out-of-band) unless one is passed explicitly. Credentials
  are never embedded and the host is never hardcoded.
* Every operation opens a fresh connection and closes it before returning;
  construction only resolves the DSN. The store lives for a whole CI suite
  while individual e2e tests run tens of minutes, so a connection cached at
  construction would outlive Neon's pooler idle timeout / compute autosuspend
  and every write after the first idle gap would fail on a dead socket.
* The schema (the two tables and the application role) is provisioned
  out-of-band, outside this repo. This module stays DML-only -- insert runs,
  read baselines, mark runs untrusted -- and never issues DDL on the
  read/write path.

Caveats:

* Keep the `psycopg` import lazy inside :meth:`_connect`, never at module
  load. The package `__init__` imports this module unconditionally, so a
  top-level driver import would make the whole metric-history package
  un-importable where the driver is not installed (offline test runs, the
  SQLite-only path); deferring it also lets the live-Postgres smoke test skip
  cleanly when the driver/DSN are absent.
"""

from __future__ import annotations

import os
import uuid

from tests.ci.metric_history.storage.store import (
    MetricHistoryStore,
    MetricSample,
    RunIdentity,
    RunProvenance,
    validate_finite_values,
)

#: Name of the environment variable carrying the Neon Postgres DSN. The value
#: is provisioned out-of-band as a CI secret.
NEON_DATABASE_URL_ENV = "NEON_DATABASE_URL"

# Mirrors the authoritative baseline query (see sqlite_store._BASELINE_SQL),
# translated to psycopg's %s placeholders and Postgres' boolean literal.
# Every coordinate column is NOT NULL, so the whole match is plain equality.
_BASELINE_SQL = """
SELECT mv.value
FROM metric_values mv
JOIN runs r USING (run_id)
WHERE r.test_path = %s
  AND r.backend = %s
  AND r.suite = %s
  AND mv.metric_key = %s
  AND mv.steps_key = %s
  AND mv.constraint_key = %s
  AND mv.step = %s
  AND r.trusted = true
ORDER BY r.created_at DESC
LIMIT %s
"""

_INSERT_RUN_SQL = """
INSERT INTO runs (
    run_id, test_path, backend, suite,
    commit_sha, pr_number, github_run_id, github_run_attempt,
    event_name, ref, created_at, trusted
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_METRIC_SQL = (
    "INSERT INTO metric_values (run_id, metric_key, steps_key, constraint_key, step, value)"
    " VALUES (%s, %s, %s, %s, %s, %s)"
)


class NeonMetricHistoryStore(MetricHistoryStore):
    def __init__(self, dsn: str | None = None):
        if dsn is None:
            dsn = os.environ.get(NEON_DATABASE_URL_ENV)
            if not dsn:
                raise RuntimeError(
                    f"No DSN passed and {NEON_DATABASE_URL_ENV} is unset; cannot connect to Neon. "
                    "Set the environment variable to the Neon Postgres DSN, or pass dsn= explicitly."
                )
        self._dsn = dsn

    def _connect(self):
        # Lazy import: see module docstring. autocommit stays off so write_run
        # controls its own transaction boundary explicitly.
        import psycopg

        return psycopg.connect(self._dsn)

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
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    _INSERT_RUN_SQL,
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
                        trusted,
                    ),
                )
                cur.executemany(
                    _INSERT_METRIC_SQL,
                    [(run_id, s.metric_key, s.steps_key, s.constraint_key, s.step, s.value) for s in values],
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
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
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    _BASELINE_SQL,
                    (test_path, backend, suite, metric_key, steps_key, constraint_key, step, limit),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
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
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE runs SET trusted = false WHERE {column} = %s AND trusted = true",
                    (value,),
                )
                affected = cur.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return affected

    def close(self) -> None:
        # No connection outlives an operation; kept for interface symmetry
        # with the SQLite store so callers can close either unconditionally.
        pass
