# doc-dev: docs/ci/03-metric-history-gate.md
"""Deferred Neon (hosted Postgres) backend for the metric-history store.

* An intentional placeholder: the hosted backend is not wired this round -- no
  Postgres driver is added to `requirements.txt` and nothing here opens a
  connection.
* The class exists so the gate can name its production backend and so the
  connection contract (the `NEON_DATABASE_URL` env var, provisioned
  out-of-band) is documented in one place.

Implementation notes for whoever lands the real driver:

* Add a driver (`psycopg[binary]` or `asyncpg`) to `requirements.txt`
  only when this is implemented -- not before.
* Read the DSN from the `NEON_DATABASE_URL` environment variable. Do not
  embed credentials and do not hardcode the host.
* Provision the schema and connection role out-of-band for now. The store's
  read/write path must remain DML-only: insert runs, read baselines, and mark
  runs untrusted.
* `recent_trusted_values` must use the authoritative baseline query verbatim
  (plain equality on `metric_key` / `steps_key` / `constraint_key` / `step`), so
  its results match
  :class:`~tests.ci.metric_history.storage.sqlite_store.SQLiteMetricHistoryStore`.
"""

from __future__ import annotations

from tests.ci.metric_history.storage.store import MetricHistoryStore, MetricSample, RunIdentity, RunProvenance

#: Name of the environment variable carrying the Neon Postgres DSN. The value
#: is provisioned out-of-band as a CI secret and is NOT wired into any workflow
#: in this round.
NEON_DATABASE_URL_ENV = "NEON_DATABASE_URL"

_NOT_IMPLEMENTED = "NeonMetricHistoryStore is not implemented yet; use SQLiteMetricHistoryStore offline."


class NeonMetricHistoryStore(MetricHistoryStore):
    def __init__(self, dsn: str | None = None):
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def write_run(
        self,
        identity: RunIdentity,
        provenance: RunProvenance,
        created_at: str,
        trusted: bool,
        values: list[MetricSample],
    ) -> str:
        raise NotImplementedError(_NOT_IMPLEMENTED)

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
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def mark_untrusted(
        self,
        *,
        run_id: str | None = None,
        github_run_id: int | None = None,
        commit_sha: str | None = None,
    ) -> int:
        raise NotImplementedError(_NOT_IMPLEMENTED)
