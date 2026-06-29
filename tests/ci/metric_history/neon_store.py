"""Deferred Neon (hosted Postgres) backend for the metric-history store.

This is an intentional placeholder. The hosted backend is not wired this round:
no Postgres driver is added to ``requirements.txt`` and nothing here opens a
connection. The class exists so the gate can name its production backend and so
the connection contract (the ``NEON_DATABASE_URL`` env var, provisioned
out-of-band) is documented in one place.

Implementation notes for whoever lands the real driver:

* Add a driver (``psycopg[binary]`` or ``asyncpg``) to ``requirements.txt``
  only when this is implemented -- not before.
* Read the DSN from the ``NEON_DATABASE_URL`` environment variable. Do not
  embed credentials and do not hardcode the host.
* Apply schema via the ``migrations/*.sql`` files; never issue DDL on the
  read/write path. The connection role granted by
  ``0002_least_privilege_role.sql`` lacks CREATE/ALTER/DROP, so a stray runtime
  DDL would fail loudly rather than mutate the schema.
* ``recent_trusted_values`` must use the authoritative baseline query verbatim
  (``sub_label IS NOT DISTINCT FROM %s``), so its results match
  :class:`~tests.ci.metric_history.sqlite_store.SQLiteMetricHistoryStore`.
"""

from __future__ import annotations

from tests.ci.metric_history.store import MetricHistoryStore, MetricSample, RunIdentity, RunProvenance

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
        sub_label: str | None,
        test_file_hash: str,
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
