"""Storage layer for the CI metric-history regression gate.

* Re-exports the :class:`MetricHistoryStore` contract and its record types
  (:class:`MetricSample`, :class:`RunIdentity`, :class:`RunProvenance`).
* Re-exports the two backends -- offline :class:`SQLiteMetricHistoryStore` and
  the deferred :class:`NeonMetricHistoryStore` placeholder -- plus
  :data:`NEON_DATABASE_URL_ENV`.
"""

from tests.ci.metric_history.storage.neon_store import NEON_DATABASE_URL_ENV, NeonMetricHistoryStore
from tests.ci.metric_history.storage.sqlite_store import SQLiteMetricHistoryStore
from tests.ci.metric_history.storage.store import MetricHistoryStore, MetricSample, RunIdentity, RunProvenance

__all__ = [
    "MetricHistoryStore",
    "MetricSample",
    "RunIdentity",
    "RunProvenance",
    "SQLiteMetricHistoryStore",
    "NeonMetricHistoryStore",
    "NEON_DATABASE_URL_ENV",
]
