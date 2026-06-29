"""Metric-history storage layer for the CI regression gate.

Public surface: the :class:`MetricHistoryStore` contract, its record types, the
offline :class:`SQLiteMetricHistoryStore`, and the deferred
:class:`NeonMetricHistoryStore`.
"""

from tests.ci.metric_history.neon_store import NEON_DATABASE_URL_ENV, NeonMetricHistoryStore
from tests.ci.metric_history.sqlite_store import SQLiteMetricHistoryStore
from tests.ci.metric_history.store import MetricHistoryStore, MetricSample, RunIdentity, RunProvenance

__all__ = [
    "MetricHistoryStore",
    "MetricSample",
    "RunIdentity",
    "RunProvenance",
    "SQLiteMetricHistoryStore",
    "NeonMetricHistoryStore",
    "NEON_DATABASE_URL_ENV",
]
