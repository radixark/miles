"""Metric-history subsystem for the CI regression gate.

* One import surface for consumers, re-exporting the :class:`MetricHistoryStore`
  contract and its record types (:class:`MetricSample`, :class:`RunIdentity`,
  :class:`RunProvenance`).
* The two backends: offline :class:`SQLiteMetricHistoryStore` and hosted
  :class:`NeonMetricHistoryStore`, plus :data:`NEON_DATABASE_URL_ENV`.
* The gate declaration marker :func:`register_ci_gate` and its parsed
  :class:`CiGateSpec`.
* Not re-exported: gate evaluation (:mod:`tests.ci.metric_history.gate`),
  step selection (:mod:`tests.ci.metric_history.selection`), constraints
  (:mod:`tests.ci.metric_history.constraints`) -- import those modules directly.
"""

from tests.ci.metric_history.register import CiGateSpec, register_ci_gate
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
    "CiGateSpec",
    "register_ci_gate",
]
