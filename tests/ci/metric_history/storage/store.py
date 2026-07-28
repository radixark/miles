# doc-dev: docs/ci/03-metric-history-gate.md
"""Storage contract for the CI metric-history regression gate.

* The gate compares a candidate run's metrics against a baseline assembled
  from the most recent *trusted* runs that share the same test identity.
* Two normalized tables back the contract. `runs` -- one row per CI
  execution of a test, holding identity (test_path, backend, suite),
  provenance (commit_sha, pr_number, github_run_id, github_run_attempt,
  event_name, ref), the `created_at` timestamp, and the run-level
  `trusted` flag.
* `metric_values` -- one row per comparison-coordinate value a run produced:
  `(metric_key, steps_key, constraint_key, step, value)`, keyed back to `runs`
  by `run_id`.
* `trusted` lives on the run, not on the metric: a run is trusted as a
  whole or not at all, so revoking trust drops every metric the run
  contributed in one operation.
* This module defines that storage contract and nothing else: it does not
  decide what counts as a regression, does not read the candidate run, and
  does not talk to CI.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSample:
    """One comparison-coordinate value a run contributed.

    The coordinate is the declaring gate's literal content plus the point:
    `steps_key` / `constraint_key` are canonical JSON of the declaration's raw
    `steps` / `constraint` literals (serialization pinned by
    `register._canonical_key`); `step` is the point the value came from -- step `k`, or `-1`
    for a whole-series reduction (e.g. `steps="last"`), which must key on
    a constant rather than the step it happened to land on, or its history
    would fragment across runs of different lengths.
    """

    metric_key: str
    steps_key: str
    constraint_key: str
    step: int
    value: float


@dataclass(frozen=True)
class RunIdentity:
    """The fields that decide whether two runs share a baseline.

    A baseline query is scoped to an exact (test_path, backend, suite)
    tuple, so any difference here isolates one test's history from
    another's. A test-file edit does not reset the series.
    """

    test_path: str
    backend: str
    suite: str


@dataclass(frozen=True)
class RunProvenance:
    """Where a run came from. Recorded for audit and for `mark_untrusted`
    targeting; never used to assemble a baseline."""

    commit_sha: str
    pr_number: int | None
    github_run_id: int | None
    github_run_attempt: int | None
    event_name: str | None
    ref: str | None


def validate_finite_values(values: list[MetricSample]) -> None:
    """Reject non-finite sample values before a backend persists anything."""
    for sample in values:
        if not math.isfinite(sample.value):
            raise ValueError(
                f"non-finite metric value {sample.value!r} for {sample.metric_key!r} "
                f"(step={sample.step}); non-finite values never enter the store"
            )


class MetricHistoryStore(abc.ABC):
    """Abstract metric-history store.

    Implementations persist runs and their metric values and answer the
    baseline query. The query and write surface are deliberately narrow: the
    gate logic depends only on this interface, so swapping the SQLite test
    backend for a hosted Postgres backend changes no caller.
    """

    @abc.abstractmethod
    def write_run(
        self,
        identity: RunIdentity,
        provenance: RunProvenance,
        created_at: str,
        trusted: bool,
        values: list[MetricSample],
    ) -> str:
        """Persist one run and its metric values; return the new `run_id`.

        `created_at` is an ISO-8601 timestamp string (timestamptz on the
        server). The store assigns and returns the `run_id`; callers do not
        supply it.

        Raises `ValueError` if any sample's value is non-finite (NaN/±Inf),
        before anything is persisted: the store is the write boundary where
        validity is enforced, so non-finite values never enter a baseline.
        Implementations enforce this via :func:`validate_finite_values`.
        """

    @abc.abstractmethod
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
        """Return up to `limit` baseline values, newest run first.

        Only trusted runs matching the exact identity tuple and the exact
        value coordinate `(metric_key, steps_key, constraint_key, step)`
        contribute. Every coordinate column is NOT NULL, so matching is plain
        equality -- no NULL-equality semantics anywhere in the query.
        """

    @abc.abstractmethod
    def mark_untrusted(
        self,
        *,
        run_id: str | None = None,
        github_run_id: int | None = None,
        commit_sha: str | None = None,
    ) -> int:
        """Revoke trust on the runs matching exactly one of the given keys.

        Returns the number of run rows whose `trusted` flag changed from true
        to false. Already-untrusted matches do not count. Revocation takes
        effect immediately for subsequent `recent_trusted_values` calls; no
        rebaseline step is required.

        Exactly one of `run_id` / `github_run_id` / `commit_sha` must be
        provided.
        """
