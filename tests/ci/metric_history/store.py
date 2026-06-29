"""Storage contract for the CI metric-history regression gate.

The gate compares a candidate run's metrics against a baseline assembled from
the most recent *trusted* runs that share the same test identity. This module
defines that storage contract and nothing else: it does not decide what counts
as a regression, does not read the candidate run, and does not talk to CI.

Two normalized tables back the contract:

* ``runs`` -- one row per CI execution of a test, holding identity
  (test_path, backend, suite, test_file_hash), provenance (commit_sha,
  pr_number, github_run_id, github_run_attempt, event_name, ref), the
  ``created_at`` timestamp, and the run-level ``trusted`` flag.
* ``metric_values`` -- the (metric_key, sub_label, value) measurements
  produced by a run, keyed back to ``runs`` by ``run_id``.

``trusted`` lives on the run, not on the metric: a run is trusted as a whole
or not at all, so revoking trust drops every metric the run contributed in one
operation.

``test_file_hash`` is the sha256 of the test file's contents. It is computed by
the caller; the store only stores and matches on it. Two runs of the same
``test_path`` with different file contents have different hashes and therefore
never share a baseline -- a test edit starts a fresh history rather than
silently comparing against measurements of older code.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSample:
    """One measurement a run contributed.

    ``sub_label`` distinguishes measurements that share a ``metric_key`` (e.g.
    per-prompt or per-shard values). ``None`` denotes the single unlabeled
    measurement for that key and is matched with NULL-equality semantics
    (``IS NOT DISTINCT FROM``) rather than ``=``.
    """

    metric_key: str
    sub_label: str | None
    value: float


@dataclass(frozen=True)
class RunIdentity:
    """The fields that decide whether two runs share a baseline.

    A baseline query is scoped to an exact (test_path, backend, suite,
    test_file_hash) tuple, so any difference here isolates one test's history
    from another's.
    """

    test_path: str
    backend: str
    suite: str
    test_file_hash: str


@dataclass(frozen=True)
class RunProvenance:
    """Where a run came from. Recorded for audit and for ``mark_untrusted``
    targeting; never used to assemble a baseline."""

    commit_sha: str
    pr_number: int | None
    github_run_id: int | None
    github_run_attempt: int | None
    event_name: str | None
    ref: str | None


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
        """Persist one run and its metric values; return the new ``run_id``.

        ``created_at`` is an ISO-8601 timestamp string (timestamptz on the
        server). The store assigns and returns the ``run_id``; callers do not
        supply it.
        """

    @abc.abstractmethod
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
        """Return up to ``limit`` baseline values, newest run first.

        Only trusted runs matching the exact identity tuple and metric
        coordinates contribute. ``sub_label`` is matched with NULL-equality
        semantics, so ``None`` matches the unlabeled measurement and never a
        labeled one.
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

        Returns the number of run rows whose ``trusted`` flag changed from true
        to false. Already-untrusted matches do not count. Revocation takes
        effect immediately for subsequent ``recent_trusted_values`` calls; no
        rebaseline step is required.

        Exactly one of ``run_id`` / ``github_run_id`` / ``commit_sha`` must be
        provided.
        """
