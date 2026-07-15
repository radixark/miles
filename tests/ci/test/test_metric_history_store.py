"""Offline unit tests for the metric-history store contract.

These run against :class:`SQLiteMetricHistoryStore` with an in-memory database:
no network, no driver install, no live Postgres. The SQLite backend mirrors the
authoritative baseline query, so passing here means the contract the gate
depends on holds.
"""

from __future__ import annotations

import re

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.ci.metric_history import (
    MetricSample,
    NeonMetricHistoryStore,
    RunIdentity,
    RunProvenance,
    SQLiteMetricHistoryStore,
)

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

IDENTITY = RunIdentity(
    test_path="tests/e2e/test_grpo.py",
    backend="megatron",
    suite="stage-c-8-gpu-h100",
)

PROVENANCE = RunProvenance(
    commit_sha="deadbeef",
    pr_number=42,
    github_run_id=1001,
    github_run_attempt=1,
    event_name="pull_request",
    ref="refs/pull/42/merge",
)

# Canonical-JSON declaration keys, as the parser would derive them from a
# test file's literal steps / constraint values.
LAST_STEPS_KEY = '"last"'
STEP_LIST_KEY = "[0,1]"
REL_CONSTRAINT_KEY = '{"rel":0.2}'
ABS_CONSTRAINT_KEY = '{"abs_floor":0.02}'


def _sample(metric_key, value, *, steps_key=LAST_STEPS_KEY, constraint_key=REL_CONSTRAINT_KEY, step=-1):
    return MetricSample(metric_key, steps_key, constraint_key, step, value)


def _recent(
    store,
    metric_key,
    *,
    steps_key=LAST_STEPS_KEY,
    constraint_key=REL_CONSTRAINT_KEY,
    step=-1,
    identity=IDENTITY,
    limit=10,
):
    return store.recent_trusted_values(
        identity.test_path,
        identity.backend,
        identity.suite,
        metric_key,
        steps_key,
        constraint_key,
        step,
        limit,
    )


@pytest.fixture
def store():
    s = SQLiteMetricHistoryStore(":memory:")
    yield s
    s.close()


def _write(
    store,
    *,
    identity=IDENTITY,
    provenance=PROVENANCE,
    created_at,
    trusted=True,
    values,
):
    return store.write_run(identity, provenance, created_at, trusted, values)


def test_write_then_recent_returns_written_values(store):
    _write(
        store,
        created_at="2026-06-01T00:00:00+00:00",
        values=[_sample("reward_mean", 0.81)],
    )
    _write(
        store,
        created_at="2026-06-02T00:00:00+00:00",
        values=[_sample("reward_mean", 0.83)],
    )

    # Newest run first.
    assert _recent(store, "reward_mean") == [0.83, 0.81]


def test_write_run_rejects_non_finite_before_persisting(store):
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite"):
            _write(
                store,
                created_at="2026-06-01T00:00:00+00:00",
                values=[_sample("reward_mean", 0.5), _sample("reward_mean", bad, step=0)],
            )

    # Validation runs before any insert: the finite sibling sample must not
    # have leaked in as a trusted row.
    assert _recent(store, "reward_mean") == []


def test_limit_caps_and_orders_newest_first(store):
    for i, created in enumerate(
        ["2026-06-01T00:00:00+00:00", "2026-06-02T00:00:00+00:00", "2026-06-03T00:00:00+00:00"]
    ):
        _write(store, created_at=created, values=[_sample("reward_mean", float(i))])

    assert _recent(store, "reward_mean", limit=2) == [2.0, 1.0]


def test_identity_isolation(store):
    # Reference run under the canonical identity.
    _write(store, created_at="2026-06-01T00:00:00+00:00", values=[_sample("reward_mean", 0.5)])

    # Same metric, but each of these differs in exactly one identity field.
    other_backend = RunIdentity(IDENTITY.test_path, "fsdp", IDENTITY.suite)
    other_suite = RunIdentity(IDENTITY.test_path, IDENTITY.backend, "stage-b-2-gpu-h200")
    other_path = RunIdentity("tests/e2e/test_other.py", IDENTITY.backend, IDENTITY.suite)
    for ident in (other_backend, other_suite, other_path):
        _write(
            store,
            identity=ident,
            created_at="2026-06-02T00:00:00+00:00",
            values=[_sample("reward_mean", 9.9)],
        )

    # Baseline for the canonical identity sees only its own single value.
    assert _recent(store, "reward_mean") == [0.5]


def test_coordinate_isolation(store):
    # Four values under one metric_key; each non-base row differs from the
    # base coordinate in exactly one coordinate column.
    _write(
        store,
        created_at="2026-06-01T00:00:00+00:00",
        values=[
            _sample("pass_rate", 0.70),
            _sample("pass_rate", 0.60, steps_key=STEP_LIST_KEY, step=0),
            _sample("pass_rate", 0.80, constraint_key=ABS_CONSTRAINT_KEY),
            _sample("pass_rate", 0.90, step=0),
        ],
    )

    # Each exact coordinate matches only its own row: plain equality on every
    # column, no cross-matching.
    assert _recent(store, "pass_rate") == [0.70]
    assert _recent(store, "pass_rate", steps_key=STEP_LIST_KEY, step=0) == [0.60]
    assert _recent(store, "pass_rate", constraint_key=ABS_CONSTRAINT_KEY) == [0.80]
    assert _recent(store, "pass_rate", step=0) == [0.90]


def test_mark_untrusted_by_run_id_excludes_immediately(store):
    keep = _write(store, created_at="2026-06-01T00:00:00+00:00", values=[_sample("reward_mean", 0.5)])
    drop = _write(store, created_at="2026-06-02T00:00:00+00:00", values=[_sample("reward_mean", 0.9)])

    affected = store.mark_untrusted(run_id=drop)
    assert affected == 1

    # No rebaseline step: the dropped run is gone from the baseline on the very
    # next query, the kept run remains.
    assert _recent(store, "reward_mean") == [0.5]
    assert keep != drop


def test_mark_untrusted_by_github_run_id(store):
    prov = RunProvenance("c1", None, 7777, 1, "push", "refs/heads/main")
    _write(store, provenance=prov, created_at="2026-06-01T00:00:00+00:00", values=[_sample("m", 1.0)])
    _write(store, provenance=prov, created_at="2026-06-02T00:00:00+00:00", values=[_sample("m", 2.0)])

    affected = store.mark_untrusted(github_run_id=7777)
    assert affected == 2
    assert _recent(store, "m") == []


def test_mark_untrusted_by_commit_sha(store):
    prov = RunProvenance("badc0de", 5, 1, 1, "pull_request", "refs/pull/5/merge")
    _write(store, provenance=prov, created_at="2026-06-01T00:00:00+00:00", values=[_sample("m", 1.0)])
    affected = store.mark_untrusted(commit_sha="badc0de")
    assert affected == 1


def test_mark_untrusted_is_idempotent(store):
    rid = _write(store, created_at="2026-06-01T00:00:00+00:00", values=[_sample("m", 1.0)])
    assert store.mark_untrusted(run_id=rid) == 1
    # Already untrusted -> no rows change.
    assert store.mark_untrusted(run_id=rid) == 0


def test_mark_untrusted_requires_exactly_one_key(store):
    with pytest.raises(ValueError):
        store.mark_untrusted()
    with pytest.raises(ValueError):
        store.mark_untrusted(run_id="x", commit_sha="y")


def test_untrusted_run_never_in_baseline(store):
    _write(store, created_at="2026-06-01T00:00:00+00:00", trusted=False, values=[_sample("m", 5.0)])
    assert _recent(store, "m") == []


def test_baseline_sql_matches_authoritative_shape():
    # Guard against drift from the authoritative query: identity predicates,
    # plain equality on every coordinate column, trusted filter,
    # created_at DESC, LIMIT.
    from tests.ci.metric_history.storage.sqlite_store import _BASELINE_SQL

    sql = re.sub(r"\s+", " ", _BASELINE_SQL).strip().lower()
    for fragment in (
        "from metric_values mv",
        "join runs r using (run_id)",
        "r.test_path = ?",
        "r.backend = ?",
        "r.suite = ?",
        "mv.metric_key = ?",
        "mv.steps_key = ?",
        "mv.constraint_key = ?",
        "mv.step = ?",
        "r.trusted = 1",
        "order by r.created_at desc",
        "limit ?",
    ):
        assert fragment in sql, f"baseline query missing: {fragment!r}"


def test_neon_store_is_deferred():
    with pytest.raises(NotImplementedError):
        NeonMetricHistoryStore()
