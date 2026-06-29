"""Offline unit tests for the metric-history store contract.

These run against :class:`SQLiteMetricHistoryStore` with an in-memory database:
no network, no driver install, no live Postgres. The SQLite backend mirrors the
authoritative baseline query, so passing here means the contract the gate
depends on holds.
"""

from __future__ import annotations

import re

import pytest
from tests.ci.metric_history import (
    MetricSample,
    NeonMetricHistoryStore,
    RunIdentity,
    RunProvenance,
    SQLiteMetricHistoryStore,
)

IDENTITY = RunIdentity(
    test_path="tests/e2e/test_grpo.py",
    backend="megatron",
    suite="stage-c-8-gpu-h100",
    test_file_hash="a" * 64,
)

PROVENANCE = RunProvenance(
    commit_sha="deadbeef",
    pr_number=42,
    github_run_id=1001,
    github_run_attempt=1,
    event_name="pull_request",
    ref="refs/pull/42/merge",
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
        values=[MetricSample("reward_mean", None, 0.81)],
    )
    _write(
        store,
        created_at="2026-06-02T00:00:00+00:00",
        values=[MetricSample("reward_mean", None, 0.83)],
    )

    got = store.recent_trusted_values(
        IDENTITY.test_path,
        IDENTITY.backend,
        IDENTITY.suite,
        "reward_mean",
        None,
        IDENTITY.test_file_hash,
        limit=10,
    )
    # Newest run first.
    assert got == [0.83, 0.81]


def test_limit_caps_and_orders_newest_first(store):
    for i, created in enumerate(
        ["2026-06-01T00:00:00+00:00", "2026-06-02T00:00:00+00:00", "2026-06-03T00:00:00+00:00"]
    ):
        _write(store, created_at=created, values=[MetricSample("reward_mean", None, float(i))])

    got = store.recent_trusted_values(
        IDENTITY.test_path,
        IDENTITY.backend,
        IDENTITY.suite,
        "reward_mean",
        None,
        IDENTITY.test_file_hash,
        limit=2,
    )
    assert got == [2.0, 1.0]


def test_identity_isolation(store):
    # Reference run under the canonical identity.
    _write(store, created_at="2026-06-01T00:00:00+00:00", values=[MetricSample("reward_mean", None, 0.5)])

    # Same metric, but each of these differs in exactly one identity field.
    other_backend = RunIdentity(IDENTITY.test_path, "fsdp", IDENTITY.suite, IDENTITY.test_file_hash)
    other_suite = RunIdentity(IDENTITY.test_path, IDENTITY.backend, "stage-b-2-gpu-h200", IDENTITY.test_file_hash)
    other_hash = RunIdentity(IDENTITY.test_path, IDENTITY.backend, IDENTITY.suite, "b" * 64)
    other_path = RunIdentity("tests/e2e/test_other.py", IDENTITY.backend, IDENTITY.suite, IDENTITY.test_file_hash)
    for ident in (other_backend, other_suite, other_hash, other_path):
        _write(
            store,
            identity=ident,
            created_at="2026-06-02T00:00:00+00:00",
            values=[MetricSample("reward_mean", None, 9.9)],
        )

    # Baseline for the canonical identity sees only its own single value.
    got = store.recent_trusted_values(
        IDENTITY.test_path,
        IDENTITY.backend,
        IDENTITY.suite,
        "reward_mean",
        None,
        IDENTITY.test_file_hash,
        limit=10,
    )
    assert got == [0.5]


def test_sub_label_null_is_distinct_from_labeled(store):
    # One unlabeled value and two labeled values under the same metric_key.
    _write(
        store,
        created_at="2026-06-01T00:00:00+00:00",
        values=[
            MetricSample("pass_rate", None, 0.70),
            MetricSample("pass_rate", "shard-0", 0.60),
            MetricSample("pass_rate", "shard-1", 0.80),
        ],
    )

    # NULL filter matches only the unlabeled measurement.
    unlabeled = store.recent_trusted_values(
        IDENTITY.test_path, IDENTITY.backend, IDENTITY.suite, "pass_rate", None, IDENTITY.test_file_hash, limit=10
    )
    assert unlabeled == [0.70]

    # A specific label matches only that label, never the unlabeled row.
    shard0 = store.recent_trusted_values(
        IDENTITY.test_path, IDENTITY.backend, IDENTITY.suite, "pass_rate", "shard-0", IDENTITY.test_file_hash, limit=10
    )
    assert shard0 == [0.60]


def test_mark_untrusted_by_run_id_excludes_immediately(store):
    keep = _write(store, created_at="2026-06-01T00:00:00+00:00", values=[MetricSample("reward_mean", None, 0.5)])
    drop = _write(store, created_at="2026-06-02T00:00:00+00:00", values=[MetricSample("reward_mean", None, 0.9)])

    affected = store.mark_untrusted(run_id=drop)
    assert affected == 1

    got = store.recent_trusted_values(
        IDENTITY.test_path, IDENTITY.backend, IDENTITY.suite, "reward_mean", None, IDENTITY.test_file_hash, limit=10
    )
    # No rebaseline step: the dropped run is gone from the baseline on the very
    # next query, the kept run remains.
    assert got == [0.5]
    assert keep != drop


def test_mark_untrusted_by_github_run_id(store):
    prov = RunProvenance("c1", None, 7777, 1, "push", "refs/heads/main")
    _write(store, provenance=prov, created_at="2026-06-01T00:00:00+00:00", values=[MetricSample("m", None, 1.0)])
    _write(store, provenance=prov, created_at="2026-06-02T00:00:00+00:00", values=[MetricSample("m", None, 2.0)])

    affected = store.mark_untrusted(github_run_id=7777)
    assert affected == 2
    assert (
        store.recent_trusted_values(
            IDENTITY.test_path, IDENTITY.backend, IDENTITY.suite, "m", None, IDENTITY.test_file_hash, limit=10
        )
        == []
    )


def test_mark_untrusted_by_commit_sha(store):
    prov = RunProvenance("badc0de", 5, 1, 1, "pull_request", "refs/pull/5/merge")
    _write(store, provenance=prov, created_at="2026-06-01T00:00:00+00:00", values=[MetricSample("m", None, 1.0)])
    affected = store.mark_untrusted(commit_sha="badc0de")
    assert affected == 1


def test_mark_untrusted_is_idempotent(store):
    rid = _write(store, created_at="2026-06-01T00:00:00+00:00", values=[MetricSample("m", None, 1.0)])
    assert store.mark_untrusted(run_id=rid) == 1
    # Already untrusted -> no rows change.
    assert store.mark_untrusted(run_id=rid) == 0


def test_mark_untrusted_requires_exactly_one_key(store):
    with pytest.raises(ValueError):
        store.mark_untrusted()
    with pytest.raises(ValueError):
        store.mark_untrusted(run_id="x", commit_sha="y")


def test_untrusted_run_never_in_baseline(store):
    _write(store, created_at="2026-06-01T00:00:00+00:00", trusted=False, values=[MetricSample("m", None, 5.0)])
    assert (
        store.recent_trusted_values(
            IDENTITY.test_path, IDENTITY.backend, IDENTITY.suite, "m", None, IDENTITY.test_file_hash, limit=10
        )
        == []
    )


def test_baseline_sql_matches_authoritative_shape():
    # Guard against drift from the authoritative query: identity predicates,
    # NULL-equality on sub_label, trusted filter, created_at DESC, LIMIT.
    from tests.ci.metric_history.sqlite_store import _BASELINE_SQL

    sql = re.sub(r"\s+", " ", _BASELINE_SQL).strip().lower()
    for fragment in (
        "from metric_values mv",
        "join runs r using (run_id)",
        "r.test_path = ?",
        "r.backend = ?",
        "r.suite = ?",
        "mv.metric_key = ?",
        "mv.sub_label is not distinct from ?",
        "r.test_file_hash = ?",
        "r.trusted = 1",
        "order by r.created_at desc",
        "limit ?",
    ):
        assert fragment in sql, f"baseline query missing: {fragment!r}"


def test_neon_store_is_deferred():
    with pytest.raises(NotImplementedError):
        NeonMetricHistoryStore()


def test_production_migration_sql_creates_tables_and_index():
    # The production DDL targets Postgres; here we only assert the migration
    # file declares the two tables and the composite baseline index, so the
    # shapes under test and in production stay aligned.
    from pathlib import Path

    sql = (Path(__file__).parent / "migrations" / "0001_create_metric_history.sql").read_text().lower()
    assert "create table if not exists runs" in sql
    assert "create table if not exists metric_values" in sql
    assert "trusted" in sql and "boolean not null" in sql
    assert "double precision not null" in sql
    assert "timestamptz not null" in sql
    assert "create index if not exists runs_baseline_idx" in sql
    assert "(test_path, backend, suite, test_file_hash, trusted, created_at desc)" in sql


def test_least_privilege_role_grants_no_ddl():
    from pathlib import Path

    sql = (Path(__file__).parent / "migrations" / "0002_least_privilege_role.sql").read_text().lower()
    assert "grant insert, select, update on runs" in sql
    assert "grant insert, select, update on metric_values" in sql
    # The app role must not be handed schema-mutating or delete verbs.
    assert "grant create" not in sql
    assert "grant delete" not in sql
    assert "grant all" not in sql
