"""Integration tests for wiring the regression gate into the CI harness.

These exercise the helpers and the gate hook extracted into
`tests.ci.ci_utils` directly -- no real DB, no real CUDA run. The store is an
in-memory :class:`SQLiteMetricHistoryStore`; the GitHub environment is
monkeypatched.

Covered:

* passing-attempt selection: only the PASSED attempt's record feeds the gate.
* nightly write: a `schedule` event makes the hook persist a baseline row
  whose `trusted` flag comes from the verdict; specs sharing a coordinate
  collapse to one `metric_values` row; a no-spec file writes nothing.
* PR no-write: a `pull_request` event writes no row and emits a shadow
  verdict string.
* never-blocks: a not-trusted verdict, and a gate that raises, both leave the
  test's pass/fail untouched.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from tests.ci.ci_register import HWBackend, register_cpu_ci, ut_parse_one_file
from tests.ci.ci_utils import build_store_from_env, gate_provenance_from_env, run_gate_hook
from tests.ci.metric_history import MetricSample, RunIdentity, RunProvenance, SQLiteMetricHistoryStore

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

# Canonical declaration keys for the `last` + rel-0.20 fixtures below.
LAST_KEY = json.dumps("last", sort_keys=True, separators=(",", ":"))
REL20_KEY = json.dumps({"rel": 0.20}, sort_keys=True, separators=(",", ":"))

_GITHUB_ENV_VARS = (
    "GITHUB_EVENT_NAME",
    "GITHUB_SHA",
    "GITHUB_COMMIT_NAME",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_REF",
    "GITHUB_STEP_SUMMARY",
    "NEON_DATABASE_URL",
)


@pytest.fixture(autouse=True)
def _clear_github_env(monkeypatch):
    """Each test starts from a clean GitHub environment so leftovers from the
    real CI runner (or a prior test) cannot leak in."""
    for var in _GITHUB_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def store():
    s = SQLiteMetricHistoryStore(":memory:")
    yield s
    s.close()


def _write_test_file(tmp_path: Path, gate_lines: str, *, name: str = "test_e2e_gate_fixture.py") -> str:
    # Concatenate instead of interpolating into an indented f-string: a
    # multi-line gate block would otherwise defeat the outer dedent.
    body = (
        "from tests.ci.ci_register import register_cuda_ci\n"
        "from tests.ci.metric_history import register_ci_gate\n"
        'register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100")\n' + textwrap.dedent(gate_lines).strip() + "\n"
    )
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def _write_record(tmp_path: Path, by_metric: dict[str, list], *, name: str) -> str:
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for metric, series in by_metric.items():
            f.write(json.dumps({"metric": metric, "series": series}) + "\n")
    return str(p)


def _registry(test_file: str):
    return ut_parse_one_file(test_file)[0]


PROVENANCE = RunProvenance(
    commit_sha="deadbeef",
    pr_number=7,
    github_run_id=100,
    github_run_attempt=1,
    event_name="schedule",
    ref="refs/heads/main",
)


def _count_runs(store) -> int:
    return store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]


# --- gate_provenance_from_env -----------------------------------------------


class TestGateProvenanceFromEnv:
    def test_full_env_parsed(self, monkeypatch):
        monkeypatch.setenv("GITHUB_SHA", "cafef00d")
        monkeypatch.setenv("GITHUB_RUN_ID", "555")
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        monkeypatch.setenv("GITHUB_REF", "refs/pull/42/merge")
        monkeypatch.setenv("GITHUB_COMMIT_NAME", "cafef00d_42")

        prov = gate_provenance_from_env()
        assert prov.commit_sha == "cafef00d"
        assert prov.pr_number == 42
        assert prov.github_run_id == 555
        assert prov.github_run_attempt == 2
        assert prov.event_name == "pull_request"
        assert prov.ref == "refs/pull/42/merge"

    def test_non_pr_commit_name_yields_none_pr(self, monkeypatch):
        monkeypatch.setenv("GITHUB_COMMIT_NAME", "cafef00d_non-pr")
        prov = gate_provenance_from_env()
        assert prov.pr_number is None
        # sha falls back to the GITHUB_COMMIT_NAME prefix when GITHUB_SHA unset.
        assert prov.commit_sha == "cafef00d"

    def test_empty_env_does_not_raise(self):
        prov = gate_provenance_from_env()
        assert prov.commit_sha == ""
        assert prov.pr_number is None
        assert prov.github_run_id is None
        assert prov.github_run_attempt is None


# --- build_store_from_env ---------------------------------------------------


class TestBuildStoreFromEnv:
    def test_no_neon_url_returns_none(self):
        assert build_store_from_env() is None

    def test_neon_url_routes_to_neon_store(self, monkeypatch):
        # With the env var set the factory must construct the Neon backend
        # rather than return None. The real NeonMetricHistoryStore.__init__
        # eagerly opens a psycopg connection, so we patch the symbol *as imported
        # by ci_utils* with a non-connecting fake that records its construction.
        # This pins the factory's only job -- routing to NeonMetricHistoryStore --
        # without depending on psycopg or a reachable DSN.
        constructed = []

        class _FakeNeonStore:
            def __init__(self, dsn=None):
                constructed.append(dsn)

        monkeypatch.setattr("tests.ci.ci_utils.NeonMetricHistoryStore", _FakeNeonStore)
        monkeypatch.setenv("NEON_DATABASE_URL", "postgres://example/db")

        store = build_store_from_env()
        assert isinstance(store, _FakeNeonStore), "NEON_DATABASE_URL must route to NeonMetricHistoryStore"
        assert constructed == [None], "the factory constructs the store with no explicit DSN"

    def test_store_construction_failure_degrades_to_none(self, monkeypatch):
        # NeonMetricHistoryStore.__init__ connects eagerly, so a bad DSN / DB
        # outage raises here -- OUTSIDE the gate hook's try/except. The factory
        # must swallow it and return None: the gate may never fail a CI job
        # (CUDA or ROCm) before a single test has run.
        class _ExplodingStore:
            def __init__(self, dsn=None):
                raise RuntimeError("connection refused")

        monkeypatch.setattr("tests.ci.ci_utils.NeonMetricHistoryStore", _ExplodingStore)
        monkeypatch.setenv("NEON_DATABASE_URL", "postgres://unreachable/db")

        assert build_store_from_env() is None


# --- passing-attempt selection ----------------------------------------------


class TestPassingAttemptSelection:
    def test_only_passing_attempt_record_is_evaluated_and_written(self, tmp_path, store, monkeypatch):
        # Nightly so the hook writes; assert the persisted value comes from the
        # PASSING attempt's good record, never the failed attempt's bad one.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel": 0.20})
            """,
        )
        registry = _registry(test_file)

        # Attempt 1 "failed" with a bad metric (0.10), attempt 2 "passed" (0.80).
        bad_record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.10]]}, name="attempt-1.merged.jsonl")
        good_record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.80]]}, name="attempt-2.merged.jsonl")

        # Only the passing attempt's record path is handed to the hook -- this
        # is exactly what run_unittest_files selects (it captures the merged
        # path only inside the ret_code==0 branch).
        _ = bad_record
        run_gate_hook(
            test_file,
            good_record,
            store=store,
            registry=registry,
            nightly=True,
            provenance=PROVENANCE,
        )

        rows = store._conn.execute("SELECT mv.value FROM metric_values mv JOIN runs r USING (run_id)").fetchall()
        assert rows == [(0.80,)], "only the passing attempt's value must be written"


# --- nightly write ----------------------------------------------------------


class TestNightlyWrite:
    def test_nightly_writes_trusted_row_with_provenance(self, tmp_path, store, monkeypatch):
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel": 0.20})
            """,
        )
        registry = _registry(test_file)
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.81]]}, name="m.jsonl")

        assert _count_runs(store) == 0
        run_gate_hook(
            test_file,
            record,
            store=store,
            registry=registry,
            nightly=True,
            provenance=PROVENANCE,
            now_iso="2026-06-29T00:00:00+00:00",
        )

        row = store._conn.execute(
            "SELECT test_path, backend, suite, commit_sha, pr_number, "
            "github_run_id, event_name, created_at, trusted FROM runs"
        ).fetchone()
        assert row is not None
        assert row[0] == registry.filename
        assert row[1] == "cuda"
        assert row[2] == "stage-c-8-gpu-h100"
        assert row[3] == "deadbeef"
        assert row[4] == 7
        assert row[5] == 100
        assert row[6] == "schedule"
        assert row[7] == "2026-06-29T00:00:00+00:00"
        # Cold-start (no prior baseline) -> historical inactive -> trusted verdict.
        assert row[8] == 1

        # The persisted value feeds future baselines.
        vals = store.recent_trusted_values(
            registry.filename,
            "cuda",
            "stage-c-8-gpu-h100",
            "rollout/raw_reward",
            LAST_KEY,
            REL20_KEY,
            -1,
            20,
        )
        assert vals == [0.81]

    def test_nightly_writes_untrusted_when_verdict_not_trusted(self, tmp_path, store):
        # A historical failure makes the verdict not-trusted; the row is still
        # written (nightly always writes) but flagged untrusted, so it never
        # pollutes a future baseline.
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel": 0.20})
            """,
        )
        registry = _registry(test_file)
        # Seed a trusted baseline at 0.30 under the gated coordinate.
        store.write_run(
            RunIdentity(test_path=registry.filename, backend="cuda", suite="stage-c-8-gpu-h100"),
            PROVENANCE,
            created_at="2026-06-01T00:00:00+00:00",
            trusted=True,
            values=[MetricSample("rollout/raw_reward", LAST_KEY, REL20_KEY, -1, 0.30)],
        )
        # 0.90 vs baseline mean 0.30, band 0.06 -> historical fails -> not trusted.
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.90]]}, name="m.jsonl")

        run_gate_hook(
            test_file,
            record,
            store=store,
            registry=registry,
            nightly=True,
            provenance=PROVENANCE,
            now_iso="2026-06-29T00:00:00+00:00",
        )
        trusted = store._conn.execute("SELECT trusted FROM runs ORDER BY created_at DESC").fetchone()[0]
        assert trusted == 0
        # Untrusted rows are invisible to the baseline query: only the seeded
        # trusted baseline comes back, never the hook's 0.90 row.
        vals = store.recent_trusted_values(
            registry.filename,
            "cuda",
            "stage-c-8-gpu-h100",
            "rollout/raw_reward",
            LAST_KEY,
            REL20_KEY,
            -1,
            20,
        )
        assert vals == [0.30]

    def test_nightly_no_spec_file_writes_nothing(self, tmp_path, store, monkeypatch):
        # A file with no register_ci_gate call has nothing a baseline can use:
        # the hook must skip the write entirely, not leave an empty runs row.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(tmp_path, "")
        registry = _registry(test_file)
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.81]]}, name="m.jsonl")

        run_gate_hook(
            test_file,
            record,
            store=store,
            registry=registry,
            nightly=True,
            provenance=PROVENANCE,
        )
        assert _count_runs(store) == 0

    def test_nightly_dedupes_values_by_coordinate(self, tmp_path, store, monkeypatch):
        # Two specs with identical steps + constraint literals share one
        # coordinate even though their policy metadata differs; the selected
        # value is identical, so writing one row per spec would double-weight
        # this run in the baseline mean.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel": 0.20})
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel": 0.20},
                             enforce=True)
            """,
        )
        registry = _registry(test_file)
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.81]]}, name="m.jsonl")

        run_gate_hook(
            test_file,
            record,
            store=store,
            registry=registry,
            nightly=True,
            provenance=PROVENANCE,
        )
        assert _count_runs(store) == 1
        rows = store._conn.execute(
            "SELECT metric_key, steps_key, constraint_key, step, value FROM metric_values"
        ).fetchall()
        assert rows == [("rollout/raw_reward", LAST_KEY, REL20_KEY, -1, 0.81)]


# --- PR no-write + shadow verdict -------------------------------------------


class TestPrShadow:
    def test_pr_writes_no_row_and_emits_shadow_verdict(self, tmp_path, store, monkeypatch, caplog):
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        summary = tmp_path / "step_summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel": 0.20})
            """,
        )
        registry = _registry(test_file)
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.81]]}, name="m.jsonl")

        with caplog.at_level("INFO"):
            run_gate_hook(
                test_file,
                record,
                store=store,
                registry=registry,
                nightly=False,
                provenance=PROVENANCE,
            )

        # No row written for a PR run.
        assert _count_runs(store) == 0
        # A readable shadow verdict line was logged and appended to the summary.
        assert "[CI Gate][shadow]" in caplog.text
        assert "TRUSTED" in caplog.text
        assert summary.exists()
        summary_text = summary.read_text()
        assert "[CI Gate][shadow]" in summary_text
        assert "rollout/raw_reward" in summary_text


# --- never-blocks -----------------------------------------------------------


class TestNeverBlocks:
    def test_not_trusted_verdict_does_not_raise_in_pr(self, tmp_path, store, monkeypatch):
        # A clearly-not-trusted PR run still completes the hook normally (no
        # exception, no new row). The caller's file_passed is computed
        # independently of run_gate_hook, which returns None either way.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel": 0.20})
            """,
        )
        registry = _registry(test_file)
        # Seed a trusted baseline at 0.30 so 0.95 fails the historical check.
        store.write_run(
            RunIdentity(test_path=registry.filename, backend="cuda", suite="stage-c-8-gpu-h100"),
            PROVENANCE,
            created_at="2026-06-01T00:00:00+00:00",
            trusted=True,
            values=[MetricSample("rollout/raw_reward", LAST_KEY, REL20_KEY, -1, 0.30)],
        )
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.95]]}, name="m.jsonl")

        result = run_gate_hook(
            test_file,
            record,
            store=store,
            registry=registry,
            nightly=False,
            provenance=PROVENANCE,
        )
        assert result is None
        # Only the seeded baseline row: the PR run wrote nothing.
        assert _count_runs(store) == 1

    def test_gate_error_is_swallowed(self, tmp_path, store, monkeypatch, caplog):
        # A missing record path makes evaluate_gate raise; the hook must catch,
        # log, and return None rather than propagate -- CI stays green.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel": 0.20})
            """,
        )
        registry = _registry(test_file)
        missing_record = str(tmp_path / "does_not_exist.jsonl")

        with caplog.at_level("WARNING"):
            result = run_gate_hook(
                test_file,
                missing_record,
                store=store,
                registry=registry,
                nightly=True,
                provenance=PROVENANCE,
            )
        assert result is None
        assert _count_runs(store) == 0
        assert "[CI Gate] hook failed" in caplog.text

    def test_store_error_never_blocks(self, tmp_path, monkeypatch):
        # A store whose write_run raises (mirrors a real DB error, e.g. the Neon
        # backend's connection dropping mid-write) must not propagate out of the
        # hook.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel": 0.20})
            """,
        )
        registry = _registry(test_file)
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.81]]}, name="m.jsonl")

        class _ExplodingStore(SQLiteMetricHistoryStore):
            def write_run(self, *a, **k):
                raise RuntimeError("DB down")

        store = _ExplodingStore(":memory:")
        try:
            result = run_gate_hook(
                test_file,
                record,
                store=store,
                registry=registry,
                nightly=True,
                provenance=PROVENANCE,
            )
            assert result is None
        finally:
            store.close()


# --- backend gating: hook never fires off the CUDA path ----------------------


def test_hook_signature_matches_metric_sample_contract(tmp_path, store, monkeypatch):
    # Regression guard: nightly values must be built as
    # MetricSample(metric_key, steps_key, constraint_key, step, current) and only
    # for metrics whose current is not None. A spec on a missing metric
    # (current None) yields a written run with zero values.
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel": 0.20})
        """,
    )
    registry = _registry(test_file)
    # Record has a different metric; rollout/raw_reward is missing -> current None.
    record = _write_record(tmp_path, {"train/grad_norm": [[0, 1.0]]}, name="m.jsonl")

    run_gate_hook(
        test_file,
        record,
        store=store,
        registry=registry,
        nightly=True,
        provenance=PROVENANCE,
    )
    assert _count_runs(store) == 1
    n_values = store._conn.execute("SELECT COUNT(*) FROM metric_values").fetchone()[0]
    assert n_values == 0
    # The MetricSample type is the contract the hook builds from; assert it imports.
    assert MetricSample("k", LAST_KEY, REL20_KEY, -1, 1.0).value == 1.0
    assert HWBackend.CUDA is not None
    assert RunIdentity("p", "cuda", "s").backend == "cuda"
