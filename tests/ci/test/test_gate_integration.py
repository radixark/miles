"""Integration tests for wiring the regression gate into the CI harness.

These exercise the helpers and the gate hook extracted into
`tests.ci.ci_utils` directly -- no real DB, no real CUDA run. The store is an
in-memory :class:`SQLiteMetricHistoryStore`; the GitHub environment is
monkeypatched.

Covered:

* passing-attempt selection: only the PASSED attempt's record feeds the gate.
* baseline write: a policy-selected writer makes the hook persist a baseline row
  whose `trusted` flag comes from the verdict; each value records its
  declaration literals; a duplicate metric declaration or a no-spec file
  writes nothing.
* PR no-write: a `pull_request` event writes no row and emits a shadow
  verdict string.
* enforcement: coordinates block numerical regressions by default after ten
  trusted baselines; metric and infrastructure errors fail closed.
* harness: a gate failure reaches the final summary without rerunning the child.
"""

from __future__ import annotations

import io
import json
import shlex
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from tests.ci import ci_utils, run_suite
from tests.ci.ci_register import HWBackend, register_cpu_ci, ut_parse_one_file
from tests.ci.ci_utils import build_store_from_env, gate_provenance_from_env, run_gate_hook
from tests.ci.metric_history import MetricSample, RunIdentity, RunProvenance, SQLiteMetricHistoryStore

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

# Canonical declaration keys for the `last` + rel-0.20 fixtures below.
LAST_KEY = json.dumps("last", sort_keys=True, separators=(",", ":"))
REL20_KEY = json.dumps({"rel_up": 0.20, "rel_down": 0.20}, sort_keys=True, separators=(",", ":"))

_GITHUB_ENV_VARS = (
    "GITHUB_EVENT_NAME",
    "GITHUB_SHA",
    "GITHUB_COMMIT_NAME",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_REF",
    "GITHUB_STEP_SUMMARY",
    "NEON_DATABASE_URL",
    "GITHUB_ACTIONS",
    ci_utils.CI_GATE_RECORD_DIR_ENV,
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
        'register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100", labels=["megatron"], hardware=["hopper"])\n'
        + textwrap.dedent(gate_lines).strip()
        + "\n"
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
    def test_no_neon_url_with_unwritable_summary_still_skips(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path))
        assert build_store_from_env() is None

    def test_no_neon_url_returns_none(self, caplog):
        with caplog.at_level("INFO"):
            assert build_store_from_env() is None
        assert "skipped: NEON_DATABASE_URL is not set" in caplog.text

    def test_neon_url_routes_to_neon_store(self, monkeypatch):
        # With the env var set the factory must construct the Neon backend
        # rather than return None. Patch the symbol imported by ci_utils with a
        # fake that records construction without accessing the hosted backend.
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

    def test_store_construction_failure_propagates(self, monkeypatch):
        # A configured store must not silently disable enforcement on failure.
        class _ExplodingStore:
            def __init__(self, dsn=None):
                raise RuntimeError("connection refused")

        monkeypatch.setattr("tests.ci.ci_utils.NeonMetricHistoryStore", _ExplodingStore)
        monkeypatch.setenv("NEON_DATABASE_URL", "postgres://unreachable/db")

        with pytest.raises(RuntimeError, match="connection refused"):
            build_store_from_env()


# --- passing-attempt selection ----------------------------------------------


class TestPassingAttemptSelection:
    def test_only_passing_attempt_record_is_evaluated_and_written(self, tmp_path, store, monkeypatch):
        # Baseline writer so the hook writes; assert the persisted value comes from the
        # PASSING attempt's good record, never the failed attempt's bad one.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
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
            executing_suite="stage-c-8-gpu-h100",
            write_baseline=True,
            provenance=PROVENANCE,
        )

        rows = store._conn.execute("SELECT mv.value FROM metric_values mv JOIN runs r USING (run_id)").fetchall()
        assert rows == [(0.80,)], "only the passing attempt's value must be written"


# --- baseline write ---------------------------------------------------------


class TestBaselineWrite:
    def test_nightly_writes_trusted_row_with_provenance(self, tmp_path, store, monkeypatch):
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
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
            executing_suite="stage-c-8-gpu-h100",
            write_baseline=True,
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
            registry.filename, "cuda", "stage-c-8-gpu-h100", "rollout/raw_reward", -1, 20
        )
        assert vals == [0.81]

    def test_nightly_writes_untrusted_when_verdict_not_trusted(self, tmp_path, store):
        # A historical failure makes the verdict not-trusted; the row is still
        # written (baseline writers always write) but flagged untrusted, so it never
        # pollutes a future baseline.
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
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
            executing_suite="stage-c-8-gpu-h100",
            write_baseline=True,
            provenance=PROVENANCE,
            now_iso="2026-06-29T00:00:00+00:00",
        )
        trusted = store._conn.execute("SELECT trusted FROM runs ORDER BY created_at DESC").fetchone()[0]
        assert trusted == 0
        # Untrusted rows are invisible to the baseline query: only the seeded
        # trusted baseline comes back, never the hook's 0.90 row.
        vals = store.recent_trusted_values(
            registry.filename, "cuda", "stage-c-8-gpu-h100", "rollout/raw_reward", -1, 20
        )
        assert vals == [0.30]

    def test_nightly_one_liner_writes_every_step(self, tmp_path, store):
        test_file = _write_test_file(
            tmp_path,
            'register_ci_gate(metric_key="rollout/raw_reward")',
        )
        registry = _registry(test_file)
        record = _write_record(
            tmp_path,
            {"rollout/raw_reward": [[0, 0.20], [1, 0.40], [2, 0.60], [3, 0.80]]},
            name="m.jsonl",
        )

        run_gate_hook(
            test_file,
            record,
            store=store,
            registry=registry,
            executing_suite="stage-c-8-gpu-h100",
            write_baseline=True,
            provenance=PROVENANCE,
        )

        rows = store._conn.execute("SELECT steps_key, step, value FROM metric_values ORDER BY step").fetchall()
        assert rows == [
            ('"all"', 0, 0.20),
            ('"all"', 1, 0.40),
            ('"all"', 2, 0.60),
            ('"all"', 3, 0.80),
        ]

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
            executing_suite="stage-c-8-gpu-h100",
            write_baseline=True,
            provenance=PROVENANCE,
        )
        assert _count_runs(store) == 0

    def test_nightly_records_declaration_literals_beside_each_value(self, tmp_path, store, monkeypatch):
        # Readers that still key on the literal columns must keep finding new rows.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(
            tmp_path,
            'register_ci_gate(metric_key="rollout/raw_reward", steps="last", '
            'constraint={"rel_up": 0.20, "rel_down": 0.20})',
        )
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.81]]}, name="m.jsonl")
        assert _run_hook(test_file, record, store, write_baseline=True) is True
        rows = store._conn.execute(
            "SELECT metric_key, steps_key, constraint_key, step, value FROM metric_values"
        ).fetchall()
        assert rows == [("rollout/raw_reward", LAST_KEY, REL20_KEY, -1, 0.81)]

    def test_duplicate_metric_declaration_fails_without_writing(self, tmp_path, store):
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20},
                             enforce=False)
            """,
        )
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.81]]}, name="m.jsonl")
        assert _run_hook(test_file, record, store, write_baseline=True) is False
        assert _count_runs(store) == 0


# --- PR no-write + shadow verdict -------------------------------------------


class TestExecutingSuiteIdentity:
    """A run's numbers belong to the stage that produced them.

    Once a test can be dispatched to a second GPU generation, keying the
    baseline on `register_*_ci(suite=...)` would merge two hardware populations
    into one series and make the regression gate compare across generations.
    """

    def test_row_is_keyed_on_the_executing_stage_not_the_declared_one(self, tmp_path, store):
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
            """,
        )
        registry = _registry(test_file)
        assert registry.suite == "stage-c-8-gpu-h100"
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.81]]}, name="m.jsonl")

        run_gate_hook(
            test_file,
            record,
            store=store,
            registry=registry,
            executing_suite="stage-c-8-gpu-b200",
            write_baseline=True,
            provenance=PROVENANCE,
        )

        assert store._conn.execute("SELECT suite FROM runs").fetchone()[0] == "stage-c-8-gpu-b200"

    def test_the_two_stages_keep_independent_series(self, tmp_path, store):
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
            """,
        )
        registry = _registry(test_file)
        for suite, value in (("stage-c-8-gpu-h100", 0.81), ("stage-c-8-gpu-b200", 0.42)):
            run_gate_hook(
                test_file,
                _write_record(tmp_path, {"rollout/raw_reward": [[0, value]]}, name=f"{suite}.jsonl"),
                store=store,
                registry=registry,
                executing_suite=suite,
                write_baseline=True,
                provenance=PROVENANCE,
            )

        coordinate = ("rollout/raw_reward", -1)
        hopper = store.recent_trusted_values(registry.filename, "cuda", "stage-c-8-gpu-h100", *coordinate, 20)
        blackwell = store.recent_trusted_values(registry.filename, "cuda", "stage-c-8-gpu-b200", *coordinate, 20)
        # 0.42 against a 0.81 baseline would be a hard fail if the two shared a series.
        assert hopper == [0.81]
        assert blackwell == [0.42]


class TestPrShadow:
    def test_pr_writes_no_row_and_emits_shadow_verdict(self, tmp_path, store, monkeypatch, caplog):
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        summary = tmp_path / "step_summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20}, enforce=False)
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
                executing_suite="stage-c-8-gpu-h100",
                write_baseline=False,
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


# --- shadow declarations never block ---------------------------------------


class TestShadowNeverBlocks:
    def test_not_trusted_verdict_does_not_raise_in_pr(self, tmp_path, store, monkeypatch):
        # A clearly-not-trusted PR run still completes the hook normally (no
        # exception, no new row); the hook allows the child's success to stand.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20}, enforce=False)
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
            executing_suite="stage-c-8-gpu-h100",
            write_baseline=False,
            provenance=PROVENANCE,
        )
        assert result is True
        # Only the seeded baseline row: the PR run wrote nothing.
        assert _count_runs(store) == 1

    def test_gate_error_is_swallowed(self, tmp_path, store, monkeypatch, caplog):
        # A missing record path makes evaluate_gate raise; the hook must catch,
        # log, and return True rather than propagate for a shadow declaration.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        test_file = _write_test_file(
            tmp_path,
            """
            register_ci_gate(metric_key="rollout/raw_reward",
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20}, enforce=False)
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
                executing_suite="stage-c-8-gpu-h100",
                write_baseline=True,
                provenance=PROVENANCE,
            )
        assert result is True
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
                             steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20}, enforce=False)
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
                executing_suite="stage-c-8-gpu-h100",
                write_baseline=True,
                provenance=PROVENANCE,
            )
            assert result is True
        finally:
            store.close()


# --- backend gating: hook never fires off the CUDA path ----------------------


def test_hook_signature_matches_metric_sample_contract(tmp_path, store, monkeypatch):
    # Regression guard: baseline values must be built as
    # MetricSample(metric_key, steps_key, constraint_key, step, current) and only
    # for metrics whose current is not None. A spec on a missing metric
    # (current None) yields a written run with zero values.
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    test_file = _write_test_file(
        tmp_path,
        """
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last", constraint={"rel_up": 0.20, "rel_down": 0.20})
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
        executing_suite="stage-c-8-gpu-h100",
        write_baseline=True,
        provenance=PROVENANCE,
    )
    assert _count_runs(store) == 1
    n_values = store._conn.execute("SELECT COUNT(*) FROM metric_values").fetchone()[0]
    assert n_values == 0
    # The MetricSample type is the contract the hook builds from; assert it imports.
    assert MetricSample("k", LAST_KEY, REL20_KEY, -1, 1.0).value == 1.0
    assert HWBackend.CUDA is not None
    assert RunIdentity("p", "cuda", "s").backend == "cuda"


# --- default enforcement --------------------------------------------------


def _seed_history(store, registry, n, *, values=None):
    for index in range(n):
        store.write_run(
            RunIdentity(registry.filename, "cuda", registry.suite),
            PROVENANCE,
            created_at=f"2026-06-{index + 1:02d}T00:00:00+00:00",
            trusted=True,
            values=values or [MetricSample("rollout/raw_reward", LAST_KEY, REL20_KEY, -1, 0.30)],
        )


def _run_hook(test_file, record, store, *, write_baseline=False):
    registry = _registry(test_file)
    return run_gate_hook(
        test_file,
        record,
        store=store,
        registry=registry,
        executing_suite=registry.suite,
        write_baseline=write_baseline,
        provenance=PROVENANCE,
        now_iso="2026-06-29T00:00:00+00:00",
    )


class TestEnforcement:
    @pytest.mark.parametrize("enforce", [False, True])
    def test_summary_error_does_not_change_gate_verdict(self, tmp_path, store, monkeypatch, enforce):
        test_file = _write_test_file(
            tmp_path,
            'register_ci_gate(metric_key="rollout/raw_reward", steps="last", '
            'constraint={"rel_up": 0.20, "rel_down": 0.20}' + ("" if enforce else ", enforce=False") + ")",
        )
        _seed_history(store, _registry(test_file), 10)
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.90]]}, name="m.jsonl")
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path))
        assert _run_hook(test_file, record, store) is not enforce

    @pytest.mark.parametrize("enforce", [False, True])
    @pytest.mark.parametrize("write_baseline", [False, True], ids=["pr", "nightly"])
    @pytest.mark.parametrize("n,current", [(0, 0.90), (1, 0.90), (9, 0.90), (10, 0.90), (10, 0.30)])
    def test_history_threshold_and_baseline_trust(self, tmp_path, store, caplog, enforce, write_baseline, n, current):
        test_file = _write_test_file(
            tmp_path,
            'register_ci_gate(metric_key="rollout/raw_reward", steps="last", '
            'constraint={"rel_up": 0.20, "rel_down": 0.20}' + ("" if enforce else ", enforce=False") + ")",
        )
        registry = _registry(test_file)
        _seed_history(store, registry, n)
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, current]]}, name="m.jsonl")

        with caplog.at_level("INFO"):
            allowed = _run_hook(test_file, record, store, write_baseline=write_baseline)

        assert allowed is not (enforce and n >= 10 and current == 0.90)
        assert _count_runs(store) == n + int(write_baseline)
        if enforce and n < 10:
            assert f"NOT ENFORCED: insufficient history ({n}/10)" in caplog.text
        if write_baseline:
            trusted = store._conn.execute("SELECT trusted FROM runs ORDER BY created_at DESC").fetchone()[0]
            assert bool(trusted) is (n == 0 or current == 0.30)
            history = store.recent_trusted_values(
                registry.filename, "cuda", registry.suite, "rollout/raw_reward", -1, 20
            )
            assert len(history) == n + int(bool(trusted))
            if enforce and n == 9:
                # The newly persisted untrusted run must not satisfy the threshold.
                assert _run_hook(test_file, record, store) is True

    @pytest.mark.parametrize("enforce_failing", [False, True])
    def test_explicit_shadow_override_applies_only_to_its_metric(self, tmp_path, store, enforce_failing):
        declarations = [
            'register_ci_gate(metric_key="rollout/raw_reward", steps="last", '
            'constraint={"rel_up": 0.20, "rel_down": 0.20}' + (", enforce=False" if enforce_failing else "") + ")",
            'register_ci_gate(metric_key="train/grad_norm", steps="last", '
            'constraint={"rel_up": 0.20, "rel_down": 0.20}' + ("" if enforce_failing else ", enforce=False") + ")",
        ]
        test_file = _write_test_file(tmp_path, "\n".join(declarations))
        samples = [
            MetricSample(metric, LAST_KEY, REL20_KEY, -1, 0.30) for metric in ("rollout/raw_reward", "train/grad_norm")
        ]
        _seed_history(store, _registry(test_file), 10, values=samples)
        record = _write_record(
            tmp_path, {"rollout/raw_reward": [[0, 0.30]], "train/grad_norm": [[0, 0.90]]}, name="m.jsonl"
        )
        assert _run_hook(test_file, record, store) is not enforce_failing
        assert _count_runs(store) == 10

    @pytest.mark.parametrize(
        "old_steps, new_steps, step",
        [('"last"', '"last"', -1), ('"all"', "[0]", 0), ("[0]", '"all"', 0)],
        ids=["constraint", "all-to-list", "list-to-all"],
    )
    def test_declaration_literal_edit_keeps_enforced_history(self, tmp_path, store, old_steps, new_steps, step):
        # History written under other literals still counts toward the threshold and the mean.
        test_file = _write_test_file(
            tmp_path,
            f'register_ci_gate(metric_key="rollout/raw_reward", steps={new_steps}, '
            'constraint={"rel_up": 0.50, "rel_down": 0.50})',
        )
        samples = [MetricSample("rollout/raw_reward", old_steps, REL20_KEY, step, 0.30)]
        _seed_history(store, _registry(test_file), 10, values=samples)
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.90]]}, name="m.jsonl")
        # Only the ten inherited baselines can make this drift an enforced failure.
        assert _run_hook(test_file, record, store) is False

    def test_history_threshold_is_per_step(self, tmp_path, store, caplog):
        test_file = _write_test_file(
            tmp_path,
            'register_ci_gate(metric_key="rollout/raw_reward", steps="all", '
            'constraint={"rel_up": 0.20, "rel_down": 0.20})',
        )
        registry = _registry(test_file)
        samples = [MetricSample("rollout/raw_reward", '"all"', REL20_KEY, step, 0.30) for step in (0, 1)]
        _seed_history(store, registry, 9, values=samples)
        _seed_history(store, registry, 1, values=samples[:1])
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.30], [1, 0.90]]}, name="m.jsonl")
        with caplog.at_level("INFO"):
            assert _run_hook(test_file, record, store) is True
        assert "NOT ENFORCED: insufficient history (9/10)" in caplog.text
        _seed_history(store, registry, 1, values=samples[1:])
        assert _run_hook(test_file, record, store) is False

    @pytest.mark.parametrize("enforce", [False, True])
    @pytest.mark.parametrize("series", [None, [], [[0, "NaN"]], [[0, "Infinity"]], [[0, "-Infinity"]]])
    @pytest.mark.parametrize("write_baseline", [False, True], ids=["pr", "nightly"])
    def test_bad_selected_metrics_do_not_wait_for_history(self, tmp_path, store, enforce, series, write_baseline):
        test_file = _write_test_file(
            tmp_path, 'register_ci_gate(metric_key="rollout/raw_reward"' + ("" if enforce else ", enforce=False") + ")"
        )
        record = _write_record(tmp_path, {} if series is None else {"rollout/raw_reward": series}, name="m.jsonl")
        assert _run_hook(test_file, record, store, write_baseline=write_baseline) is not enforce
        assert _count_runs(store) == int(write_baseline)
        if write_baseline:
            assert store._conn.execute("SELECT trusted FROM runs").fetchone()[0] == 0

    @pytest.mark.parametrize("enforce", [False, True])
    @pytest.mark.parametrize(
        "failure", ["record_unavailable", "record_missing", "record_malformed", "read", "write", "evaluate"]
    )
    def test_gate_io_failures_obey_enforcement(self, tmp_path, store, monkeypatch, caplog, enforce, failure):
        test_file = _write_test_file(
            tmp_path, 'register_ci_gate(metric_key="rollout/raw_reward"' + ("" if enforce else ", enforce=False") + ")"
        )
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.30]]}, name="m.jsonl")
        if failure == "record_unavailable":
            record = None
        elif failure == "record_missing":
            Path(record).unlink()
        elif failure == "record_malformed":
            Path(record).write_text("invalid json\n")
        else:

            def explode(*args, **kwargs):
                raise RuntimeError("gate infrastructure unavailable")

            if failure == "evaluate":
                monkeypatch.setattr(ci_utils, "evaluate_gate", explode)
            else:
                monkeypatch.setattr(store, "write_run" if failure == "write" else "recent_trusted_values", explode)
        with caplog.at_level("WARNING"):
            assert _run_hook(test_file, record, store, write_baseline=True) is not enforce
        assert "[CI Gate] hook failed" in caplog.text
        assert _count_runs(store) == 0

    def test_invalid_declaration_fails_closed(self, tmp_path, store):
        test_file = _write_test_file(tmp_path, 'register_ci_gate(metric_key="rollout/raw_reward", enforce="yes")')
        record = _write_record(tmp_path, {"rollout/raw_reward": [[0, 0.30]]}, name="m.jsonl")
        assert _run_hook(test_file, record, store) is False


class TestHarnessEnforcement:
    @pytest.mark.parametrize("continue_on_error", [False, True])
    @pytest.mark.parametrize("failure", ["regression", "record_unavailable", "merge"])
    def test_gate_failure_reaches_suite_summary_without_child_retry(
        self, tmp_path, store, monkeypatch, caplog, capsys, continue_on_error, failure
    ):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        if failure != "record_unavailable":
            monkeypatch.setenv(ci_utils.CI_GATE_RECORD_DIR_ENV, str(tmp_path / "records"))
        files = [
            _registry(
                _write_test_file(
                    tmp_path,
                    'register_ci_gate(metric_key="rollout/raw_reward", steps="last", '
                    'constraint={"rel_up": 0.20, "rel_down": 0.20})',
                    name="test_regression.py",
                )
            ),
            _registry(_write_test_file(tmp_path, "", name="test_next.py")),
        ]
        _seed_history(store, files[0], 10)
        attempted = []

        def run_child(func, args, kwargs, timeout):
            attempted.append(args[0])
            record_dir = kwargs["record_dir"]
            if record_dir:
                Path(record_dir).mkdir(parents=True)
                _write_record(Path(record_dir), {"rollout/raw_reward": [[0, 0.90]]}, name="child.jsonl")
            return 0

        monkeypatch.setattr(ci_utils, "run_with_timeout", run_child)
        if failure == "merge":

            def broken_merge(*args):
                raise OSError("record merge unavailable")

            monkeypatch.setattr(ci_utils, "_merge_attempt_records", broken_merge)
        with caplog.at_level("INFO"):
            rc = ci_utils.run_unittest_files(
                files,
                timeout_per_file=10,
                continue_on_error=continue_on_error,
                enable_retry=True,
                max_attempts=3,
                retry_wait_seconds=0,
                gate_store=store,
                gate_executing_suite=files[0].suite,
            )

        assert rc != 0
        assert attempted == [f.filename for f in files[: 2 if continue_on_error else 1]]
        output = capsys.readouterr().out
        assert f"[1/2] {files[0].filename}  FAIL" in output
        assert f"[1/2] {files[0].filename}  PASS" not in output
        assert f"Test Summary: {int(continue_on_error)}/2 passed" in caplog.text
        summary = caplog.text.split("Test Summary:", 1)[1]
        assert f"{files[0].filename} (metric-history gate failed" in summary
        if continue_on_error:
            passed, failed = summary.split("FAILED:")
            assert files[1].filename in passed
            assert files[0].filename not in passed
            assert files[1].filename not in failed
        else:
            assert "PASSED:" not in summary
        assert _count_runs(store) == 10

    @pytest.mark.parametrize("current,allowed", [(0.30, True), (0.90, False)])
    def test_only_successful_child_attempt_is_gated_after_retry(
        self, tmp_path, store, monkeypatch, caplog, current, allowed
    ):
        monkeypatch.setenv(ci_utils.CI_GATE_RECORD_DIR_ENV, str(tmp_path / "records"))
        registry = _registry(
            _write_test_file(
                tmp_path,
                'register_ci_gate(metric_key="rollout/raw_reward", steps="last", '
                'constraint={"rel_up": 0.20, "rel_down": 0.20})',
            )
        )
        _seed_history(store, registry, 10)
        attempts = []

        def child(*args, env, **kwargs):
            record_dir = Path(env[ci_utils.CI_GATE_RECORD_DIR_ENV])
            attempts.append(record_dir)
            first = len(attempts) == 1
            _write_record(record_dir, {"rollout/raw_reward": [[0, 9.0 if first else current]]}, name="child.jsonl")
            return SimpleNamespace(
                stdout=io.StringIO("accuracy retry\n" if first else ""),
                returncode=1 if first else 0,
                wait=lambda: None,
            )

        monkeypatch.setattr(ci_utils.subprocess, "Popen", child)
        with caplog.at_level("INFO"):
            rc = ci_utils.run_unittest_files(
                [registry],
                timeout_per_file=10,
                enable_retry=True,
                max_attempts=3,
                retry_wait_seconds=0,
                gate_store=store,
                gate_executing_suite=registry.suite,
                gate_write_baseline=True,
            )
        assert (rc == 0) is allowed
        assert len(attempts) == 2
        assert not Path(f"{attempts[0]}.merged.jsonl").exists()
        assert Path(f"{attempts[1]}.merged.jsonl").exists()
        assert _count_runs(store) == 11
        row = store._conn.execute(
            "SELECT mv.value, r.trusted FROM metric_values mv JOIN runs r USING (run_id) "
            "ORDER BY r.created_at DESC LIMIT 1"
        ).fetchone()
        assert row == (current, int(allowed))
        assert f"(2 attempts, {'passed' if allowed else 'failed'})" in caplog.text
        assert f"Test Summary: {int(allowed)}/1 passed" in caplog.text


@pytest.mark.parametrize("nightly", [False, True])
@pytest.mark.parametrize(
    "baseline_n,current,allowed", [(9, 0.90, True), (10, 0.90, False), (10, 0.30, True), (10, None, False)]
)
def test_comment_file_command_runs_gate_without_writing_history(
    tmp_path, store, monkeypatch, caplog, nightly, baseline_n, current, allowed
):
    workflow = yaml.safe_load((Path(__file__).parents[3] / ".github/workflows/run-ci-file.yml").read_text())
    command = workflow["jobs"]["run-cuda-file"]["with"]["execute_command"]
    test_file = Path(
        _write_test_file(
            tmp_path,
            'register_ci_gate(metric_key="rollout/raw_reward", steps="last", '
            'constraint={"rel_up": 0.20, "rel_down": 0.20})',
        )
    )
    test_file.write_text(test_file.read_text().replace('labels=["megatron"]', f'labels=["long"], nightly={nightly}'))
    other = Path(_write_test_file(tmp_path, "", name="test_not_requested.py"))
    monkeypatch.chdir(tmp_path)
    registry = _registry(test_file.name)
    _seed_history(store, registry, baseline_n)
    command = command.replace("${{ inputs.test_file }}", test_file.name)
    command = command.replace("${{ needs.resolve-file-run.outputs.suite }}", registry.suite)
    command = command.replace("${{ needs.resolve-file-run.outputs.timeout_seconds }}", "2700")
    argv = shlex.split(command)
    monkeypatch.setattr(sys, "argv", argv[argv.index("python3") + 1 :])
    monkeypatch.setattr(run_suite, "discover_ci_files", lambda: [test_file.name, other.name])
    monkeypatch.setattr(run_suite, "build_store_from_env", lambda: store)
    monkeypatch.setattr(run_suite, "reaping_is_isolated", lambda: False)
    monkeypatch.setattr(run_suite.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path / "records"))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    attempted = []

    def run_child(func, args, kwargs, timeout):
        attempted.append(args[0])
        assert timeout == 2700
        record_dir = Path(kwargs["record_dir"])
        record_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"rollout/raw_reward": [[0, current]]} if current is not None else {}
        _write_record(record_dir, metrics, name="child.jsonl")
        return 0

    monkeypatch.setattr(ci_utils, "run_with_timeout", run_child)
    with caplog.at_level("INFO"), pytest.raises(SystemExit) as result:
        run_suite.main()
    assert (result.value.code == 0) is allowed
    assert attempted == [test_file.name]
    assert _count_runs(store) == baseline_n
    assert f"[CI Gate][enforce] {test_file.name}: {'PASS' if allowed else 'FAIL'}" in caplog.text
