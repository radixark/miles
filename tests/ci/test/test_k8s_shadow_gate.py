from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path

from tests.ci import ci_utils, run_suite
from tests.ci.ci_register import CIRegistry, HWBackend, register_cpu_ci
from tests.ci.metric_history.gate import MIN_ENFORCED_BASELINES
from tests.ci.metric_history.storage.sqlite_store import SQLiteMetricHistoryStore
from tests.ci.metric_history.storage.store import MetricSample, RunIdentity, RunProvenance

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

_TEST_SUITE = "stage-c-4-gpu-h200"


def _write_enforced_test_file(path: Path) -> None:
    path.write_text(
        "from tests.ci.ci_register import register_cuda_ci\n"
        "from tests.ci.metric_history.register import register_ci_gate\n"
        "\n"
        'register_cuda_ci(est_time=60, suite="stage-c-4-gpu-h200", labels=["short"], '
        'hardware=["hopper"])\n'
        'register_ci_gate(metric_key="train/ppo_kl", steps=[0], enforce=True)\n'
    )


def _seed_ray_baselines(store: SQLiteMetricHistoryStore, *, test_path: str, count: int) -> None:
    identity = RunIdentity(test_path=test_path, backend="cuda", suite=_TEST_SUITE)
    provenance = RunProvenance(
        commit_sha="0" * 40,
        pr_number=None,
        github_run_id=1,
        github_run_attempt=1,
        event_name="schedule",
        ref="refs/heads/main",
    )
    for idx in range(count):
        store.write_run(
            identity,
            provenance,
            f"2026-11-{idx + 1:02d}T15:00:00+00:00",
            trusted=True,
            values=[
                MetricSample(
                    metric_key="train/ppo_kl",
                    steps_key="[0]",
                    constraint_key='{"abs_floor_down":0.02,"abs_floor_up":0.02,"rel_down":0.8,"rel_up":0.5}',
                    step=0,
                    value=0.01,
                )
            ],
        )


def _count_runs_rows(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])


class TestKubernetesShadowMetricGate:
    def test_k8s_shadow_passes_logs_deviation_and_writes_zero_rows_against_ten_ray_baselines(
        self, tmp_path: Path, caplog
    ):
        """With >=10 Ray baselines and an enforced metric out of bounds, K8s shadow logs [CI Gate][shadow][k8s], returns PASS, and writes 0 rows."""
        db_path = tmp_path / "history.db"
        store = SQLiteMetricHistoryStore(str(db_path), repository="radixark/miles")
        try:
            test_file = tmp_path / "test_enforced_dummy.py"
            _write_enforced_test_file(test_file)
            _seed_ray_baselines(store, test_path=str(test_file), count=MIN_ENFORCED_BASELINES)
            assert _count_runs_rows(db_path) == MIN_ENFORCED_BASELINES

            record_path = tmp_path / "passing.merged.jsonl"
            record_path.write_text(json.dumps({"metric": "train/ppo_kl", "series": [[0, 9.5]]}) + "\n")

            registry = CIRegistry(
                backend=HWBackend.CUDA,
                filename=str(test_file),
                est_time=60.0,
                suite=_TEST_SUITE,
                labels=["short"],
                hardware=["hopper"],
            )
            provenance = RunProvenance(
                commit_sha="a" * 40,
                pr_number=None,
                github_run_id=99,
                github_run_attempt=1,
                event_name="schedule",
                ref="refs/heads/main",
            )

            with caplog.at_level(logging.INFO, logger=ci_utils.logger.name):
                k8s_passed = ci_utils.run_gate_hook(
                    str(test_file),
                    str(record_path),
                    store=store,
                    registry=registry,
                    executing_suite=_TEST_SUITE,
                    write_baseline=False,
                    provenance=provenance,
                    enforce=False,
                    shadow_tag="k8s",
                )

            assert k8s_passed is True
            assert _count_runs_rows(db_path) == MIN_ENFORCED_BASELINES
            assert f"[CI Gate][shadow][k8s] {test_file}: NOT-TRUSTED" in caplog.text
            assert f"baseline_n={MIN_ENFORCED_BASELINES}" in caplog.text

            # Contrast: the exact same failing metric under Ray enforcement fails the gate.
            ray_passed = ci_utils.run_gate_hook(
                str(test_file),
                str(record_path),
                store=store,
                registry=registry,
                executing_suite=_TEST_SUITE,
                write_baseline=False,
                provenance=provenance,
                enforce=True,
            )
            assert ray_passed is False
            assert _count_runs_rows(db_path) == MIN_ENFORCED_BASELINES
        finally:
            store.close()

    def test_k8s_shadow_does_not_fail_when_metric_record_is_missing(self, tmp_path: Path, caplog):
        db_path = tmp_path / "history.db"
        store = SQLiteMetricHistoryStore(str(db_path), repository="radixark/miles")
        try:
            test_file = tmp_path / "test_enforced_missing.py"
            _write_enforced_test_file(test_file)
            _seed_ray_baselines(store, test_path=str(test_file), count=MIN_ENFORCED_BASELINES)

            registry = CIRegistry(
                backend=HWBackend.CUDA,
                filename=str(test_file),
                est_time=60.0,
                suite=_TEST_SUITE,
                labels=["short"],
                hardware=["hopper"],
            )
            provenance = RunProvenance(
                commit_sha="b" * 40,
                pr_number=None,
                github_run_id=100,
                github_run_attempt=1,
                event_name="schedule",
                ref="refs/heads/main",
            )

            with caplog.at_level(logging.WARNING, logger=ci_utils.logger.name):
                passed = ci_utils.run_gate_hook(
                    str(test_file),
                    None,
                    store=store,
                    registry=registry,
                    executing_suite=_TEST_SUITE,
                    write_baseline=False,
                    provenance=provenance,
                    enforce=False,
                    shadow_tag="k8s",
                )

            assert passed is True
            assert _count_runs_rows(db_path) == MIN_ENFORCED_BASELINES
            assert "[CI Gate][shadow][k8s] hook failed" in caplog.text
        finally:
            store.close()

    def test_run_suite_forces_shadow_gate_on_kubernetes_even_for_nightly_cadence(self, monkeypatch):
        """Even on nightly cadence (where policy.write_baseline=True), Kubernetes backend forces shadow mode."""
        monkeypatch.setenv("MILES_SCRIPT_CLUSTER_BACKEND", "kubernetes")
        monkeypatch.setattr(run_suite, "discover_ci_files", lambda: ["tests/e2e/dummy.py"])
        dummy_reg = CIRegistry(
            backend=HWBackend.CUDA,
            filename="tests/e2e/dummy.py",
            est_time=60.0,
            suite=_TEST_SUITE,
            labels=["short"],
            hardware=["hopper"],
        )
        monkeypatch.setattr(run_suite, "collect_tests", lambda files, sanity_check=True: [dummy_reg])
        monkeypatch.setattr(run_suite, "build_store_from_env", lambda backend: object())
        captured: dict[str, object] = {}

        def fake_run_unittest_files(ci_tests, **kwargs):
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(run_suite, "run_unittest_files", fake_run_unittest_files)

        args = argparse.Namespace(
            hw="cuda",
            suite=_TEST_SUITE,
            test_file=None,
            cadence="nightly",
            timeout_per_file=1800,
            continue_on_error=False,
            auto_partition_id=None,
            auto_partition_size=None,
            enable_retry=False,
            max_attempts=2,
            retry_wait_seconds=60,
            retry_timeout_increase=600,
            list_only=False,
            labels=[],
            github_output=None,
            match_all_labels=True,
        )
        rc = run_suite.run_a_suite(args)

        assert rc == 0
        assert captured["gate_write_baseline"] is False
        assert captured["gate_enforce"] is False
        assert captured["gate_shadow_tag"] == "k8s"
