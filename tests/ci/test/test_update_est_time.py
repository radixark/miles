from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import tests.ci.runtime_estimate.update_est_time as update_est_time_module
from tests.ci.ci_register import CIRegistry, HWBackend, register_cpu_ci
from tests.ci.runtime_estimate.runtime_history import RuntimeSample
from tests.ci.runtime_estimate.update_est_time import (
    EstimateChange,
    RuntimeEstimate,
    bucket_estimate,
    build_estimates,
    inclusive_percentile,
    render_file_updates,
    render_report,
    update_registered_estimates,
)

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


def _sample(elapsed: float, *, day: int, run_id: int = 1, run_attempt: int = 1) -> RuntimeSample:
    return RuntimeSample(
        test_path="tests/e2e/test_example.py",
        backend="CUDA",
        suite="stage-c-8-gpu-h100",
        elapsed_seconds=elapsed,
        github_run_id=run_id,
        github_run_attempt=run_attempt,
        recorded_at=datetime(2026, 8, 1, tzinfo=UTC) + timedelta(days=day),
    )


def test_inclusive_p90_and_bucket_boundaries():
    assert inclusive_percentile([10, 20, 30], 0.9) == pytest.approx(28)
    assert bucket_estimate(0) == 10
    assert bucket_estimate(28) == 30
    assert bucket_estimate(200) == 200
    assert bucket_estimate(201) == 300


def test_build_estimates_requires_three_samples():
    assert build_estimates([_sample(10, day=1), _sample(20, day=2)]) == {}
    estimate = build_estimates([_sample(10, day=1), _sample(20, day=2), _sample(30, day=3)])
    assert estimate[("tests/e2e/test_example.py", "CUDA", "stage-c-8-gpu-h100")].est_time == 30


def test_build_estimates_caps_at_fifteen_newest_samples():
    samples = [_sample(10, day=day, run_id=day) for day in range(1, 16)]
    samples.append(_sample(10_000, day=0, run_id=99))
    estimate = build_estimates(samples)[("tests/e2e/test_example.py", "CUDA", "stage-c-8-gpu-h100")]
    assert estimate.sample_count == 15
    assert estimate.p90_seconds == 10
    assert (99, 1) not in estimate.run_attempts


def test_build_estimates_keeps_distinct_workflow_attempts_for_one_run():
    samples = [
        _sample(10, day=1, run_id=7, run_attempt=1),
        _sample(20, day=2, run_id=7, run_attempt=2),
        _sample(30, day=3, run_id=8, run_attempt=1),
    ]
    estimate = build_estimates(samples)[("tests/e2e/test_example.py", "CUDA", "stage-c-8-gpu-h100")]
    assert estimate.run_attempts == ((8, 1), (7, 2), (7, 1))


def test_render_updates_keyword_and_positional_literals_only(tmp_path: Path):
    path = tmp_path / "test_example.py"
    path.write_text(
        "from tests.ci.ci_register import register_cpu_ci, register_cuda_ci\n"
        'label = "中文"; register_cuda_ci(est_time=60, suite="stage-c-8-gpu-h100", labels=[])\n'
        'register_cuda_ci(70, "stage-c-8-gpu-h200", labels=[])\n'
        'register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])\n',
        encoding="utf-8",
    )
    estimates = {
        (str(path), "CUDA", "stage-c-8-gpu-h100"): RuntimeEstimate(3, 88, 90, ((1, 1), (2, 1), (3, 1))),
        (str(path), "CUDA", "stage-c-8-gpu-h200"): RuntimeEstimate(3, 115, 120, ((1, 1), (2, 1), (3, 1))),
    }

    rendered, changes = render_file_updates(str(path), estimates)
    text = rendered.decode("utf-8")

    assert 'label = "中文"; register_cuda_ci(est_time=90, suite="stage-c-8-gpu-h100", labels=[])' in text
    assert 'register_cuda_ci(120, "stage-c-8-gpu-h200", labels=[])' in text
    assert 'register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])' in text
    assert [(change.old_est_time, change.new_est_time) for change in changes] == [(60, 90), (70, 120)]


def test_render_leaves_unmatched_registration_byte_identical(tmp_path: Path):
    path = tmp_path / "test_example.py"
    source = (
        "from tests.ci.ci_register import register_cuda_ci\n" 'register_cuda_ci(70, "stage-c-8-gpu-h200", labels=[])\n'
    )
    path.write_text(source, encoding="utf-8")
    rendered, changes = render_file_updates(str(path), {})
    assert rendered == source.encode()
    assert changes == []


def test_render_updates_only_active_registration_when_disabled_shares_identity(tmp_path: Path):
    path = tmp_path / "test_example.py"
    path.write_text(
        "from tests.ci.ci_register import register_cuda_ci\n"
        'register_cuda_ci(est_time=60, suite="stage-c-8-gpu-h100", labels=[], disabled="flaky")\n'
        'register_cuda_ci(est_time=70, suite="stage-c-8-gpu-h100", labels=[])\n',
        encoding="utf-8",
    )
    identity = (str(path), "CUDA", "stage-c-8-gpu-h100")
    estimate = RuntimeEstimate(3, 88, 90, ((1, 1), (2, 1), (3, 1)))

    rendered, changes = render_file_updates(str(path), {identity: estimate})

    text = rendered.decode()
    assert 'register_cuda_ci(est_time=60, suite="stage-c-8-gpu-h100", labels=[], disabled="flaky")' in text
    assert 'register_cuda_ci(est_time=90, suite="stage-c-8-gpu-h100", labels=[])' in text
    assert [(change.old_est_time, change.new_est_time) for change in changes] == [(70, 90)]


def test_render_rejects_duplicate_active_registration_identity(tmp_path: Path):
    path = tmp_path / "test_example.py"
    path.write_text(
        "from tests.ci.ci_register import register_cuda_ci\n"
        'register_cuda_ci(est_time=60, suite="stage-c-8-gpu-h100", labels=[])\n'
        'register_cuda_ci(est_time=70, suite="stage-c-8-gpu-h100", labels=[])\n',
        encoding="utf-8",
    )
    identity = (str(path), "CUDA", "stage-c-8-gpu-h100")
    estimate = RuntimeEstimate(3, 88, 90, ((1, 1), (2, 1), (3, 1)))

    with pytest.raises(ValueError, match="expected exactly one active CUDA registration.*found 2"):
        render_file_updates(str(path), {identity: estimate})


def test_render_rejects_target_without_active_registration(tmp_path: Path):
    path = tmp_path / "test_example.py"
    path.write_text(
        "from tests.ci.ci_register import register_cuda_ci\n"
        'register_cuda_ci(est_time=60, suite="stage-c-8-gpu-h100", labels=[], disabled="flaky")\n',
        encoding="utf-8",
    )
    identity = (str(path), "CUDA", "stage-c-8-gpu-h100")
    estimate = RuntimeEstimate(3, 88, 90, ((1, 1), (2, 1), (3, 1)))

    with pytest.raises(ValueError, match="expected exactly one active CUDA registration.*found 0"):
        render_file_updates(str(path), {identity: estimate})


def test_dry_run_is_repeatable_and_byte_identical(tmp_path: Path, monkeypatch):
    path = tmp_path / "test_example.py"
    path.write_text(
        "from tests.ci.ci_register import register_cuda_ci\n"
        'register_cuda_ci(est_time=60, suite="stage-c-8-gpu-h100", labels=[])\n',
        encoding="utf-8",
    )
    registry = CIRegistry(HWBackend.CUDA, str(path), 60, "stage-c-8-gpu-h100")
    monkeypatch.setattr(update_est_time_module, "_active_cuda_e2e_registrations", lambda: [registry])
    identity = (str(path), "CUDA", "stage-c-8-gpu-h100")
    estimates = {identity: RuntimeEstimate(3, 88, 90, ((1, 1), (2, 1), (3, 1)))}
    original = path.read_bytes()

    first = update_registered_estimates(estimates, dry_run=True)
    after_first = path.read_bytes()
    second = update_registered_estimates(estimates, dry_run=True)

    assert first == second
    assert original == after_first == path.read_bytes()


def test_report_uses_half_open_window_and_attempt_links(monkeypatch):
    cutoff = datetime(2026, 7, 22, tzinfo=UTC)
    upper = datetime(2026, 8, 12, tzinfo=UTC)
    change = EstimateChange(
        "tests/e2e/test_example.py",
        "stage-c-8-gpu-h100",
        60,
        90,
        3,
        88,
        ((123, 1), (123, 2)),
    )
    monkeypatch.setenv("GITHUB_REPOSITORY", "radixark/miles")

    report = render_report([change], cutoff, upper)

    assert f"`[{cutoff.isoformat()}, {upper.isoformat()})`" in report
    assert "https://github.com/radixark/miles/actions/runs/123/attempts/1" in report
    assert "https://github.com/radixark/miles/actions/runs/123/attempts/2" in report


def test_main_uses_as_of_midnight_as_exclusive_upper(monkeypatch, capsys):
    captured = {}

    class _Store:
        def recent_successful_attempts(self, cutoff, before, limit):
            captured["query"] = (cutoff, before, limit)
            return []

    monkeypatch.setattr(update_est_time_module, "NeonRuntimeHistoryStore", _Store)
    monkeypatch.setattr(update_est_time_module, "update_registered_estimates", lambda estimates, *, dry_run: [])
    monkeypatch.setattr(sys, "argv", ["update_est_time.py", "--as-of", "2026-08-12", "--dry-run"])

    assert update_est_time_module.main() == 0
    assert captured["query"] == (
        datetime(2026, 7, 22, tzinfo=UTC),
        datetime(2026, 8, 12, tzinfo=UTC),
        15,
    )
    assert "`[2026-07-22T00:00:00+00:00, 2026-08-12T00:00:00+00:00)`" in capsys.readouterr().out


def test_workflow_publish_condition_uses_explicit_boolean_comparison():
    workflow = (Path(__file__).resolve().parents[3] / ".github" / "workflows" / "ci-runtime-est-time.yml").read_text()
    publish_condition = workflow.split("- name: Publish pull request", 1)[1].split("env:", 1)[0]
    assert "inputs.dry_run == false" in publish_condition
    assert "!inputs.dry_run" not in publish_condition
