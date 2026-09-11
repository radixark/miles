from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from miles.utils.audit_utils.sample_ownership import checker as checker_module
from miles.utils.audit_utils.sample_ownership.checker import SampleOwnershipChecker


@pytest.mark.parametrize("ci_test", [False, True])
async def test_check_failure_is_logged_or_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, ci_test: bool
) -> None:
    """CI propagates check failures while normal training logs their traceback."""
    checker = SampleOwnershipChecker(
        args=SimpleNamespace(
            enable_sample_ownership_checker=True, sample_ownership_check_interval_seconds=0, ci_test=ci_test
        )
    )

    async def fail(*, rollout_id: int) -> None:
        raise ValueError("missing training outcome")

    monkeypatch.setattr(checker, "_check", fail)
    if ci_test:
        with pytest.raises(ValueError, match="missing training outcome"):
            await checker.check(rollout_id=7)
    else:
        await checker.check(rollout_id=7)
        assert any(record.levelname == "ERROR" and record.exc_info for record in caplog.records)


async def test_disabled_checker_does_no_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabled checking does not collect witnesses or inspect the interval."""
    checker = SampleOwnershipChecker(args=SimpleNamespace(enable_sample_ownership_checker=False))

    async def fail(*, rollout_id: int) -> None:
        pytest.fail("disabled checker ran")

    monkeypatch.setattr(checker, "_check", fail)
    await checker.check(rollout_id=7)


async def test_check_interval_skips_calls_until_due(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated rollout calls check once per configured interval."""
    checker = SampleOwnershipChecker(
        args=SimpleNamespace(
            enable_sample_ownership_checker=True, sample_ownership_check_interval_seconds=10, ci_test=True
        )
    )
    observed = []
    times = iter([0, 9, 10])
    monkeypatch.setattr(checker_module, "time", SimpleNamespace(monotonic=lambda: next(times)))

    async def record(*, rollout_id: int) -> None:
        observed.append(rollout_id)

    monkeypatch.setattr(checker, "_check", record)
    for rollout_id in [1, 2, 3]:
        await checker.check(rollout_id=rollout_id)
    assert observed == [1, 3]


@pytest.mark.parametrize("has_mature_samples", [False, True])
async def test_completed_step_window_controls_analysis(
    monkeypatch: pytest.MonkeyPatch, has_mature_samples: bool
) -> None:
    """The checker analyzes a published cohort without calling a trainer."""
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc) if has_mature_samples else None
    marker = SimpleNamespace(mature_before=cutoff)
    snapshot = SimpleNamespace(marker=marker, snapshots=["published"])
    calls = []

    class Store:
        def read_current(self) -> SimpleNamespace:
            return snapshot

        def read_history(self) -> list[Any]:
            return ["issued"]

    def analyze(events: list[Any], **kwargs: Any) -> None:
        assert events == ["issued", "published", marker]
        assert kwargs["now"] == cutoff == kwargs["process_started_at"]
        calls.append("analysis")

    monkeypatch.setattr(checker_module, "get_event_logger", lambda: None)
    monkeypatch.setattr(checker_module, "SampleOwnershipEventStore", lambda logger: Store())
    monkeypatch.setattr(checker_module, "run_sample_ownership_analysis", analyze)
    checker = SampleOwnershipChecker(args=SimpleNamespace(save_debug_event_data="/events"))
    await checker._check(rollout_id=7)
    assert calls == ["analysis"]


async def test_no_completed_training_step_defers_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first rollout has no completed trainer cohort to analyze."""
    monkeypatch.setattr(checker_module, "get_event_logger", lambda: None)
    monkeypatch.setattr(
        checker_module, "SampleOwnershipEventStore", lambda logger: SimpleNamespace(read_current=lambda: None)
    )
    monkeypatch.setattr(
        checker_module, "run_sample_ownership_analysis", lambda *args, **kwargs: pytest.fail("No cohort")
    )
    checker = SampleOwnershipChecker(args=SimpleNamespace(save_debug_event_data="/events"))
    await checker._check(rollout_id=0)
