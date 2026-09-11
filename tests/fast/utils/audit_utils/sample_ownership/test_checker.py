from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from miles.utils.audit_utils.sample_ownership import checker as checker_module
from miles.utils.audit_utils.sample_ownership.checker import SampleOwnershipChecker


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
