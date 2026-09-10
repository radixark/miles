from datetime import datetime, timedelta, timezone

import pytest

from miles.utils.audit_utils.sample_ownership.step_window import SampleOwnershipStepWindow


class TestSampleOwnershipStepWindow:
    def test_elapsed_time_without_completed_steps_does_not_mature_samples(self) -> None:
        """A slow or stalled trainer does not consume the step grace period."""
        window = SampleOwnershipStepWindow(2)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        window.complete_step(started_at=start)

        assert window.mature_before(now=start + timedelta(days=1)) is None

    def test_the_cutoff_advances_with_completed_steps(self) -> None:
        """Every check keeps all samples older than the last two complete steps eligible."""
        window = SampleOwnershipStepWindow(2)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        window.complete_step(started_at=start)
        window.complete_step(started_at=start + timedelta(seconds=1))

        assert window.mature_before(now=start + timedelta(seconds=10)) == start

        window.complete_step(started_at=start + timedelta(seconds=8))

        assert window.mature_before(now=start + timedelta(seconds=10)) == start + timedelta(seconds=1)
        assert SampleOwnershipStepWindow(2).mature_before(now=start + timedelta(days=1)) is None

    def test_zero_grace_checks_through_the_current_snapshot(self) -> None:
        """An explicit zero grace needs no completed training steps."""
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)

        assert SampleOwnershipStepWindow(0).mature_before(now=now) == now

    def test_negative_grace_is_rejected(self) -> None:
        """Negative step windows cannot silently disable checking."""
        with pytest.raises(ValueError, match="non-negative"):
            SampleOwnershipStepWindow(-1)
