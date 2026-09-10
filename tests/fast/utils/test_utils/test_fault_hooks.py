from pathlib import Path

import pytest

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import FaultHookEvent
from miles.utils.test_utils import fault_hooks
from miles.utils.test_utils.fault_hooks import FaultHookRegistry, FaultHookRequest


class TestFaultHookRegistry:
    @pytest.mark.parametrize("arm_first", [False, True])
    def test_cancelled_request_cannot_rearm_or_fire(
        self, fault_hook_registry: FaultHookRegistry, arm_first: bool
    ) -> None:
        """Retrying a cancelled request cannot revive a fault."""
        registry = fault_hook_registry
        request = FaultHookRequest(
            request_id="cancelled", instance_id=registry.instance_id, hook="trainer_before_all_gather", mode="exit"
        )
        if arm_first:
            registry.arm(request)
        cancelled = registry.cancel(request)

        assert cancelled.status == "cancelled"
        assert registry.arm(request) == cancelled
        registry.reach(hook=request.hook, rollout_id=1, attempt=0)
        assert registry.read(request_id=request.request_id, instance_id=registry.instance_id) == cancelled

    def test_replacement_process_rejects_old_request(self, fault_hook_registry: FaultHookRegistry) -> None:
        """A recreated worker cannot accept an earlier incarnation's command."""
        request = FaultHookRequest(
            request_id="stale",
            instance_id=FaultHookRegistry().instance_id,
            hook="trainer_before_all_gather",
            mode="exit",
        )
        with pytest.raises(ValueError, match="incarnation"):
            fault_hook_registry.arm(request)

    def test_expired_arm_cannot_fire(
        self, fault_hook_registry: FaultHookRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unobserved expiry still prevents a later hook hit."""
        registry = fault_hook_registry
        monkeypatch.setattr(fault_hooks.time, "monotonic", lambda: 10.0)
        request = FaultHookRequest(
            request_id="expired", instance_id=registry.instance_id, hook="trainer_before_all_gather", mode="exit"
        )
        registry.arm(request)
        monkeypatch.setattr(fault_hooks.time, "monotonic", lambda: 70.0)

        registry.reach(hook=request.hook, rollout_id=2, attempt=1)
        assert registry.read(request_id=request.request_id, instance_id=registry.instance_id).status == "expired"

    def test_firing_wins_cancellation_and_is_recorded_before_fault(
        self, fault_hook_registry: FaultHookRegistry, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Cancellation after dispatch cannot claim to have prevented the fault."""
        registry = fault_hook_registry
        request = FaultHookRequest(
            request_id="fired", instance_id=registry.instance_id, hook="trainer_before_all_gather", mode="exit"
        )
        registry.arm(request)

        def terminate(*, mode: str, request_id: str, receipt_url: str | None) -> None:
            assert mode == "exit"
            assert request_id == request.request_id
            record = registry.cancel(request)
            assert record.status == "fired"
            events = [event for event in read_events(tmp_path) if isinstance(event, FaultHookEvent)]
            assert events[-1].status == "fired"
            assert events[-1].rollout_id == 3
            assert events[-1].attempt == 2
            raise SystemExit(1)

        monkeypatch.setattr(fault_hooks, "inject_fault", terminate)
        with pytest.raises(SystemExit):
            registry.reach(hook=request.hook, rollout_id=3, attempt=2)
        registry.reach(hook=request.hook, rollout_id=4, attempt=0)

    def test_failed_fault_dispatch_is_not_reported_as_applied(
        self, fault_hook_registry: FaultHookRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A returning fault action leaves an explicit failure record."""
        registry = fault_hook_registry
        request = FaultHookRequest(
            request_id="failed", instance_id=registry.instance_id, hook="trainer_before_all_gather", mode="exit"
        )
        registry.arm(request)
        monkeypatch.setattr(fault_hooks, "inject_fault", lambda **kwargs: None)
        with pytest.raises(RuntimeError, match="unexpectedly returned"):
            registry.reach(hook=request.hook, rollout_id=1, attempt=0)
        assert registry.read(request_id=request.request_id, instance_id=registry.instance_id).status == "failed"

    def test_second_arm_and_conflicting_retries_are_rejected(self, fault_hook_registry: FaultHookRegistry) -> None:
        """A live arm cannot be overwritten by another request or changed parameters."""
        registry = fault_hook_registry
        request = FaultHookRequest(
            request_id="first", instance_id=registry.instance_id, hook="trainer_before_all_gather", mode="exit"
        )
        original = registry.arm(request)
        assert registry.arm(request) == original
        with pytest.raises(ValueError, match="already armed"):
            registry.arm(request.model_copy(update={"request_id": "second"}))
        with pytest.raises(ValueError, match="reused"):
            registry.arm(request.model_copy(update={"mode": "sigkill"}))
        registry.reach(hook="trainer_before_weight_send", rollout_id=1, attempt=0)
        assert registry.read(request_id=request.request_id, instance_id=registry.instance_id) == original
