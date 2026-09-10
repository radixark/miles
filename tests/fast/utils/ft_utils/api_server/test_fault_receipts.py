import pytest

from miles.utils.ft_utils.api_server.fault_receipts import FaultExitSubmission, FaultReceiptRegistry
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import FaultTarget


def test_request_replay_is_idempotent_and_cannot_change_target_or_receipt() -> None:
    """Retries preserve the original target and cannot dispatch or replace its evidence twice."""
    registry = FaultReceiptRegistry()
    target = FaultTarget(cell_id="actor-0", sub_index=0, workers_hash="generation")
    assert registry.register(request_id="request", target=target, mode=FailureMode.SIGKILL)
    assert not registry.register(request_id="request", target=target, mode=FailureMode.SIGKILL)
    assert registry.read("request") is None
    with pytest.raises(ValueError):
        registry.register(
            request_id="request",
            target=target.model_copy(update={"workers_hash": "replacement"}),
            mode=FailureMode.SIGKILL,
        )
    receipt = registry.publish(request_id="request", submission=FaultExitSubmission(exited_pids=[42]))
    assert receipt.target == target and receipt.request_id == "request"
    assert registry.publish(request_id="request", submission=FaultExitSubmission(exited_pids=[42])) == receipt
    with pytest.raises(ValueError):
        registry.publish(request_id="request", submission=FaultExitSubmission(exited_pids=[43]))
    assert registry.read("request") == receipt


def test_unregistered_receipt_is_rejected() -> None:
    """Unsolicited evidence cannot create a successful fault request."""
    registry = FaultReceiptRegistry()
    with pytest.raises(KeyError):
        registry.publish(request_id="unknown", submission=FaultExitSubmission(exited_pids=[42]))
