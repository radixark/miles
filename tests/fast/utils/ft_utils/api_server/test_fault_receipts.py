import os

import pytest

from miles.utils.ft_utils.api_server.fault_receipts import (
    FaultDeadlockSubmission,
    FaultExitSubmission,
    FaultReceiptRegistry,
    FaultStopSubmission,
)
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


@pytest.mark.parametrize("mode", [FailureMode.SIGSTOP, FailureMode.SIGKILL])
def test_stop_and_exit_evidence_cannot_substitute_for_each_other(mode: FailureMode) -> None:
    """A terminated target does not prove a stop fault was applied, or vice versa."""
    registry = FaultReceiptRegistry()
    target = FaultTarget(cell_id="actor-0", sub_index=0, workers_hash="generation")
    registry.register(request_id="request", target=target, mode=mode)
    stopped = FaultStopSubmission(stopped_pids=[42])
    exited = FaultExitSubmission(exited_pids=[42])
    wrong = exited if mode is FailureMode.SIGSTOP else stopped
    correct = stopped if mode is FailureMode.SIGSTOP else exited
    with pytest.raises(ValueError):
        registry.publish(request_id="request", submission=wrong)
    assert registry.read("request") is None
    receipt = registry.publish(request_id="request", submission=correct)
    assert receipt.target == target
    assert registry.publish(request_id="request", submission=correct) == receipt


@pytest.mark.parametrize("holds_gil", [False, True])
def test_deadlock_receipt_requires_matching_blocking_scope(holds_gil: bool) -> None:
    """Thread-only blocking cannot satisfy a GIL-blocking request or vice versa."""
    registry = FaultReceiptRegistry()
    target = FaultTarget(cell_id="actor-0", sub_index=0, workers_hash="generation")
    mode = FailureMode.DEADLOCK if holds_gil else FailureMode.THREAD_DEADLOCK
    registry.register(request_id="request", target=target, mode=mode)
    submission = FaultDeadlockSubmission(
        blocked_pid=42,
        blocked_tid=43,
        lock_device=os.makedev(0, 1),
        lock_inode=99,
        holds_gil=holds_gil,
        lock_evidence=["3: FLOCK ADVISORY WRITE 42 00:01:99 0 EOF", "3: -> FLOCK ADVISORY WRITE 42 00:01:99 0 EOF"],
    )
    with pytest.raises(ValueError, match="blocking scope"):
        registry.publish(request_id="request", submission=submission.model_copy(update={"holds_gil": not holds_gil}))
    assert registry.read("request") is None
    receipt = registry.publish(request_id="request", submission=submission)
    assert receipt.mode is mode
    assert receipt.target == target
    assert receipt.lock_evidence == submission.lock_evidence


def test_unregistered_receipt_is_rejected() -> None:
    """Unsolicited evidence cannot create a successful fault request."""
    registry = FaultReceiptRegistry()
    with pytest.raises(KeyError):
        registry.publish(request_id="unknown", submission=FaultExitSubmission(exited_pids=[42]))
