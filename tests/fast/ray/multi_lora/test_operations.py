"""Operation ledger invariants: strict per-registration serialization,
ordinal ordering, idempotency, cancel/fence/ack semantics, backpressure."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import pytest

from miles.ray.multi_lora.operations import OperationBackpressure, OperationLedger


def enqueue(ledger, op_id, ordinal, kind="forward_backward", name="A", reg="ra", payload=None):
    return ledger.enqueue(op_id, name, reg, ordinal, kind, payload)


class TestEnqueue:
    def test_ordinals_must_arrive_strictly_increasing(self):
        ledger = OperationLedger()
        enqueue(ledger, "op1", 1)
        enqueue(ledger, "op3", 3)  # gaps are legal
        with pytest.raises(ValueError, match="strictly increasing"):
            enqueue(ledger, "op2", 2)

    def test_retry_of_a_known_id_is_idempotent(self):
        ledger = OperationLedger()
        first = enqueue(ledger, "op1", 1)
        retry = enqueue(ledger, "op1", 1)
        assert retry == first
        with pytest.raises(ValueError, match="different identity"):
            enqueue(ledger, "op1", 9)

    def test_pending_depth_backpressure(self):
        ledger = OperationLedger(max_pending=2)
        enqueue(ledger, "op1", 1)
        enqueue(ledger, "op2", 2)
        with pytest.raises(OperationBackpressure):
            enqueue(ledger, "op3", 3)

    def test_unacked_results_backpressure(self):
        ledger = OperationLedger(max_unacked_results=1)
        enqueue(ledger, "op1", 1)
        ledger.claim_data_operation("A", "ra")
        ledger.complete("op1", {"ok": True})
        with pytest.raises(OperationBackpressure, match="unacknowledged"):
            enqueue(ledger, "op2", 2)
        ledger.ack("op1")
        enqueue(ledger, "op2", 2)


class TestSerialization:
    def test_nothing_overtakes_an_open_operation(self):
        # fb(1), optim(2): the control head must not be claimable while the
        # data op is anywhere between claim and terminal — this is the ordering
        # that makes client fb->optim sequences safe.
        ledger = OperationLedger()
        enqueue(ledger, "fb", 1, "forward_backward")
        enqueue(ledger, "optim", 2, "optim_step")
        assert ledger.claim_control_operation("A", "ra") is None
        claimed = ledger.claim_data_operation("A", "ra")
        assert claimed["operation_id"] == "fb"
        assert ledger.claim_control_operation("A", "ra") is None
        assert ledger.claim_data_operation("A", "ra") is None  # fb still open
        ledger.complete("fb", {})
        assert ledger.claim_control_operation("A", "ra")["operation_id"] == "optim"

    def test_control_head_blocks_data_claims(self):
        ledger = OperationLedger()
        enqueue(ledger, "optim", 1, "optim_step")
        enqueue(ledger, "fb", 2, "forward_backward")
        assert ledger.claim_data_operation("A", "ra") is None
        assert ("A", "ra") in ledger.claimable_control_tenants()
        ledger.claim_control_operation("A", "ra")
        ledger.complete("optim", {})
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "fb"

    def test_registrations_are_independent(self):
        ledger = OperationLedger()
        enqueue(ledger, "a1", 1, name="A", reg="ra")
        enqueue(ledger, "b1", 1, name="B", reg="rb")
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "a1"
        assert ledger.claim_data_operation("B", "rb")["operation_id"] == "b1"


class TestTerminals:
    def test_cancel_applies_only_to_queued(self):
        ledger = OperationLedger()
        enqueue(ledger, "op1", 1)
        enqueue(ledger, "op2", 2)
        assert ledger.cancel("op2")["state"] == "CANCELLED"
        ledger.claim_data_operation("A", "ra")
        with pytest.raises(ValueError, match="only QUEUED"):
            ledger.cancel("op1")

    def test_cancelled_head_unblocks_the_next_operation(self):
        ledger = OperationLedger()
        enqueue(ledger, "op1", 1)
        enqueue(ledger, "op2", 2)
        ledger.cancel("op1")
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "op2"

    def test_fail_records_error_and_category(self):
        ledger = OperationLedger()
        enqueue(ledger, "op1", 1)
        ledger.claim_data_operation("A", "ra")
        ledger.fail("op1", "bad payload", "user")
        view = ledger.get("op1")
        assert view["state"] == "FAILED" and view["error_category"] == "user"

    def test_double_terminal_is_rejected(self):
        ledger = OperationLedger()
        enqueue(ledger, "op1", 1)
        ledger.claim_data_operation("A", "ra")
        ledger.complete("op1", {})
        with pytest.raises(ValueError, match="already terminal"):
            ledger.fail("op1", "late failure")


class TestFencing:
    def test_fence_fails_open_ops_and_refuses_new_ones(self):
        ledger = OperationLedger()
        enqueue(ledger, "done", 1)
        ledger.claim_data_operation("A", "ra")
        ledger.complete("done", {"kept": True})
        enqueue(ledger, "pending", 2)
        assert ledger.fence("A", "ra") == ["pending"]
        assert ledger.get("pending")["state"] == "FAILED"
        assert ledger.get("pending")["error_category"] == "user"
        # Terminal results of the dead registration stay retrievable.
        assert ledger.get("done")["result"] == {"kept": True}
        with pytest.raises(ValueError, match="fenced"):
            enqueue(ledger, "late", 3)

    def test_a_new_registration_of_the_same_name_starts_fresh(self):
        ledger = OperationLedger()
        enqueue(ledger, "old", 5, name="A", reg="ra")
        ledger.fence("A", "ra")
        fresh = enqueue(ledger, "new", 1, name="A", reg="rb")  # new tenant, ordinals restart
        assert fresh["state"] == "QUEUED"


class TestAck:
    def test_ack_drops_only_terminal_records(self):
        ledger = OperationLedger()
        enqueue(ledger, "op1", 1)
        with pytest.raises(ValueError, match="ack applies to terminal"):
            ledger.ack("op1")
        ledger.claim_data_operation("A", "ra")
        ledger.complete("op1", {})
        ledger.ack("op1")
        assert ledger.get("op1") is None
        ledger.ack("op1")  # idempotent
