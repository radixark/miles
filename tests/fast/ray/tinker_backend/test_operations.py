"""Operation ledger invariants: strict per-registration EXECUTION order under
out-of-order ARRIVAL (gap-buffered ordinals), fingerprinted idempotency,
cancel/fence/ack semantics, and backpressure."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import pytest

from miles.ray.tinker_backend.operations import OperationBackpressure, OperationLedger


def enqueue(ledger, op_id, ordinal, kind="forward_backward", name="A", reg="ra", payload=None):
    return ledger.enqueue(op_id, name, reg, ordinal, kind, payload)


class TestArrivalBuffering:
    def test_out_of_order_arrival_executes_in_ordinal_order(self):
        # The tinker SDK posts the first chunk of a large forward_backward
        # LAST: arrival 2,3,1 must execute 1,2,3.
        ledger = OperationLedger()
        enqueue(ledger, "op2", 2)
        enqueue(ledger, "op3", 3)
        assert ledger.claim_data_operation("A", "ra") is None  # gap below head
        enqueue(ledger, "op1", 1)
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "op1"
        ledger.complete("op1", {})
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "op2"

    def test_gap_blocks_control_claims_too(self):
        ledger = OperationLedger()
        enqueue(ledger, "opt2", 2, "optim_step")
        assert ledger.claimable_control_tenants() == []
        enqueue(ledger, "fb1", 1)
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "fb1"

    def test_duplicate_ordinal_is_a_conflict(self):
        ledger = OperationLedger()
        enqueue(ledger, "op1", 1)
        with pytest.raises(ValueError, match="already taken"):
            enqueue(ledger, "op1b", 1)

    def test_ordinals_start_at_one(self):
        ledger = OperationLedger()
        with pytest.raises(ValueError, match=">= 1"):
            enqueue(ledger, "op0", 0)


class TestFingerprintedIdempotency:
    def test_identical_retry_returns_the_original(self):
        ledger = OperationLedger()
        first = enqueue(ledger, "op1", 1, payload={"samples": [1]})
        retry = enqueue(ledger, "op1", 1, payload={"samples": [1]})
        assert retry == first

    def test_same_id_different_payload_is_a_conflict(self):
        ledger = OperationLedger()
        enqueue(ledger, "op1", 1, payload={"samples": [1]})
        with pytest.raises(ValueError, match="different content"):
            enqueue(ledger, "op1", 1, payload={"samples": [2]})

    def test_same_id_different_kind_is_a_conflict(self):
        ledger = OperationLedger()
        enqueue(ledger, "op1", 1, "forward_backward")
        with pytest.raises(ValueError, match="different content"):
            enqueue(ledger, "op1", 1, "optim_step")

    def test_same_id_different_ordinal_is_a_conflict(self):
        # A "retry" that moves the operation's sequence position is not a
        # retry: client and server would disagree on execution order.
        ledger = OperationLedger()
        enqueue(ledger, "op1", 1, payload={"samples": [1]})
        with pytest.raises(ValueError, match="different content"):
            enqueue(ledger, "op1", 2, payload={"samples": [1]})


class TestClaimViews:
    def test_claims_carry_the_request_payload(self):
        # The executor consumes the claim directly: a data claim without its
        # samples (or a control claim without its adam_params/tag/path) would
        # execute against an empty request.
        ledger = OperationLedger()
        enqueue(ledger, "fb", 1, payload={"samples": [{"tokens": [1, 2]}]})
        enqueue(ledger, "optim", 2, "optim_step", payload={"adam_params": {"learning_rate": 2e-4}})
        assert ledger.claim_data_operation("A", "ra")["payload"] == {"samples": [{"tokens": [1, 2]}]}
        ledger.complete("fb", {})
        assert ledger.claim_control_operation("A", "ra")["payload"] == {"adam_params": {"learning_rate": 2e-4}}
        # Poll results stay lean: get() never exposes the payload.
        assert "payload" not in ledger.get("optim")


class TestSerialization:
    def test_nothing_overtakes_an_open_operation(self):
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

    def test_control_claim_kind_filter(self):
        ledger = OperationLedger()
        enqueue(ledger, "save", 1, "save_state")
        assert ledger.claim_control_operation("A", "ra", kinds=("optim_step",)) is None
        assert ledger.claim_control_operation("A", "ra", kinds=("save_state",))["operation_id"] == "save"

    def test_registrations_are_independent(self):
        ledger = OperationLedger()
        enqueue(ledger, "a1", 1, name="A", reg="ra")
        enqueue(ledger, "b1", 1, name="B", reg="rb")
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "a1"
        assert ledger.claim_data_operation("B", "rb")["operation_id"] == "b1"


class TestPoisonedWindow:
    """#2258 §5: a failed forward_backward chunk poisons its whole gradient
    window; the window resets only at an optim_step that actually executed."""

    def fail_fb(self, ledger, op_id, ordinal, category="user"):
        enqueue(ledger, op_id, ordinal, "forward_backward")
        claimed = ledger.claim_data_operation("A", "ra")
        assert claimed["operation_id"] == op_id
        ledger.fail(op_id, "bad chunk", category)

    def complete_fb(self, ledger, op_id, ordinal):
        enqueue(ledger, op_id, ordinal, "forward_backward")
        ledger.claim_data_operation("A", "ra")
        ledger.complete(op_id, {})

    def test_failed_chunk_poisons_and_success_does_not(self):
        ledger = OperationLedger()
        self.complete_fb(ledger, "fb1", 1)
        assert ledger.poisoned_window_blocker("A", "ra", 2) is None
        self.fail_fb(ledger, "fb2", 2)
        blocker = ledger.poisoned_window_blocker("A", "ra", 3)
        assert blocker is not None and "ordinal 2" in blocker

    def test_executed_optim_delimits_the_window(self):
        ledger = OperationLedger()
        self.fail_fb(ledger, "fb1", 1)
        enqueue(ledger, "opt2", 2, "optim_step")
        ledger.claim_control_operation("A", "ra")
        ledger.fail("opt2", "window poisoned", "user")
        # Terminal alone is not enough: only the executor's confirmation that
        # the gradients were consumed (step/discard/veto) makes a delimiter.
        assert ledger.poisoned_window_blocker("A", "ra", 4) is not None
        ledger.mark_window_consumed("opt2")  # executed: it cleared the grads
        self.complete_fb(ledger, "fb3", 3)
        assert ledger.poisoned_window_blocker("A", "ra", 4) is None

    def test_cancelled_optim_is_no_delimiter_and_cancelled_fb_poisons(self):
        ledger = OperationLedger()
        self.fail_fb(ledger, "fb1", 1)
        enqueue(ledger, "opt2", 2, "optim_step")
        ledger.cancel("opt2")  # never executed: the partial gradients survive it
        assert ledger.poisoned_window_blocker("A", "ra", 3) is not None

        enqueue(ledger, "fb3", 3, "forward_backward")
        ledger.cancel("fb3")  # a cancelled fb is a non-success terminal: it poisons too
        blocker = ledger.poisoned_window_blocker("A", "ra", 4)
        assert blocker is not None and "ordinal 3" in blocker

    def test_failed_forward_does_not_poison(self):
        ledger = OperationLedger()
        enqueue(ledger, "fw1", 1, "forward")
        ledger.claim_data_operation("A", "ra")
        ledger.fail("fw1", "bad forward", "user")  # forward accumulates nothing
        assert ledger.poisoned_window_blocker("A", "ra", 2) is None

    def test_claims_stamp_was_claimed(self):
        ledger = OperationLedger()
        enqueue(ledger, "fb1", 1, "forward_backward")
        enqueue(ledger, "opt2", 2, "optim_step")
        assert ledger.by_id["fb1"].was_claimed is False
        ledger.claim_data_operation("A", "ra")
        assert ledger.by_id["fb1"].was_claimed is True
        ledger.complete("fb1", {})
        ledger.claim_control_operation("A", "ra")
        assert ledger.by_id["opt2"].was_claimed is True


class TestTerminals:
    def test_cancel_applies_only_to_queued_and_keeps_contiguity(self):
        ledger = OperationLedger()
        enqueue(ledger, "op1", 1)
        enqueue(ledger, "op2", 2)
        assert ledger.cancel("op2")["state"] == "CANCELLED"
        ledger.claim_data_operation("A", "ra")
        with pytest.raises(ValueError, match="only QUEUED"):
            ledger.cancel("op1")
        ledger.complete("op1", {})
        enqueue(ledger, "op3", 3)
        # the cancelled ordinal 2 still counts as arrived+terminal.
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "op3"

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


class TestBackpressureAndRetention:
    def test_pending_depth_backpressure(self):
        ledger = OperationLedger(max_pending=2)
        enqueue(ledger, "op1", 1)
        enqueue(ledger, "op2", 2)
        with pytest.raises(OperationBackpressure):
            enqueue(ledger, "op3", 3)

    def test_gap_filler_bypasses_the_pending_cap(self):
        # Arrival 2,3 fills the cap; without the bypass the hole at 1 would be
        # refused forever while 2 and 3 stay unclaimable: a permanent deadlock.
        ledger = OperationLedger(max_pending=2)
        enqueue(ledger, "op2", 2)
        enqueue(ledger, "op3", 3)
        assert ledger.claim_data_operation("A", "ra") is None
        enqueue(ledger, "op1", 1)  # admitted despite the cap
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "op1"
        # A beyond-the-tail arrival is NOT a gap filler: still backpressured.
        with pytest.raises(OperationBackpressure):
            enqueue(ledger, "op4", 4)

    def test_ack_releases_the_payload_and_result(self):
        # The ordinal slot survives the ack for contiguity, but the retained
        # record must not pin the (possibly large) payload/result forever.
        ledger = OperationLedger()
        enqueue(ledger, "op1", 1, payload={"samples": ["x" * 64]})
        ledger.claim_data_operation("A", "ra")
        ledger.complete("op1", {"logprobs": [[0.0] * 64]})
        ledger.ack("op1")
        residue = ledger.queues[("A", "ra")].by_ordinal[1]
        assert residue.payload == {} and residue.result is None

    def test_unacked_results_backpressure_and_ack_release(self):
        ledger = OperationLedger(max_unacked_results=1)
        enqueue(ledger, "op1", 1)
        ledger.claim_data_operation("A", "ra")
        ledger.complete("op1", {"ok": True})
        with pytest.raises(OperationBackpressure, match="unacknowledged"):
            enqueue(ledger, "op2", 2)
        ledger.ack("op1")
        enqueue(ledger, "op2", 2)
        # acked ordinal 1 still counts for contiguity.
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "op2"

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
        assert ledger.get("done")["result"] == {"kept": True}
        with pytest.raises(ValueError, match="fenced"):
            enqueue(ledger, "late", 3)

    def test_a_new_registration_of_the_same_name_starts_fresh(self):
        ledger = OperationLedger()
        enqueue(ledger, "old", 1, name="A", reg="ra")
        ledger.fence("A", "ra")
        fresh = enqueue(ledger, "new", 1, name="A", reg="rb")
        assert fresh["state"] == "QUEUED"


class TestRecordRejected:
    def test_rejected_ordinal_keeps_the_sequence_gap_free(self):
        # seq 1 ok, seq 2 rejected at the boundary, seq 3 ok: 3 must still
        # become claimable once 1 completes (2 is terminal on arrival).
        ledger = OperationLedger()
        enqueue(ledger, "op1", 1)
        rejected = ledger.record_rejected("op2", "A", "ra", 2, "optim_step", {"adam_params": {}}, "bad params")
        assert rejected["state"] == "FAILED"
        assert rejected["error_category"] == "user"
        enqueue(ledger, "op3", 3)
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "op1"
        ledger.complete("op1", {})
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "op3"

    def test_identical_retry_replays_the_terminal_record(self):
        ledger = OperationLedger()
        first = ledger.record_rejected("op1", "A", "ra", 1, "forward", {"samples": []}, "empty")
        again = ledger.record_rejected("op1", "A", "ra", 1, "forward", {"samples": []}, "empty")
        assert again == first

    def test_different_payload_at_the_same_id_is_a_conflict(self):
        ledger = OperationLedger()
        ledger.record_rejected("op1", "A", "ra", 1, "forward", {"samples": []}, "empty")
        with pytest.raises(ValueError, match="different content"):
            ledger.record_rejected("op1", "A", "ra", 1, "forward", {"samples": [1]}, "empty")

    def test_taken_ordinal_and_fence_still_refuse(self):
        ledger = OperationLedger()
        enqueue(ledger, "op1", 1)
        with pytest.raises(ValueError, match="already taken"):
            ledger.record_rejected("op1b", "A", "ra", 1, "forward", {}, "x")
        ledger.fence("A", "ra")
        with pytest.raises(ValueError, match="fenced"):
            ledger.record_rejected("op2", "A", "ra", 2, "forward", {}, "x")

    def test_rejected_flood_hits_the_unacked_results_budget(self):
        # An invalid-request flood must not grow born-terminal records without
        # bound: past the budget it backpressures like any unretrieved pile-up.
        ledger = OperationLedger(max_unacked_results=8)
        accepted = 0
        for i in range(1, 1001):
            try:
                ledger.record_rejected(f"op{i}", "A", "ra", i, "forward_backward", {"i": i}, "bad")
                accepted += 1
            except OperationBackpressure:
                break
        assert accepted == 8
        assert ledger.queues[("A", "ra")].unacked_terminal_count() == 8
        # Acking terminal records frees budget for the retried rejection.
        ledger.ack("op1")
        assert ledger.record_rejected("op9", "A", "ra", 9, "forward_backward", {"i": 9}, "bad")["state"] == "FAILED"

    def test_rejected_hole_filler_bypasses_the_unacked_budget(self):
        # Refusing the blocking-gap rejection would deadlock the buffered tail.
        ledger = OperationLedger(max_unacked_results=1)
        enqueue(ledger, "fb1", 1)
        enqueue(ledger, "fb3", 3)  # buffered above the future hole
        ledger.claim_data_operation("A", "ra")
        ledger.fail("fb1", "boom", "user")  # the budget is now full
        with pytest.raises(OperationBackpressure):
            ledger.record_rejected("tail", "A", "ra", 4, "forward_backward", {}, "bad")
        # ...but ordinal 2 is the blocking gap below buffered fb3: always admitted.
        ledger.record_rejected("hole", "A", "ra", 2, "forward_backward", {}, "bad")
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "fb3"

    def test_born_terminal_optim_is_no_window_delimiter(self):
        # A rejected optim_step never executed: it cleared nothing, so a
        # poisoned window stays poisoned across it.
        ledger = OperationLedger()
        enqueue(ledger, "fb1", 1)
        ledger.claim_data_operation("A", "ra")
        ledger.fail("fb1", "bad chunk", "user")
        ledger.record_rejected("opt2", "A", "ra", 2, "optim_step", {"adam_params": {"beta1": 9}}, "bad params")
        assert ledger.poisoned_window_blocker("A", "ra", 3) is not None

    def test_rejection_bypasses_backpressure_like_a_hole_filler(self):
        ledger = OperationLedger(max_pending=1)
        enqueue(ledger, "op1", 1)
        with pytest.raises(OperationBackpressure):
            enqueue(ledger, "op3", 3)
        # A terminal-on-arrival record occupies no execution capacity.
        assert ledger.record_rejected("op2", "A", "ra", 2, "forward", {}, "x")["state"] == "FAILED"

    def test_rejected_record_is_ackable(self):
        ledger = OperationLedger()
        ledger.record_rejected("op1", "A", "ra", 1, "forward", {}, "x")
        ledger.ack("op1")
        assert ledger.get("op1") is None
        enqueue(ledger, "op2", 2)
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "op2"
