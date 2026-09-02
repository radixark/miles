import pytest

from miles.ray.multi_lora.operations import OperationBackpressure, OperationLedger


def enqueue(ledger, op_id, ordinal, kind="forward_backward", name="A", reg="ra", payload=None):
    return ledger.enqueue(op_id, name, reg, ordinal, kind, payload)


class TestArrivalBuffering:
    def test_out_of_order_arrival_executes_in_ordinal_order(self):
        ledger = OperationLedger()
        enqueue(ledger, "op2", 2)
        enqueue(ledger, "op3", 3)
        assert ledger.claim_data_operation("A", "ra") is None
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
        ledger = OperationLedger()
        enqueue(ledger, "op1", 1, payload={"samples": [1]})
        with pytest.raises(ValueError, match="different content"):
            enqueue(ledger, "op1", 2, payload={"samples": [1]})


class TestClaimViews:
    def test_claims_carry_the_request_payload(self):
        ledger = OperationLedger()
        enqueue(ledger, "fb", 1, payload={"samples": [{"tokens": [1, 2]}]})
        enqueue(ledger, "optim", 2, "optim_step", payload={"adam_params": {"learning_rate": 2e-4}})
        assert ledger.claim_data_operation("A", "ra")["payload"] == {"samples": [{"tokens": [1, 2]}]}
        ledger.complete("fb", {})
        assert ledger.claim_control_operation("A", "ra")["payload"] == {"adam_params": {"learning_rate": 2e-4}}
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
        assert ledger.claim_data_operation("A", "ra") is None
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
        assert ledger.poisoned_window_blocker("A", "ra", 4) is not None
        ledger.mark_window_consumed("opt2")
        self.complete_fb(ledger, "fb3", 3)
        assert ledger.poisoned_window_blocker("A", "ra", 4) is None

    def test_cancelled_optim_is_no_delimiter_and_cancelled_fb_poisons(self):
        ledger = OperationLedger()
        self.fail_fb(ledger, "fb1", 1)
        enqueue(ledger, "opt2", 2, "optim_step")
        ledger.cancel("opt2")
        assert ledger.poisoned_window_blocker("A", "ra", 3) is not None

        enqueue(ledger, "fb3", 3, "forward_backward")
        ledger.cancel("fb3")
        blocker = ledger.poisoned_window_blocker("A", "ra", 4)
        assert blocker is not None and "ordinal 3" in blocker

    def test_failed_forward_does_not_poison(self):
        ledger = OperationLedger()
        enqueue(ledger, "fw1", 1, "forward")
        ledger.claim_data_operation("A", "ra")
        ledger.fail("fw1", "bad forward", "user")
        assert ledger.poisoned_window_blocker("A", "ra", 2) is None


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
        ledger = OperationLedger(max_pending=2)
        enqueue(ledger, "op2", 2)
        enqueue(ledger, "op3", 3)
        assert ledger.claim_data_operation("A", "ra") is None
        enqueue(ledger, "op1", 1)
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "op1"
        with pytest.raises(OperationBackpressure):
            enqueue(ledger, "op4", 4)

    def test_ack_releases_the_payload_and_result(self):
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
        ledger.ack("op1")


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


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class TestGapTimeout:
    def gapped(self, timeout=10.0):
        clock = Clock()
        ledger = OperationLedger(gap_timeout=timeout, time_fn=clock)
        enqueue(ledger, "fb1", 1)
        ledger.claim_data_operation("A", "ra")
        ledger.complete("fb1", {})
        enqueue(ledger, "opt3", 3, "optim_step")
        ledger.sweep_gap_timeouts()
        return ledger, clock

    def test_stall_is_observable_before_expiry(self):
        ledger, clock = self.gapped()
        clock.now += 4
        [stall] = ledger.gap_stalls()
        assert stall["missing_ordinal"] == 2 and stall["blocked_operations"] == 1
        assert stall["stalled_for"] == pytest.approx(4.0)
        assert ledger.sweep_gap_timeouts() == []
        assert ledger.get("opt3")["state"] == "QUEUED"

    def test_legit_out_of_order_fill_beats_the_timeout(self):
        ledger, clock = self.gapped()
        clock.now += 9
        enqueue(ledger, "fb2", 2)
        assert ledger.sweep_gap_timeouts() == []
        assert ledger.gap_stalls() == []
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "fb2"

    def test_expiry_fails_blocked_ops_typed_and_seals_the_hole(self):
        ledger, clock = self.gapped()
        clock.now += 11
        [event] = ledger.sweep_gap_timeouts()
        assert event["missing_ordinal"] == 2
        assert event["sealed_ordinals"] == [2] and event["failed_operations"] == ["opt3"]
        view = ledger.get("opt3")
        assert view["state"] == "FAILED" and view["error_category"] == "user"
        assert "missing ordinal 2" in view["error"] and "resubmit" in view["error"]
        with pytest.raises(ValueError, match="already taken"):
            enqueue(ledger, "late2", 2, "optim_step")
        enqueue(ledger, "opt4", 4, "optim_step")
        assert ledger.claimable_control_tenants() == [("A", "ra")]
        assert ledger.claim_control_operation("A", "ra")["operation_id"] == "opt4"

    def test_expiry_seals_every_hole_below_the_arrived_tail(self):
        clock = Clock()
        ledger = OperationLedger(gap_timeout=10.0, time_fn=clock)
        enqueue(ledger, "fb1", 1)
        ledger.claim_data_operation("A", "ra")
        ledger.complete("fb1", {})
        enqueue(ledger, "fb3", 3)
        enqueue(ledger, "fb5", 5)
        ledger.sweep_gap_timeouts()
        clock.now += 11
        [event] = ledger.sweep_gap_timeouts()
        assert event["sealed_ordinals"] == [2, 4]
        assert sorted(event["failed_operations"]) == ["fb3", "fb5"]
        enqueue(ledger, "fb6", 6)
        assert ledger.claim_data_operation("A", "ra")["operation_id"] == "fb6"

    def test_sealed_hole_is_poison_neutral_and_no_delimiter(self):
        ledger, clock = self.gapped()
        clock.now += 11
        ledger.sweep_gap_timeouts()
        enqueue(ledger, "opt4", 4, "optim_step")
        assert ledger.poisoned_window_blocker("A", "ra", 4) is None

    def test_gap_failed_forward_backward_still_poisons_its_window(self):
        clock = Clock()
        ledger = OperationLedger(gap_timeout=10.0, time_fn=clock)
        enqueue(ledger, "fb1", 1)
        ledger.claim_data_operation("A", "ra")
        ledger.complete("fb1", {})
        enqueue(ledger, "fb3", 3)
        ledger.sweep_gap_timeouts()
        clock.now += 11
        [event] = ledger.sweep_gap_timeouts()
        assert event["failed_operations"] == ["fb3"]
        enqueue(ledger, "opt4", 4, "optim_step")
        blocker = ledger.poisoned_window_blocker("A", "ra", 4)
        assert blocker is not None and "ordinal 3" in blocker

    def test_disabled_timeout_reports_but_never_expires(self):
        ledger, clock = self.gapped(timeout=0)
        clock.now += 10_000
        assert ledger.sweep_gap_timeouts() == []
        [stall] = ledger.gap_stalls()
        assert stall["missing_ordinal"] == 2 and stall["stalled_for"] == pytest.approx(10_000.0)
        assert ledger.get("opt3")["state"] == "QUEUED"

    def test_a_new_hole_restarts_the_stall_clock(self):
        ledger, clock = self.gapped()
        clock.now += 9
        enqueue(ledger, "fb2", 2)
        for op_id in ("fb2", "opt3"):
            if op_id == "fb2":
                ledger.claim_data_operation("A", "ra")
            else:
                ledger.claim_control_operation("A", "ra")
            ledger.complete(op_id, {})
        enqueue(ledger, "fb5", 5)
        assert ledger.sweep_gap_timeouts() == []
        clock.now += 9
        assert ledger.sweep_gap_timeouts() == []
        clock.now += 2
        [event] = ledger.sweep_gap_timeouts()
        assert event["missing_ordinal"] == 4

    def test_fenced_queue_never_stalls(self):
        ledger, clock = self.gapped()
        ledger.fence("A", "ra")
        clock.now += 100
        assert ledger.gap_stalls() == [] and ledger.sweep_gap_timeouts() == []


class TestClaimedTimeout:
    def claimed(self, ttl=100.0):
        clock = Clock()
        ledger = OperationLedger(gap_timeout=10.0, claimed_ttl=ttl, time_fn=clock)
        enqueue(ledger, "fb1", 1)
        ledger.claim_data_operation("A", "ra")
        return ledger, clock

    def test_over_age_claimed_is_reported_with_its_age(self):
        ledger, clock = self.claimed()
        clock.now += 101
        [view] = ledger.claimed_timeouts()
        assert view["operation_id"] == "fb1" and view["state"] == "CLAIMED"
        assert view["claimed_age"] == pytest.approx(101.0)

    def test_younger_claimed_is_untouched(self):
        ledger, clock = self.claimed()
        clock.now += 99
        assert ledger.claimed_timeouts() == []
        assert ledger.get("fb1")["state"] == "CLAIMED"

    def test_control_claims_age_too(self):
        ledger, clock = self.claimed()
        ledger.complete("fb1", {})
        enqueue(ledger, "opt2", 2, "optim_step")
        ledger.claim_control_operation("A", "ra")
        clock.now += 101
        [view] = ledger.claimed_timeouts()
        assert view["operation_id"] == "opt2"

    def test_disabled_ttl_never_reports(self):
        ledger, clock = self.claimed(ttl=0)
        clock.now += 1_000_000
        assert ledger.claimed_timeouts() == []
        assert ledger.get("fb1")["state"] == "CLAIMED"

    def test_queued_operations_age_by_gap_rules_only(self):
        clock = Clock()
        ledger = OperationLedger(claimed_ttl=100.0, time_fn=clock)
        enqueue(ledger, "fb1", 1)
        clock.now += 1000
        assert ledger.claimed_timeouts() == []
        assert ledger.get("fb1")["state"] == "QUEUED"

    def test_a_claimed_head_is_not_a_gap_stall(self):
        ledger, clock = self.claimed()
        clock.now += 1000
        assert ledger.gap_stalls() == [] and ledger.sweep_gap_timeouts() == []


class TestTenantEviction:
    def test_drop_tenant_purges_the_dead_registration_only(self):
        ledger = OperationLedger()
        enqueue(ledger, "old1", 1, payload={"samples": ["x" * 64]}, name="A", reg="ra")
        ledger.complete("old1", {"kept": True})
        ledger.fence("A", "ra")
        assert ledger.by_id["old1"].payload == {}
        assert ledger.get("old1")["result"] == {"kept": True}
        enqueue(ledger, "young1", 1, name="B", reg="rb")
        ledger.complete("young1", {})
        ledger.fence("B", "rb")
        ledger.drop_tenant("A", "ra")
        assert ledger.get("old1") is None and ("A", "ra") not in ledger.queues
        assert not any(op.tenant == ("A", "ra") for op in ledger.by_id.values())
        assert ledger.get("young1")["state"] == "SUCCEEDED"
        ledger.drop_tenant("A", "ra")
