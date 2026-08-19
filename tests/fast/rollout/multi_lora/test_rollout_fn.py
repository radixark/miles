"""Tinker operation-to-batch adapter: one claimed operation becomes one
stamped batch, bad payloads fail their own operation, and the selection loop
enforces the homogeneous kind lock with persistent round-robin fairness — all
driven through FAKE OperationQueuePort/BatchResidencyPort transports (no Ray
import, per codex-rollout-fullparameter-design-0810 §8.2)."""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio

import pytest

from miles.ray.multi_lora.config import AdapterRun, AdapterRunConfig
from miles.ray.multi_lora.residency import ResidentBinding
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainOutput
from miles.rollout.multi_lora.rollout_fn import AdapterRolloutRuntime, ClaimedOperationBatch, MultiLoraOperationBatchFn
from miles.utils.operation_contract import BatchExecutionLease, EmptyBatchTimeoutError


def make_run(name="X", reg="rx", slot=3, version=2) -> AdapterRun:
    config = AdapterRunConfig(rank=8, alpha=16, metadata={"team": "t1"})
    return AdapterRun(name=name, config=config, slot=slot, version=version, registration_id=reg)


def claim_batch(run: AdapterRun, operations) -> ClaimedOperationBatch:
    """Drive the adapter's claim path for one registration runtime."""
    fn = MultiLoraOperationBatchFn(
        RolloutFnConstructorInput(args=SimpleNamespace(), data_source=None),
        operations=operations,
        residency=FakeResidency(),
    )
    return asyncio.run(fn._claim_batch(AdapterRolloutRuntime(run)))


def sample_payload(n=2) -> dict:
    return {
        "batch_id": "batch-7",  # client-side bookkeeping key the server ignores
        "samples": [
            {"prompt": "p", "tokens": [1, 2, 3, 4], "response_length": 2, "loss_mask": [1, 1]} for _ in range(n)
        ],
        "loss": {"loss_fn": "cross_entropy"},
    }


class FakeOperationQueue:
    """Scripted OperationQueuePort: claims pop in order, failures record."""

    def __init__(self, claims=(), ready=None):
        self._claims = list(claims)
        self._ready = ready or {}
        self.failed: list[tuple] = []

    async def ready_streams(self) -> dict:
        return self._ready

    async def claim_data(self, key):
        return self._claims.pop(0) if self._claims else None

    async def fail(self, operation_id, error, category):
        self.failed.append((operation_id, error, category))


class FakeResidency:
    """Scripted BatchResidencyPort: mints deterministic leases."""

    def __init__(self):
        self.leases: list[tuple] = []
        self.releases: list[BatchExecutionLease] = []

    async def acquire_batch(self, bindings_by_operation):
        self.leases.append(tuple(bindings_by_operation))
        return BatchExecutionLease(dispatch_id="lease-1", bindings_by_operation=tuple(bindings_by_operation))

    async def release_batch(self, lease):
        self.releases.append(lease)


class FakeBatchAbort:
    """Recording BatchAbortPort used by lifecycle handoff tests."""

    def __init__(self):
        self.aborts: list[tuple] = []

    async def abort_batch(self, operation_ids, error, lease_metadata):
        self.aborts.append((list(operation_ids), error, lease_metadata))


@pytest.fixture()
def fast_poll(monkeypatch):
    import miles.rollout.multi_lora.rollout_fn as rollout_module

    monkeypatch.setattr(rollout_module, "_CLAIM_POLL_S", 0.01)


def op(op_id="op1", kind="forward_backward", payload=None, slot=3):
    # A claim always carries its fixed binding (claim-and-bind).
    return dict(
        operation_id=op_id,
        name="X",
        registration_id="rx",
        kind=kind,
        payload=sample_payload() if payload is None else payload,
        state="CLAIMED",
        binding=ResidentBinding(registration_key=("X", "rx"), training_slot=slot),
    )


class TestClaimBatch:
    def test_one_operation_becomes_one_stamped_batch(self):
        output = claim_batch(make_run(), FakeOperationQueue([op()]))

        assert len(output.samples) == 2 and all(len(group) == 1 for group in output.samples)
        stamped = output.samples[0][0]
        assert (stamped.adapter.name, stamped.adapter.registration_id) == ("X", "rx")
        assert stamped.adapter.serving_version == 2 and stamped.adapter.slot == 3
        assert stamped.metadata["team"] == "t1"  # run metadata merged in
        assert stamped.status == stamped.Status.COMPLETED
        assert [group[0].index for group in output.samples] == [0, 1]  # result-plane row identity
        assert isinstance(output, ClaimedOperationBatch)
        assert output.operation_id == "op1"
        assert output.kind == "forward_backward"
        assert output.loss_spec == {"loss_fn": "cross_entropy"}
        assert output.binding == ResidentBinding(registration_key=("X", "rx"), training_slot=3)

    def test_client_supplied_row_index_is_overwritten(self):
        # index is server-owned: a client -1 would alias the DP-padding
        # sentinel (row silently dropped from the result plane) and duplicates
        # would collide in the (lane, row) logprob collector.
        payload = sample_payload()
        payload["samples"][0]["index"] = -1
        payload["samples"][1]["index"] = 0
        queue = FakeOperationQueue([op(payload=payload)])
        output = claim_batch(make_run(), queue)
        assert [group[0].index for group in output.samples] == [0, 1]

    def test_child_waits_for_a_claim(self, fast_poll):
        queue = FakeOperationQueue([None, None, op()])
        output = claim_batch(make_run(), queue)
        assert output.operation_id == "op1"

    def test_bad_payload_fails_its_operation_and_the_child_continues(self):
        queue = FakeOperationQueue([op("bad", payload={"samples": []}), op("good")])
        output = claim_batch(make_run(), queue)

        assert output.operation_id == "good"
        [(failed_id, error, category)] = queue.failed
        assert failed_id == "bad" and category == "user" and "no samples" in error

    def test_forward_operations_build_batches_too(self):
        payload = {"samples": [{"prompt": "p", "tokens": [1, 2], "response_length": 1, "loss_mask": [1]}]}
        queue = FakeOperationQueue([op("fwd", kind="forward", payload=payload)])
        output = claim_batch(make_run(), queue)
        assert output.kind == "forward"
        assert output.loss_spec is None
        assert queue.failed == []


def ready_runtime(fn: MultiLoraOperationBatchFn, name: str, slot: int, kind: str) -> AdapterRolloutRuntime:
    # The runtime's stamped slot (9) is deliberately stale: the claim's
    # binding, not the long-lived AdapterRun view, is the dispatch truth.
    run = make_run(name=name, reg=f"r-{name}", slot=9)
    runtime = AdapterRolloutRuntime(run)
    runtime.state = AdapterRolloutRuntime.READY
    runtime.ready_output = ClaimedOperationBatch(
        operation_id=f"op-{name}",
        kind=kind,
        loss_spec=None,
        binding=ResidentBinding(registration_key=(name, f"r-{name}"), training_slot=slot),
        samples=[[SimpleNamespace(adapter=None, metadata={})]],
    )
    fn.runtimes[(run.name, run.registration_id)] = runtime
    fn._sync_rotation()
    return runtime


def merge(fn: MultiLoraOperationBatchFn, selected) -> RolloutFnTrainOutput:
    return asyncio.run(fn._merge(selected))


def make_fn(soft_target=100) -> MultiLoraOperationBatchFn:
    args = SimpleNamespace(
        rollout_batch_size=soft_target,
        n_samples_per_prompt=1,
        tinker_max_coalesce_wait_s=0.05,
        tinker_max_empty_wait_s=0.05,
    )
    return MultiLoraOperationBatchFn(
        RolloutFnConstructorInput(args=args, data_source=None),
        operations=FakeOperationQueue(),
        residency=FakeResidency(),
        abort=FakeBatchAbort(),
    )


class TestSelectionKindLock:
    def test_first_ready_locks_the_kind(self):
        fn = make_fn()
        ready_runtime(fn, "A", 0, "forward_backward")
        other = ready_runtime(fn, "B", 1, "forward")
        ready_runtime(fn, "C", 2, "forward_backward")

        selected = asyncio.run(fn._select())
        assert sorted(r.run.name for r in selected) == ["A", "C"]
        # The other-kind batch is untouched and stays READY for the next call.
        assert other.state == AdapterRolloutRuntime.READY

    def test_all_forward_selection_is_fine(self):
        fn = make_fn()
        ready_runtime(fn, "A", 0, "forward")
        ready_runtime(fn, "B", 1, "forward")
        selected = asyncio.run(fn._select())
        assert {r.ready_kind for r in selected} == {"forward"}

    def test_soft_target_stops_collection_but_never_trims(self):
        fn = make_fn(soft_target=1)
        ready_runtime(fn, "A", 0, "forward_backward")
        ready_runtime(fn, "B", 1, "forward_backward")
        selected = asyncio.run(fn._select())
        assert len(selected) == 1  # whole batches; B waits for the next call

    def test_empty_selection_times_out(self):
        fn = make_fn()
        with pytest.raises(EmptyBatchTimeoutError):
            asyncio.run(fn._select())

    @pytest.mark.asyncio
    async def test_cancelled_coalescing_restores_local_selection_to_ready(self):
        fn = make_fn()
        fn.args.tinker_max_coalesce_wait_s = 60
        selected_runtime = ready_runtime(fn, "A", 0, "forward_backward")
        other_kind = ready_runtime(fn, "B", 1, "forward")

        task = asyncio.create_task(fn._select())
        while selected_runtime.state != AdapterRolloutRuntime.SELECTED:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert selected_runtime.state == AdapterRolloutRuntime.READY
        assert selected_runtime.ready_output is not None
        assert other_kind.state == AdapterRolloutRuntime.READY

    def test_merge_ships_the_converted_plan_and_pad_policy(self):
        """Correlation is batch-local (§3.3): the selected operation gets lane
        0, the loss/result maps key by lane, and the exact registration rides
        along for the commit. The claim's binding is the single binding truth
        — it flows into the batch lease (§5.3) and the routing helper; the
        runtime's stale stamped slot (9) appears nowhere."""
        fn = make_fn()
        first = ready_runtime(fn, "A", 0, "forward_backward")
        selected = asyncio.run(fn._select())
        output = merge(fn, selected)
        assert output.conversion_metadata == {
            "batch_kind": "tinker",
            "tinker_operation_lanes": [0],
            "tinker_loss_by_lane": {0: {}},
            "operation_by_lane": {0: "op-A"},
            "registration_by_lane": {0: ("A", "r-A")},
            "batch_execution_lease": {
                "dispatch_id": "lease-1",
                "bindings_by_operation": [["op-A", ["A", "r-A", 0]]],
            },
        }
        assert output.postprocess.pad_to_dp is True
        assert output.handoff.receipt == {
            "operation_ids": ["op-A"],
            "lease": output.conversion_metadata["batch_execution_lease"],
        }
        assert first.state == AdapterRolloutRuntime.IDLE and first.ready_output is None

    def test_failed_lease_acquisition_keeps_claimed_output_retryable(self):
        """External review P1: acquisition is fallible (fencing races), and a
        failure must not orphan the only in-memory copy of an already-CLAIMED
        output — the selected runtimes return to READY with their outputs
        intact, and the next selection retries them."""

        class RefusingOnceResidency(FakeResidency):
            def __init__(self):
                super().__init__()
                self.refusals_left = 1

            async def acquire_batch(self, bindings_by_operation):
                if self.refusals_left:
                    self.refusals_left -= 1
                    raise ValueError("stale binding")
                return await super().acquire_batch(bindings_by_operation)

        fn = make_fn()
        fn.residency = RefusingOnceResidency()
        runtime = ready_runtime(fn, "A", 0, "forward_backward")
        selected = asyncio.run(fn._select())

        with pytest.raises(ValueError, match="stale binding"):
            merge(fn, selected)
        assert runtime.state == AdapterRolloutRuntime.READY
        assert runtime.ready_output is not None

        # Retry-once: the SAME claimed output dispatches on the next cycle.
        selected = asyncio.run(fn._select())
        output = merge(fn, selected)
        assert output.conversion_metadata["operation_by_lane"] == {0: "op-A"}
        assert runtime.state == AdapterRolloutRuntime.IDLE and runtime.ready_output is None

    def test_post_lease_handoff_build_failure_aborts_the_claimed_batch(self, monkeypatch):
        import miles.rollout.multi_lora.rollout_fn as rollout_module

        fn = make_fn()
        runtime = ready_runtime(fn, "A", 0, "forward_backward")
        selected = asyncio.run(fn._select())

        def fail_output_build(**_kwargs):
            raise RuntimeError("handoff construction failed")

        monkeypatch.setattr(rollout_module, "RolloutFnTrainOutput", fail_output_build)
        with pytest.raises(RuntimeError, match="handoff construction failed"):
            merge(fn, selected)

        [(operation_ids, error, lease_metadata)] = fn.abort.aborts
        assert operation_ids == ["op-A"]
        assert "handoff construction failed" in error
        assert lease_metadata["dispatch_id"] == "lease-1"
        assert runtime.state == AdapterRolloutRuntime.IDLE and runtime.ready_output is None

    def test_lease_encoding_failure_releases_the_exact_typed_lease(self, monkeypatch):
        import miles.rollout.multi_lora.rollout_fn as rollout_module

        fn = make_fn()
        runtime = ready_runtime(fn, "A", 0, "forward_backward")
        selected = asyncio.run(fn._select())

        def fail_encoding(_lease):
            raise RuntimeError("lease encoding failed")

        monkeypatch.setattr(rollout_module, "lease_to_metadata", fail_encoding)
        with pytest.raises(RuntimeError, match="lease encoding failed"):
            merge(fn, selected)

        [(operation_ids, error, lease_metadata)] = fn.abort.aborts
        assert operation_ids == ["op-A"]
        assert "lease encoding failed" in error
        assert lease_metadata is None
        [released] = fn.residency.releases
        assert released.dispatch_id == "lease-1"
        assert runtime.state == AdapterRolloutRuntime.IDLE and runtime.ready_output is None

    def test_merge_of_a_forward_selection_marks_forward_only(self):
        """Forward kind: the same composition with ``tinker_forward_only``
        set — the flag that keeps forward operations gradient-free must
        survive the lane re-keying."""
        fn = make_fn()
        ready_runtime(fn, "A", 0, "forward")
        ready_runtime(fn, "B", 1, "forward")
        selected = asyncio.run(fn._select())
        output = merge(fn, selected)
        assert output.conversion_metadata["tinker_forward_only"] is True
        assert output.conversion_metadata["operation_by_lane"] == {0: "op-A", 1: "op-B"}
        assert output.conversion_metadata["tinker_operation_lanes"] == [0, 1]
        assert output.postprocess.pad_to_dp is True

    def test_lanes_are_selection_local_and_independent_of_slots(self):
        """Two operations on HIGH slots (7, 2) still get lanes 0 and 1 in
        selection order: identity never rides the physical slot, so a future
        parameterization (or slot reuse across operations) cannot collide in
        the collector/result plane."""
        fn = make_fn()
        ready_runtime(fn, "A", 7, "forward_backward")
        ready_runtime(fn, "B", 2, "forward_backward")
        selected = asyncio.run(fn._select())
        output = merge(fn, selected)
        assert output.conversion_metadata["tinker_operation_lanes"] == [0, 1]
        assert output.conversion_metadata["registration_by_lane"] == {0: ("A", "r-A"), 1: ("B", "r-B")}
        lease = output.conversion_metadata["batch_execution_lease"]
        assert lease["bindings_by_operation"] == [["op-A", ["A", "r-A", 7]], ["op-B", ["B", "r-B", 2]]]


class TestDriverHandoff:
    def test_merge_mints_exact_operation_ids_and_the_conversion_lease(self):
        import ray.cloudpickle

        fn = make_fn()
        ready_runtime(fn, "A", 7, "forward_backward")
        ready_runtime(fn, "B", 2, "forward_backward")
        output = merge(fn, asyncio.run(fn._select()))

        assert output.handoff.receipt["operation_ids"] == ["op-A", "op-B"]
        assert output.handoff.receipt["lease"] is output.conversion_metadata["batch_execution_lease"]
        assert ray.cloudpickle.loads(ray.cloudpickle.dumps(output.handoff.receipt)) == output.handoff.receipt

    def test_abort_handoff_finalizes_the_exact_batch(self):
        fn = make_fn()
        ready_runtime(fn, "A", 0, "forward_backward")
        output = merge(fn, asyncio.run(fn._select()))

        asyncio.run(fn.abort_handoff(output.handoff, OSError("object-store placement failed")))

        [(operation_ids, error, lease_metadata)] = fn.abort.aborts
        assert operation_ids == ["op-A"]
        assert lease_metadata is output.handoff.receipt["lease"]
        assert "placement failed" in error and "poisoned" in error and "resubmit" in error
