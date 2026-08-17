"""FixedSlotResidency + claim-and-bind + batch lease
(codex-rollout-fullparameter-design-0810 §5.3/§3.6/§8.2).

The port only snapshots/validates what fixed residency already established:
binding_for is the claim gate (exact READY + slot), acquire is the
dispatch gates (exact ownership; RETIRING allowed for in-flight work),
release_batch is a no-op. Nothing here binds, evicts, or moves state, and
active never exceeds slots."""

import copy
from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio

import pytest

from miles.ray.tinker_backend.backend import TinkerBackend
from miles.ray.tinker_backend.config import AdapterRunConfig
from miles.ray.tinker_backend.registry import AdapterRegistry, AdapterState
from miles.ray.tinker_backend.residency import (
    FixedSlotResidency,
    ResidentBinding,
    lease_from_metadata,
    lease_to_metadata,
)


def make_registry(n=1) -> AdapterRegistry:
    return AdapterRegistry(n)


def register_ready(registry, name) -> tuple[str, str]:
    registry.register(name, AdapterRunConfig())
    registry.mark_ready([name])
    return (name, registry.find(name).registration_id)


def make_backend(max_adapters=1) -> TinkerBackend:
    args = SimpleNamespace(
        multi_lora_n_adapters=max_adapters,
        save="/tmp/tinker-test-save",
        lora_rank=32,
        lora_alpha=64,
        hf_checkpoint="Qwen/Qwen3-0.6B",
    )
    return TinkerBackend(args, "http://unused")


def fb_payload():
    return {
        "samples": [{"tokens": [1, 2, 3, 4], "response_length": 2, "loss_mask": [1, 1], "loss_weights": [1.0, 1.0]}],
        "loss": {"loss_fn": "cross_entropy"},
    }


class TestBindingFor:
    def test_exact_ready_with_slot_only(self):
        registry = make_registry(2)
        key = register_ready(registry, "A")
        residency = FixedSlotResidency(registry)
        assert residency.binding_for(key) == ResidentBinding(registration_key=key, training_slot=0)

    def test_every_other_state_is_rejected_without_mutation(self):
        registry = make_registry(1)
        residency = FixedSlotResidency(registry)

        # PENDING (bound but not loaded yet)
        registry.register("A", AdapterRunConfig())
        key_a = ("A", registry.find("A").registration_id)
        assert residency.binding_for(key_a) is None

        # unbound PENDING (pool full)
        registry.register("B", AdapterRunConfig())
        key_b = ("B", registry.find("B").registration_id)
        assert residency.binding_for(key_b) is None

        # wrong registration id
        registry.mark_ready(["A"])
        assert residency.binding_for(("A", "not-the-registration")) is None

        # RETIRING: binding_for is the CLAIM gate — no new claims
        registry.deregister("A")
        assert residency.binding_for(key_a) is None

        # CLEANUP
        registry.retire_adapters()
        assert residency.binding_for(key_a) is None

        # the lookups mutated nothing: A still owns slot 0, B still queued
        assert registry.records["A"].slot == 0
        assert registry.records["B"].slot is None
        before = copy.deepcopy(registry.snapshot())
        residency.binding_for(key_a)
        assert registry.snapshot() == before


class TestClaimAndBind:
    def test_data_claim_carries_the_binding(self):
        backend = make_backend()
        asyncio.run(backend.register("A", AdapterRunConfig()))
        backend.registry.mark_ready(["A"])
        rid = backend.registry.find("A").registration_id
        backend.enqueue_operation("A", "fb1", 1, "forward_backward", fb_payload())
        claim = backend.claim_data_operation("A", rid)
        assert claim["operation_id"] == "fb1"
        assert claim["binding"] == ResidentBinding(registration_key=("A", rid), training_slot=0)

    def test_unbound_pending_is_never_claimed_and_head_stays_queued(self):
        """S_train=1 capacity fence: B queues unbound behind A; B's operations
        buffer but are unclaimable (all-or-nothing claim-and-bind: no binding,
        no CLAIMED). Only A's FULL cleanup binds and opens B."""
        backend = make_backend(max_adapters=1)
        asyncio.run(backend.register("A", AdapterRunConfig()))
        backend.registry.mark_ready(["A"])
        asyncio.run(backend.register("B", AdapterRunConfig()))
        rid_b = backend.registry.find("B").registration_id
        backend.enqueue_operation("B", "b-fb1", 1, "forward_backward", fb_payload())

        assert backend.claim_data_operation("B", rid_b) is None
        assert backend.operations.get("b-fb1")["state"] == "QUEUED"  # not CLAIMED, not failed

        # A's full retirement path frees the slot; bootstrap binds B.
        backend.registry.deregister("A")
        backend.registry.retire_adapters()
        backend.registry.free_slot("A")
        assert backend.registry.bootstrap_pending() == ["B"]
        backend.registry.mark_ready(["B"])
        claim = backend.claim_data_operation("B", rid_b)
        assert claim["operation_id"] == "b-fb1"
        assert claim["binding"].training_slot == 0

    def test_control_claims_still_require_ready_and_slot(self):
        backend = make_backend(max_adapters=1)
        asyncio.run(backend.register("A", AdapterRunConfig()))
        backend.registry.mark_ready(["A"])
        asyncio.run(backend.register("B", AdapterRunConfig()))  # unbound
        backend.enqueue_operation("B", "b-opt1", 1, "optim_step")
        assert backend.claim_ready_control_operations() == {"operations": [], "lease": None}
        assert backend.operations.get("b-opt1")["state"] == "QUEUED"


class TestBatchLease:
    def test_acquire_release_roundtrip(self):
        registry = make_registry(2)
        key_a = register_ready(registry, "A")
        key_b = register_ready(registry, "B")
        residency = FixedSlotResidency(registry)
        lease = residency.acquire_batch(
            (
                ("op-A", residency.binding_for(key_a)),
                ("op-B", residency.binding_for(key_b)),
            )
        )
        assert lease.binding_of("op-A").training_slot == 0
        assert lease.binding_of("op-B").training_slot == 1
        assert lease.binding_of("op-unknown") is None
        before = copy.deepcopy(registry.snapshot())
        residency.release_batch(lease)  # no-op lifecycle hook
        assert registry.snapshot() == before
        # plain-data roundtrip for the object-store crossing
        assert lease_from_metadata(lease_to_metadata(lease)) == lease

    def test_retiring_after_claim_keeps_the_receipt_valid(self):
        """Race characterization (§8.2): claimed at READY, deregistered before
        acquire — the exact registration still owns and loads the slot, so
        acquire must succeed and the in-flight operation completes; only
        cleanup/reassign invalidates (acquire refuses). Trainer-side lease
        validation is validate_batch_lease — the sole validator."""
        registry = make_registry(1)
        key = register_ready(registry, "A")
        residency = FixedSlotResidency(registry)
        binding = residency.binding_for(key)

        registry.deregister("A")  # READY -> RETIRING mid-flight
        lease = residency.acquire_batch((("op-A", binding),))
        assert lease.binding_of("op-A") is binding

        # Full cleanup reassigns the slot: the receipt dies with the tenancy.
        registry.retire_adapters()
        registry.free_slot("A")
        with pytest.raises(ValueError, match="no longer owns trainer slot"):
            residency.acquire_batch((("op-A", binding),))

    def test_wrong_slot_or_foreign_registration_is_refused(self):
        registry = make_registry(2)
        key = register_ready(registry, "A")
        residency = FixedSlotResidency(registry)
        with pytest.raises(ValueError, match="no longer owns"):
            residency.acquire_batch((("op-A", ResidentBinding(registration_key=key, training_slot=1)),))
        with pytest.raises(ValueError, match="no longer owns"):
            residency.acquire_batch(
                (("op-A", ResidentBinding(registration_key=("A", "stale-registration"), training_slot=0)),)
            )


class TestTrainerLocalValidation:
    def test_lease_must_match_locally_loaded_adapters(self):
        from miles.backends.megatron_utils.tinker_backend.trainer import validate_batch_lease

        loaded = {"A": SimpleNamespace(registration_id="r-A", slot=0)}
        good = {"batch_execution_lease": {"dispatch_id": "d", "bindings_by_operation": [["op-A", ["A", "r-A", 0]]]}}
        validate_batch_lease(good, loaded)  # exact match passes

        for name, rid, slot in [("A", "r-A", 1), ("A", "r-OLD", 0), ("Z", "r-Z", 0)]:
            bad = {
                "batch_execution_lease": {"dispatch_id": "d", "bindings_by_operation": [["op-A", [name, rid, slot]]]}
            }
            with pytest.raises(RuntimeError, match="does not match"):
                validate_batch_lease(bad, loaded)

        with pytest.raises(RuntimeError, match="no execution lease"):
            validate_batch_lease({}, loaded)

    def test_retiring_lifecycle_does_not_invalidate_the_local_receipt(self):
        """The trainer check is ownership-based (name, registration, slot vs
        loaded_adapters) — a claim-then-deregister still validates because the
        adapter stays loaded until the next reconcile; AdapterState never
        enters the local check."""
        from miles.backends.megatron_utils.tinker_backend.trainer import validate_batch_lease

        loaded = {"A": SimpleNamespace(registration_id="r-A", slot=0)}
        lease = {"batch_execution_lease": {"dispatch_id": "d", "bindings_by_operation": [["op-A", ["A", "r-A", 0]]]}}
        validate_batch_lease(lease, loaded)


def test_registry_lifecycle_untouched_by_residency_reads():
    """Fixed residency invariant (§5.1): N_active == READY == fixed-resident
    <= slots; the port adds lookups, never new lifecycle transitions."""
    registry = make_registry(1)
    residency = FixedSlotResidency(registry)
    register_ready(registry, "A")
    registry.register("B", AdapterRunConfig())
    assert registry.records["B"].slot is None
    assert registry.records["B"].state is AdapterState.PENDING
    for _ in range(3):
        residency.binding_for(("B", registry.records["B"].registration_id))
    assert registry.records["B"].slot is None  # still queued; no LRU, no swap
