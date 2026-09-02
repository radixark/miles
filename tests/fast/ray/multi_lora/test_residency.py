import asyncio
import copy
from types import SimpleNamespace

import pytest

from miles.ray.multi_lora.backend import MultiLoraOperationBackend
from miles.ray.multi_lora.config import AdapterRunConfig
from miles.ray.multi_lora.registry import AdapterRegistry
from miles.ray.multi_lora.residency import FixedSlotResidency, ResidentBinding, lease_from_metadata, lease_to_metadata


def make_registry(n=1) -> AdapterRegistry:
    return AdapterRegistry(n)


def register_ready(registry, name) -> tuple[str, str]:
    registry.register(name, AdapterRunConfig())
    registry.mark_ready([name])
    return (name, registry.find(name).registration_id)


def make_backend(max_adapters=1) -> MultiLoraOperationBackend:
    args = SimpleNamespace(
        multi_lora_n_adapters=max_adapters,
        save="/tmp/tinker-test-save",
        lora_rank=32,
        lora_alpha=64,
        hf_checkpoint="Qwen/Qwen3-0.6B",
    )
    return MultiLoraOperationBackend(args, "http://unused")


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

        registry.register("A", AdapterRunConfig())
        key_a = ("A", registry.find("A").registration_id)
        assert residency.binding_for(key_a) is None

        registry.register("B", AdapterRunConfig())
        key_b = ("B", registry.find("B").registration_id)
        assert residency.binding_for(key_b) is None

        registry.mark_ready(["A"])
        assert residency.binding_for(("A", "not-the-registration")) is None

        registry.deregister("A")
        assert residency.binding_for(key_a) is None

        registry.retire_adapters()
        assert residency.binding_for(key_a) is None

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
        backend = make_backend(max_adapters=1)
        asyncio.run(backend.register("A", AdapterRunConfig()))
        backend.registry.mark_ready(["A"])
        asyncio.run(backend.register("B", AdapterRunConfig()))
        rid_b = backend.registry.find("B").registration_id
        backend.enqueue_operation("B", "b-fb1", 1, "forward_backward", fb_payload())

        assert backend.claim_data_operation("B", rid_b) is None
        assert backend.operations.get("b-fb1")["state"] == "QUEUED"

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
        asyncio.run(backend.register("B", AdapterRunConfig()))
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
        residency.release_batch(lease)
        assert registry.snapshot() == before
        assert lease_from_metadata(lease_to_metadata(lease)) == lease

    def test_retiring_after_claim_keeps_the_receipt_valid(self):
        registry = make_registry(1)
        key = register_ready(registry, "A")
        residency = FixedSlotResidency(registry)
        binding = residency.binding_for(key)

        registry.deregister("A")
        lease = residency.acquire_batch((("op-A", binding),))
        assert lease.binding_of("op-A") is binding

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
        from miles.backends.megatron_utils.api_backends.multi_lora.trainer import validate_batch_lease

        loaded = {"A": SimpleNamespace(registration_id="r-A", slot=0)}
        good = {"batch_execution_lease": {"dispatch_id": "d", "bindings_by_operation": [["op-A", ["A", "r-A", 0]]]}}
        validate_batch_lease(good, loaded)

        for name, rid, slot in [("A", "r-A", 1), ("A", "r-OLD", 0), ("Z", "r-Z", 0)]:
            bad = {
                "batch_execution_lease": {"dispatch_id": "d", "bindings_by_operation": [["op-A", [name, rid, slot]]]}
            }
            with pytest.raises(RuntimeError, match="does not match"):
                validate_batch_lease(bad, loaded)

        with pytest.raises(RuntimeError, match="no execution lease"):
            validate_batch_lease({}, loaded)
