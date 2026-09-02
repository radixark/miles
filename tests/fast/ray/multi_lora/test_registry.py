import pytest

from miles.ray.multi_lora.config import AdapterRunConfig
from miles.ray.multi_lora.registry import AdapterRegistry, AdapterState
from miles.ray.multi_lora.slot_pool import SlotPool


class TestSlotPool:
    def test_binds_lowest_free_and_queues_when_full(self):
        pool = SlotPool(2)
        assert pool.bind_immediately(("a", "r1")) == 0
        assert pool.bind_immediately(("b", "r1")) == 1
        assert pool.bind_immediately(("c", "r1")) is None
        assert pool.release(("a", "r1")) == 0
        assert pool.bind_immediately(("c", "r1")) == 0

    def test_release_clears_pins(self):
        pool = SlotPool(1)
        pool.bind_immediately(("a", "r1"))
        pool.pin(("a", "r1"), "dirty-grads")
        assert pool.is_pinned(("a", "r1"), "dirty-grads")
        pool.release(("a", "r1"))
        pool.bind_immediately(("b", "r1"))
        assert not pool.is_pinned(("b", "r1"), "dirty-grads")

    def test_occupied_ids(self):
        pool = SlotPool(3)
        pool.bind_immediately(("a", "r1"))
        pool.bind_immediately(("b", "r1"))
        assert pool.occupied_slot_ids() == [0, 1]
        assert pool.free_slot_ids() == {2}


def config(**overrides) -> AdapterRunConfig:
    return AdapterRunConfig(**overrides)


def register_ready(registry, name):
    registry.register(name, config())
    registry.mark_ready([name])
    return registry.find(name)


class TestLifecycle:
    def test_ready_comes_from_trainer_load_not_from_a_publish(self):
        registry = AdapterRegistry(2)
        registry.register("A", config())
        assert registry.find("A").state is AdapterState.PENDING
        registry.record_weight_update(["A"])
        assert registry.find("A").state is AdapterState.PENDING
        assert registry.find("A").serving_version == 1
        registry.mark_ready(["A"])
        assert registry.find("A").state is AdapterState.READY

    def test_unbound_pending_cannot_become_ready(self):
        registry = AdapterRegistry(1)
        registry.register("A", config())
        registry.register("B", config())
        assert registry.find("B").slot is None
        registry.mark_ready(["B"])
        assert registry.find("B").state is AdapterState.PENDING

    def test_queue_drains_at_retirement(self):
        registry = AdapterRegistry(1)
        registry.register("A", config())
        registry.register("B", config())
        registry.deregister("A")
        assert registry.retire_adapters() == ["A"]
        assert registry.free_slot("A") == 0
        assert registry.bootstrap_pending() == ["B"]
        assert registry.find("B").slot == 0

    def test_queue_drains_in_arrival_order_not_name_order(self):
        registry = AdapterRegistry(1)
        registry.register("A", config())
        registry.register("Z", config())
        registry.register("B", config())
        registry.deregister("A")
        registry.retire_adapters()
        registry.free_slot("A")
        assert registry.bootstrap_pending() == ["Z"]
        registry.deregister("Z")
        registry.retire_adapters()
        registry.free_slot("Z")
        assert registry.bootstrap_pending() == ["B"]

    def test_duplicate_and_invalid_names_rejected(self):
        registry = AdapterRegistry(2)
        registry.register("A", config())
        with pytest.raises(ValueError, match="already registered"):
            registry.register("A", config())
        with pytest.raises(ValueError, match="invalid"):
            registry.register("bad name", config())

    def test_save_dir_conflict_rejected(self):
        registry = AdapterRegistry(2)
        registry.register("A", config(save="/tmp/x"))
        with pytest.raises(ValueError, match="already used"):
            registry.register("B", config(save="/tmp/x"))


class TestClocksAndPins:
    def test_committed_step_mirrors_clock_and_releases_the_pin(self):
        registry = AdapterRegistry(1)
        record = register_ready(registry, "A")
        registry.mark_accumulated(["A"])
        assert registry.is_dirty("A")
        registry.on_step_committed("A", record.registration_id, 1)
        assert not registry.is_dirty("A")
        assert record.step == 1

    def test_hook_ignores_a_stale_registration(self):
        registry = AdapterRegistry(1)
        record = register_ready(registry, "A")
        registry.on_step_committed("A", "not-the-registration", 7)
        assert record.step == 0

    def test_veto_path_clears_dirty_without_advancing(self):
        registry = AdapterRegistry(1)
        record = register_ready(registry, "A")
        registry.mark_accumulated(["A"])
        registry.clear_dirty("A")
        assert not registry.is_dirty("A")
        assert record.step == 0

    def test_num_step_bound_deregisters(self):
        registry = AdapterRegistry(1)
        registry.register("A", config(num_step=2))
        registry.mark_ready(["A"])
        rid = registry.find("A").registration_id
        registry.on_step_committed("A", rid, 1)
        assert registry.find("A").state is AdapterState.READY
        registry.on_step_committed("A", rid, 2)
        assert registry.records["A"].state is AdapterState.RETIRING

    def test_set_step_repositions_baseline(self):
        registry = AdapterRegistry(1)
        registry.register("A", config(num_step=2))
        registry.mark_ready(["A"])
        rid = registry.find("A").registration_id
        registry.set_step("A", 10)
        registry.on_step_committed("A", rid, 11)
        assert registry.records["A"].state is AdapterState.READY
        registry.on_step_committed("A", rid, 12)
        assert registry.records["A"].state is AdapterState.RETIRING


class TestViews:
    def test_snapshot_vocabulary(self):
        registry = AdapterRegistry(2)
        register_ready(registry, "A")
        registry.register("B", config())
        snap = registry.snapshot()
        assert list(snap["ready"]) == ["A"] and list(snap["pending"]) == ["B"]
        assert snap["ready"]["A"].registration_id
        assert registry.ready_adapters()["A"].slot == 0
