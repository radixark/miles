"""Fast tests for AdapterRegistry + MultiLoRABackend validation
(no Ray, no HTTP I/O, no SGLang, no torch)."""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import pytest

from miles.utils.adapter_config import AdapterRunConfig
from miles.utils.multi_lora import AdapterRegistry, AdapterState, MultiLoRABackend, make_rid, parse_adapter


def make_args(max_adapters: int = 4, save: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(multi_lora_n_adapters=max_adapters, save=save)


def make_backend(max_adapters: int = 4, save: str | None = None) -> MultiLoRABackend:
    return MultiLoRABackend(make_args(max_adapters, save), "http://unused")


def make_config(save: str | None = None) -> AdapterRunConfig:
    return AdapterRunConfig(
        rank=8, alpha=16, data="/d", save=save, input_key="text", label_key="label", rm_type="math"
    )


def register_and_promote(registry: AdapterRegistry, name: str, config=None) -> None:
    registry.register(name, config)
    registry.record_weight_update([name])


def test_rid_roundtrip_preserves_names_with_underscores():
    for name in ["a", "adapter_a", "weird__name", "x_y_z"]:
        assert parse_adapter(make_rid(name)) == name


def test_register_starts_pending_and_push_promotes():
    registry = AdapterRegistry(max_adapters=4)
    result = registry.register("A", config={"rm_type": "x"})
    assert result == {"name": "A", "slot": 0}
    assert registry.active_adapters() == {}  # pending: not sampleable

    registry.record_weight_update(["A"])
    assert registry.active_adapters()["A"].slot == 0
    view = registry.active_adapters()["A"]
    assert view.slot == 0
    assert view.config == {"rm_type": "x"}
    assert view.version == 1


def test_snapshot_reports_sets_in_registry_vocabulary():
    registry = AdapterRegistry(max_adapters=4)
    register_and_promote(registry, "A")
    registry.register("B", None)
    snapshot = registry.snapshot()
    assert set(snapshot["active"]) == {"A"}
    assert set(snapshot["pending"]) == {"B"}
    assert snapshot["retiring"] == {}
    assert snapshot["cleanup"] == []
    assert set(registry.active_adapters()) == {"A"}  # only active adapters are sampleable


def test_slot_version_is_monotonic_across_slot_reuse():
    registry = AdapterRegistry(max_adapters=2)
    register_and_promote(registry, "A")  # slot 0, version 1
    registry.record_weight_update(["A"])  # version 2
    registry.deregister("A")
    registry.retire_adapters()
    registry.free_slot("A")

    registry.register("A2", None)  # reuses slot 0
    assert registry.snapshot()["pending"]["A2"].version == 2  # inherits, not reset
    registry.record_weight_update(["A2"])
    assert registry.active_adapters()["A2"].version == 3


def test_record_weight_update_only_touches_reported_names():
    registry = AdapterRegistry(max_adapters=4)
    register_and_promote(registry, "A")
    register_and_promote(registry, "B")
    registry.record_weight_update(["A"])
    assert registry.active_adapters()["A"].version == 2
    assert registry.active_adapters()["B"].version == 1


def test_register_name_rejected_until_cleanup_done():
    registry = AdapterRegistry(max_adapters=4)
    register_and_promote(registry, "A")
    registry.deregister("A")
    with pytest.raises(ValueError, match="cleaning up"):
        registry.register("A", None)  # retiring
    registry.retire_adapters()
    with pytest.raises(ValueError, match="cleaning up"):
        registry.register("A", None)  # cleanup
    registry.free_slot("A")
    assert registry.register("A", None) == {"name": "A", "slot": 0}


def test_deregister_retires_but_keeps_serving_until_demoted():
    registry = AdapterRegistry(max_adapters=4)
    register_and_promote(registry, "A")
    registry.deregister("A")
    assert registry.adapter_state("A") == AdapterState.RETIRING
    assert "A" in registry.active_adapters()  # still sampleable this iteration
    assert "A" in registry.snapshot()["retiring"]
    assert registry.retire_adapters() == ["A"]
    assert registry.active_adapters() == {}
    assert registry.adapter_state("A") == AdapterState.CLEANUP
    assert registry.retire_adapters() == []  # idempotent


def test_batch_record_counts_steps_on_confirmation():
    registry = AdapterRegistry(max_adapters=4)
    register_and_promote(registry, "A")
    register_and_promote(registry, "B")

    registry.record_batch_adapters(7, ["A"])
    assert registry.step_count("A") == 0  # recorded, not yet trained
    assert registry.mark_batch_trained(7) == ["A"]
    assert registry.step_count("A") == 1
    assert registry.step_count("B") == 0
    assert registry.mark_batch_trained(7) == []  # record consumed


def test_batch_trained_counts_deregistered_adapter_until_freed():
    registry = AdapterRegistry(max_adapters=4)
    register_and_promote(registry, "A")
    registry.record_batch_adapters(3, ["A"])
    registry.deregister("A")  # deregistered while its batch is training
    assert registry.mark_batch_trained(3) == ["A"]
    assert registry.step_count("A") == 1  # final ckpt reads this
    registry.retire_adapters()
    assert registry.step_count("A") == 1  # cleanup record still holds it
    registry.free_slot("A")
    assert registry.step_count("A") == 0


def test_set_step_on_resume():
    registry = AdapterRegistry(max_adapters=2)
    registry.register("A", None)
    registry.set_step("A", 40)
    registry.record_batch_adapters(1, ["A"])
    registry.record_weight_update(["A"])
    registry.mark_batch_trained(1)
    assert registry.step_count("A") == 41


def test_deregister_holds_slot_until_free_slot():
    registry = AdapterRegistry(max_adapters=2)
    register_and_promote(registry, "A")  # slot 0
    register_and_promote(registry, "B")  # slot 1
    registry.deregister("A")
    registry.retire_adapters()
    assert not registry.free_slots  # slot 0 held until cleanup
    with pytest.raises(RuntimeError, match="No free adapter slots"):
        registry.register("C", None)
    registry.free_slot("A")
    assert registry.register("C", None) == {"name": "C", "slot": 0}


@pytest.mark.asyncio
async def test_custom_backend_validation_rejects():
    class StrictBackend(MultiLoRABackend):
        async def validate_adapter(self, name, config):
            if not config:
                raise ValueError("adapter config is required")

    backend = StrictBackend(make_args(), "http://unused")
    with pytest.raises(ValueError, match="config is required"):
        await backend.register("A", None)
    assert backend.registry.active_adapters() == {}

    result = await backend.register("A", {"rm_type": "x"})
    assert result == {"name": "A", "slot": 0}


def test_register_rejects_unsafe_names():
    registry = AdapterRegistry(max_adapters=4)
    for bad in ["a/b", "..", "a::b", "a b", ""]:
        with pytest.raises(ValueError, match="invalid"):
            registry.register(bad, None)
    registry.register("ok-name_1.2", None)


def test_register_rejects_duplicate_save_dir(tmp_path):
    registry = AdapterRegistry(max_adapters=4)
    registry.register("A", make_config(save=tmp_path / "x"))
    with pytest.raises(ValueError, match="already used by adapter 'A'"):
        registry.register("B", make_config(save=tmp_path / "x"))
    registry.register("C", make_config(save=tmp_path / "y"))


@pytest.mark.asyncio
async def test_save_dir_defaults_under_save_root(tmp_path):
    backend = make_backend(save=str(tmp_path))
    await backend.register("A", make_config())
    saved = backend.registry.records["A"].config.save
    assert saved == tmp_path / "adapters" / "A"


@pytest.mark.asyncio
async def test_explicit_save_dir_wins_over_root(tmp_path):
    backend = make_backend(save=str(tmp_path))
    await backend.register("A", make_config(save=tmp_path / "custom"))
    assert backend.registry.records["A"].config.save == tmp_path / "custom"


@pytest.mark.asyncio
async def test_register_fails_without_any_save_dir():
    backend = make_backend(save=None)
    with pytest.raises(ValueError, match="no save dir"):
        await backend.register("A", make_config())
