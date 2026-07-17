"""Fast tests for AdapterRegistry + MultiLoRABackend validation
(no Ray, no HTTP I/O, no SGLang, no torch)."""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import pytest

from miles.utils.adapter_config import AdapterRunConfig
from miles.utils.multi_lora import (
    AdapterRegistry,
    AdapterState,
    MultiLoRABackend,
    min_groups_per_dp_split,
    make_rid,
    parse_adapter,
)


def make_args(max_adapters: int = 4, save: str | None = None, dp_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        multi_lora_n_adapters=max_adapters,
        save=save,
        lora_rank=32,
        lora_alpha=32,
        rollout_batch_size=16,
        n_samples_per_prompt=4,
        multi_lora_dp_size=dp_size,
        multi_lora_max_adapter_global_batch_size=256,
    )


def make_backend(max_adapters: int = 4, save: str | None = None, dp_size: int = 2) -> MultiLoRABackend:
    return MultiLoRABackend(make_args(max_adapters, save, dp_size), "http://unused")


def make_config(save: str | None = None, **overrides) -> AdapterRunConfig:
    kwargs = dict(
        rank=8,
        alpha=16,
        data="/d",
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        save=save,
        input_key="text",
        label_key="label",
        rm_type="math",
    )
    kwargs.update(overrides)
    return AdapterRunConfig(**kwargs)


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


# make_config(): rollout_batch_size=4 groups/step, n_samples_per_prompt=4.


def test_mark_batch_trained_accumulates_and_steps_on_completion():
    registry = AdapterRegistry(max_adapters=4)
    register_and_promote(registry, "A", make_config())
    register_and_promote(registry, "B", make_config())

    # Two partial batches accumulate; the third completes the adapter batch.
    registry.record_batch_adapters(1, {"A": 1, "B": 2}, step_names=[])
    assert registry.mark_batch_trained(1) == []
    assert registry.records["A"].accumulated_groups == 1
    assert registry.records["B"].accumulated_groups == 2

    registry.record_batch_adapters(2, {"A": 1}, step_names=[])
    assert registry.mark_batch_trained(2) == []
    assert registry.records["A"].accumulated_groups == 2

    registry.record_batch_adapters(3, {"A": 2, "B": 2}, step_names=["A", "B"])
    assert registry.mark_batch_trained(3) == ["A", "B"]
    assert registry.step_count("A") == 1
    assert registry.step_count("B") == 1
    assert registry.records["A"].accumulated_groups == 0
    assert registry.records["B"].accumulated_groups == 0

    assert registry.mark_batch_trained(3) == []  # record consumed


def test_batch_trained_counts_deregistered_adapter_until_freed():
    registry = AdapterRegistry(max_adapters=4)
    register_and_promote(registry, "A", make_config())
    registry.record_batch_adapters(3, {"A": 4}, step_names=["A"])
    registry.deregister("A")  # deregistered while its batch is training
    assert registry.mark_batch_trained(3) == ["A"]
    assert registry.step_count("A") == 1  # final ckpt reads this
    registry.retire_adapters()
    assert registry.step_count("A") == 1  # cleanup record still holds it
    registry.free_slot("A")
    assert registry.step_count("A") == 0


def test_set_step_on_resume():
    registry = AdapterRegistry(max_adapters=2)
    registry.register("A", make_config())
    registry.set_step("A", 40)
    registry.record_batch_adapters(1, {"A": 4}, step_names=["A"])
    registry.record_weight_update(["A"])
    registry.mark_batch_trained(1)
    assert registry.step_count("A") == 41


def test_num_step_deregisters_on_committed_steps():
    registry = AdapterRegistry(max_adapters=2)
    register_and_promote(registry, "A", make_config(num_step=2))
    registry.record_batch_adapters(1, {"A": 4}, step_names=["A"])
    assert registry.mark_batch_trained(1) == ["A"]
    assert registry.adapter_state("A") == AdapterState.ACTIVE

    registry.record_batch_adapters(2, {"A": 4}, step_names=["A"])
    assert registry.mark_batch_trained(2) == ["A"]
    assert registry.step_count("A") == 2
    assert registry.adapter_state("A") == AdapterState.RETIRING


def test_num_step_is_relative_to_resume_step():
    registry = AdapterRegistry(max_adapters=2)
    register_and_promote(registry, "A", make_config(num_step=2))
    registry.set_step("A", 40)

    registry.record_batch_adapters(1, {"A": 4}, step_names=["A"])
    registry.mark_batch_trained(1)
    assert registry.step_count("A") == 41
    assert registry.adapter_state("A") == AdapterState.ACTIVE

    registry.record_batch_adapters(2, {"A": 4}, step_names=["A"])
    registry.mark_batch_trained(2)
    assert registry.step_count("A") == 42
    assert registry.adapter_state("A") == AdapterState.RETIRING


def test_min_groups_per_dp_split():
    assert min_groups_per_dp_split(n_samples_per_prompt=4, dp_size=8) == 2  # divisor
    assert min_groups_per_dp_split(n_samples_per_prompt=8, dp_size=8) == 1  # equal
    assert min_groups_per_dp_split(n_samples_per_prompt=16, dp_size=8) == 1  # multiple
    with pytest.raises(ValueError, match="divisor or a multiple"):
        min_groups_per_dp_split(n_samples_per_prompt=6, dp_size=8)


@pytest.mark.asyncio
async def test_register_resolves_batch_shape_defaults(tmp_path):
    backend = make_backend(save=str(tmp_path))
    await backend.register("A", AdapterRunConfig(data="/d"))
    config = backend.registry.records["A"].config
    assert config.rollout_batch_size == 16  # <- args.rollout_batch_size
    assert config.n_samples_per_prompt == 4  # <- args.n_samples_per_prompt
    assert config.rank == 32 and config.alpha == 32
    assert config.adapter_global_batch_size == 64


@pytest.mark.asyncio
async def test_register_rejects_bad_batch_shapes(tmp_path):
    backend = make_backend(save=str(tmp_path), dp_size=8)
    with pytest.raises(ValueError, match="divisor or a multiple"):
        await backend.register("B", make_config(n_samples_per_prompt=6, rollout_batch_size=4))
    with pytest.raises(ValueError, match="min_groups_per_dp_split"):
        # dp=8, n_samples=4 -> multiple of 2 groups; 3 groups is not
        await backend.register("C", make_config(rollout_batch_size=3))
    with pytest.raises(ValueError, match="exceeding"):
        await backend.register("D", make_config(rollout_batch_size=128))  # 512 samples > cap 256
    with pytest.raises(ValueError, match="exceeds the allocated maximum rank"):
        await backend.register("E", make_config(rank=64))
    with pytest.raises(ValueError, match="positive integer"):
        await backend.register("F", make_config(rollout_batch_size=0))
    with pytest.raises(ValueError, match="num_step must be a positive integer"):
        await backend.register("G", make_config(num_step=0))
    with pytest.raises(ValueError, match="num_epoch must be a positive integer"):
        await backend.register("H", make_config(num_epoch=0))
    # A valid shape registers fine.
    await backend.register("OK", make_config(rollout_batch_size=8))


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
async def test_free_slot_reaborts_before_releasing_slot():
    """Requests can survive the single retire-time abort (multi-turn groups
    submitting between turns, engine tokenizer-adapter batch misses); free_slot must
    fire one more abort round before the slot becomes reusable."""
    backend = make_backend()
    aborted: list[str] = []

    async def record_abort(name: str) -> None:
        aborted.append(name)

    backend.abort_adapter_requests = record_abort

    register_and_promote(backend.registry, "A")
    await backend.deregister("A")
    await backend.retire_adapters()
    assert aborted == ["A"]

    assert await backend.free_slot("A") == 0
    assert aborted == ["A", "A"]
    assert backend.registry.free_slots == {0, 1, 2, 3}


@pytest.mark.asyncio
async def test_free_slot_skips_abort_when_not_in_cleanup():
    backend = make_backend()
    aborted: list[str] = []

    async def record_abort(name: str) -> None:
        aborted.append(name)

    backend.abort_adapter_requests = record_abort

    register_and_promote(backend.registry, "A")  # ACTIVE, not CLEANUP
    assert await backend.free_slot("A") == -1
    assert await backend.free_slot("never-registered") == -1
    assert aborted == []


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
