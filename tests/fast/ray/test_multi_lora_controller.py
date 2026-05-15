"""Tests for MultiLoRAController state machine, slot management, and lifecycle."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-fast")


import pytest
import yaml

from miles.ray.multi_lora_controller import MultiLoRAControllerImpl
from miles.utils.adapter_config import AdapterConfig, AdapterState


def make_config(tmp_path, **overrides) -> AdapterConfig:
    defaults = dict(
        rank=8, alpha=16,
        data="/data.parquet",
        dir=str(tmp_path / "adapter_dir"),
        input_key="text", label_key="label",
        rm_type="math",
    )
    return AdapterConfig(**(defaults | overrides))


def drain_to_trainable(controller: MultiLoRAControllerImpl, name: str) -> None:
    """Walk an adapter through ACTIVE → DRAINING_TRAINABLE."""
    for s in (
        AdapterState.ACTIVE,
        AdapterState.DRAINING_DATASOURCE,
        AdapterState.DRAINING_INFLIGHT,
        AdapterState.DRAINING_TRAINABLE,
    ):
        controller.update_adapter_state(name, s)


@pytest.fixture
def controller():
    return MultiLoRAControllerImpl(max_adapters=2, max_rank=32)


# ---------------------------------------------------------------------------
# register_adapter
# ---------------------------------------------------------------------------


class TestRegisterAdapter:
    def test_pending_state_and_slot_zero(self, controller, tmp_path):
        result = controller.register_adapter("a", make_config(tmp_path))
        assert result == {"name": "a", "slot": 0}
        assert controller.configs["a"].state == AdapterState.PENDING
        assert controller.configs["a"].slot == 0

    def test_lowest_slot_allocated_first(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        controller.register_adapter("b", make_config(tmp_path))
        assert controller.configs["a"].slot == 0
        assert controller.configs["b"].slot == 1
        assert controller.free_slots == set()

    def test_creates_dir(self, tmp_path):
        c = MultiLoRAControllerImpl(max_adapters=1, max_rank=32)
        target = tmp_path / "deep" / "nested" / "adapter"
        assert not target.exists()
        c.register_adapter("a", make_config(tmp_path, dir=str(target)))
        assert target.is_dir()

    def test_existing_dir_no_error(self, controller, tmp_path):
        existing = tmp_path / "exists"
        existing.mkdir()
        controller.register_adapter("a", make_config(tmp_path, dir=str(existing)))
        assert existing.is_dir()

    def test_rank_exceeds_max_raises(self, controller, tmp_path):
        with pytest.raises(AssertionError, match="exceeds max rank"):
            controller.register_adapter("a", make_config(tmp_path, rank=64))

    def test_duplicate_name_raises(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        with pytest.raises(ValueError, match="already registered"):
            controller.register_adapter("a", make_config(tmp_path))

    def test_no_free_slots_raises(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        controller.register_adapter("b", make_config(tmp_path))
        with pytest.raises(RuntimeError, match="No free adapter slots"):
            controller.register_adapter("c", make_config(tmp_path))

    def test_invalid_type_raises(self, controller):
        with pytest.raises(ValueError, match="Invalid type"):
            controller.register_adapter("a", 123)

    def test_register_from_yaml_path(self, controller, tmp_path):
        yaml_path = tmp_path / "adapter.yaml"
        yaml_path.write_text(yaml.safe_dump({
            "rank": 16, "alpha": 32,
            "data": "/d.parquet",
            "label_key": "label",
            "rm_type": "math",
            "dir": str(tmp_path / "out"),
        }))
        controller.register_adapter("a", str(yaml_path))
        assert controller.configs["a"].rank == 16
        assert controller.configs["a"].slot == 0


# ---------------------------------------------------------------------------
# update_adapter_state
# ---------------------------------------------------------------------------


class TestUpdateAdapterState:
    def test_single_name(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        controller.update_adapter_state("a", AdapterState.ACTIVE)
        assert controller.configs["a"].state == AdapterState.ACTIVE

    def test_list_of_names(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        controller.register_adapter("b", make_config(tmp_path))
        controller.update_adapter_state(["a", "b"], AdapterState.ACTIVE)
        assert controller.configs["a"].state == AdapterState.ACTIVE
        assert controller.configs["b"].state == AdapterState.ACTIVE

    def test_unknown_name_raises(self, controller):
        with pytest.raises(KeyError, match="not registered"):
            controller.update_adapter_state("ghost", AdapterState.ACTIVE)

    def test_backward_transition_raises(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        controller.update_adapter_state("a", AdapterState.ACTIVE)
        with pytest.raises(AssertionError, match="Cannot transition"):
            controller.update_adapter_state("a", AdapterState.PENDING)

    def test_same_state_raises(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        with pytest.raises(AssertionError, match="Cannot transition"):
            controller.update_adapter_state("a", AdapterState.PENDING)


# ---------------------------------------------------------------------------
# deregister_adapter
# ---------------------------------------------------------------------------


class TestDeregisterAdapter:
    def test_unknown_raises(self, controller):
        with pytest.raises(KeyError):
            controller.deregister_adapter("ghost")

    def test_pending_goes_to_drained(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        controller.deregister_adapter("a")
        assert controller.configs["a"].state == AdapterState.DRAINED

    def test_active_goes_to_draining_datasource(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        controller.update_adapter_state("a", AdapterState.ACTIVE)
        controller.deregister_adapter("a")
        assert controller.configs["a"].state == AdapterState.DRAINING_DATASOURCE

    @pytest.mark.parametrize("state", [
        AdapterState.DRAINING_DATASOURCE,
        AdapterState.DRAINING_INFLIGHT,
        AdapterState.DRAINING_TRAINABLE,
        AdapterState.DRAINED,
    ])
    def test_already_draining_is_noop(self, controller, tmp_path, state):
        controller.register_adapter("a", make_config(tmp_path))
        controller.update_adapter_state("a", AdapterState.ACTIVE)
        controller.update_adapter_state("a", state)
        controller.deregister_adapter("a")
        assert controller.configs["a"].state == state


# ---------------------------------------------------------------------------
# mark_last_training_rollout_id
# ---------------------------------------------------------------------------


class TestMarkLastTrainingRolloutId:
    def test_records_first_value(self, controller):
        controller.mark_last_training_rollout_id("a", 5)
        assert controller.drain_until_rollout_id["a"] == 5

    def test_takes_max_with_existing(self, controller):
        controller.mark_last_training_rollout_id("a", 5)
        controller.mark_last_training_rollout_id("a", 3)
        assert controller.drain_until_rollout_id["a"] == 5
        controller.mark_last_training_rollout_id("a", 10)
        assert controller.drain_until_rollout_id["a"] == 10

    def test_list_of_names(self, controller):
        controller.mark_last_training_rollout_id(["a", "b"], 7)
        assert controller.drain_until_rollout_id == {"a": 7, "b": 7}


# ---------------------------------------------------------------------------
# report_training_completed
# ---------------------------------------------------------------------------


class TestReportTrainingCompleted:
    def test_monotonic_last_trained(self, controller):
        controller.report_training_completed(5)
        assert controller.last_trained_rollout_id == 5
        controller.report_training_completed(3)
        assert controller.last_trained_rollout_id == 5

    def test_drains_when_target_reached(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        drain_to_trainable(controller, "a")
        controller.mark_last_training_rollout_id("a", 5)
        controller.report_training_completed(5)
        assert controller.configs["a"].state == AdapterState.DRAINED

    def test_does_not_drain_when_target_not_reached(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        drain_to_trainable(controller, "a")
        controller.mark_last_training_rollout_id("a", 10)
        controller.report_training_completed(5)
        assert controller.configs["a"].state == AdapterState.DRAINING_TRAINABLE

    def test_only_drains_in_draining_trainable(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        controller.mark_last_training_rollout_id("a", 5)
        controller.report_training_completed(10)
        assert controller.configs["a"].state == AdapterState.PENDING

    def test_train_steps_increment(self, controller):
        controller.set_train_step("a", 0)
        controller.set_train_step("b", 5)
        controller.report_training_completed(0)
        assert controller.train_steps == {"a": 1, "b": 6}


# ---------------------------------------------------------------------------
# mark_removed
# ---------------------------------------------------------------------------


class TestMarkRemoved:
    def test_removes_and_returns_slot(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        controller.set_train_step("a", 3)
        slot = controller.mark_removed("a")
        assert slot == 0
        assert "a" not in controller.configs
        assert "a" not in controller.train_steps
        assert 0 in controller.free_slots

    def test_idempotent_returns_minus_one(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        controller.set_train_step("a", 0)
        controller.mark_removed("a")
        assert controller.mark_removed("a") == -1

    def test_unknown_returns_minus_one(self, controller):
        assert controller.mark_removed("ghost") == -1

    def test_slot_reusable(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        controller.set_train_step("a", 0)
        controller.register_adapter("b", make_config(tmp_path))
        controller.set_train_step("b", 0)
        controller.mark_removed("a")
        controller.register_adapter("c", make_config(tmp_path))
        assert controller.configs["c"].slot == 0


# ---------------------------------------------------------------------------
# adapter_configs / adapter_train_steps snapshots
# ---------------------------------------------------------------------------


class TestSnapshots:
    def test_adapter_configs_returns_copy(self, controller, tmp_path):
        controller.register_adapter("a", make_config(tmp_path))
        snap = controller.adapter_configs()
        snap.clear()
        assert "a" in controller.configs

    def test_adapter_train_steps_reflects_set(self, controller):
        controller.set_train_step("a", 7)
        assert controller.adapter_train_steps() == {"a": 7}
