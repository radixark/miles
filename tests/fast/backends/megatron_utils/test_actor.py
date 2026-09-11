import importlib
import inspect
import sys
from argparse import Namespace
from collections.abc import Iterator
from functools import partial
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from miles.backends.training_utils.model_companion import ModelCompanion
from miles.utils.tensor_backper import TensorBackuper
from miles.utils.types import SampleLineage

_ACTOR_MODULE_NAME = "miles.backends.megatron_utils.actor"


@pytest.fixture(scope="module")
def actor_module() -> Iterator[ModuleType]:
    """Import the Megatron actor with its unavailable native memory dependency stubbed."""
    package = importlib.import_module("miles.backends.megatron_utils")
    missing = object()
    saved_module = sys.modules.get(_ACTOR_MODULE_NAME, missing)
    saved_saver = sys.modules.get("torch_memory_saver", missing)
    saved_package_attr = getattr(package, "actor", missing)

    saver_module = ModuleType("torch_memory_saver")
    saver_module.torch_memory_saver = Mock()
    sys.modules["torch_memory_saver"] = saver_module
    sys.modules.pop(_ACTOR_MODULE_NAME, None)
    if saved_package_attr is not missing:
        delattr(package, "actor")

    try:
        yield importlib.import_module(_ACTOR_MODULE_NAME)
    finally:
        sys.modules.pop(_ACTOR_MODULE_NAME, None)
        if saved_module is not missing:
            sys.modules[_ACTOR_MODULE_NAME] = saved_module
        if saved_package_attr is missing:
            if hasattr(package, "actor"):
                delattr(package, "actor")
        else:
            package.actor = saved_package_attr
        if saved_saver is missing:
            sys.modules.pop("torch_memory_saver", None)
        else:
            sys.modules["torch_memory_saver"] = saved_saver


class TestCriticValuesValueSpec:
    def test_critic_values_are_shipped_as_a_typed_ragged_field(self, actor_module: ModuleType) -> None:
        """Variable-length critic sequences require the typed ragged object-store codec."""
        assert actor_module.CRITIC_VALUES_VALUE_SPEC["values"].codec == "typed_ragged"


def test_actor_ref_actor_switch_restores_model_companion(
    actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actor backup tag restores its companion after a ref model switch."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    real_empty_like = torch.empty_like
    monkeypatch.setattr(
        torch,
        "empty_like",
        lambda tensor, **kwargs: real_empty_like(
            tensor, **{key: value for key, value in kwargs.items() if key != "pin_memory"}
        ),
    )
    train_actor = object.__new__(actor_module.MegatronTrainRayActor)
    train_actor.args = SimpleNamespace(colocate=False, keep_old_actor=False, megatron_to_hf_mode="bridge")
    train_actor.with_ref = True
    train_actor.with_opd_teacher = False
    train_actor.model = [torch.nn.Module()]
    train_actor.model[0].add_module(
        "model_companion", ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))
    )
    weight = torch.nn.Parameter(torch.tensor([1.0]))
    train_actor.model[0].register_parameter("weight", weight)
    train_actor.weights_backuper = TensorBackuper.create(
        source_getter=partial(train_actor._named_actor_weights, include_model_companion=True)
    )
    train_actor._active_model_tag = "actor"
    actor_sample = SampleLineage(source_sample_index=7, output_index=0, output_count=1)
    ref_sample = SampleLineage(source_sample_index=8, output_index=0, output_count=1)
    train_actor.model[0].model_companion.record_sample_consumptions([actor_sample])
    train_actor.weights_backuper.backup("actor")
    weight.data.fill_(2)
    train_actor.model[0].model_companion.record_sample_consumptions([ref_sample])
    train_actor.weights_backuper.backup("ref")
    train_actor._active_model_tag = "ref"

    assert set(train_actor._get_actor_weights()) == {"vp_stages.0.weight"}

    train_actor._switch_model("actor")

    assert train_actor.model[0].model_companion.snapshot_sample_consumptions(is_skipped=False) == {actor_sample: 1}
    assert torch.equal(weight, torch.tensor([1.0]))
    assert set(train_actor._get_actor_weights()) == {"vp_stages.0.weight"}

    train_actor._switch_model("ref")

    assert train_actor.model[0].model_companion.snapshot_sample_consumptions(is_skipped=False) == {
        actor_sample: 1,
        ref_sample: 1,
    }
    assert torch.equal(weight, torch.tensor([2.0]))


class _FakeController:
    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot_value = snapshot
        self.events: list[tuple[str, str | None]] = []
        self.retirement_completed = False
        self.snapshot_consumed = False
        self.freed_slots: set[str] = set()

    async def retire_adapters(self) -> None:
        self.retirement_completed = True
        self.events.append(("retire_adapters", None))

    async def snapshot(self) -> dict[str, Any]:
        self.snapshot_consumed = True
        self.events.append(("snapshot", None))
        return self.snapshot_value

    async def free_slot(self, name: str) -> None:
        self.freed_slots.add(name)
        self.events.append(("free_slot", name))


class TestReconcileAdapters:
    def test_the_independent_controller_coroutines_are_completed_during_reconciliation(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First-replica reconciliation awaits retirement, snapshot, and cleanup-only slot release."""
        snapshot = {"active": {}, "pending": {}, "retiring": {}, "cleanup": ["orphan"]}
        controller = _FakeController(snapshot)
        train_actor = SimpleNamespace(
            args=SimpleNamespace(multi_lora=True),
            loaded_adapters={},
            model=object(),
            optimizer=object(),
            _multi_lora_pending_push=set(),
            weights_backuper=SimpleNamespace(backup=lambda _name: None),
        )
        monkeypatch.setattr(actor_module, "is_multi_lora_enabled", lambda _args: True)
        monkeypatch.setattr(actor_module, "is_first_replica_megatron_main_rank", lambda: True)
        monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)
        monkeypatch.setattr("miles.ray.multi_lora.controller.get_multi_lora_controller", lambda: controller)
        inspect.unwrap(actor_module.MegatronTrainRayActor.reconcile_adapters)(train_actor)

        assert controller.retirement_completed
        assert controller.snapshot_consumed
        assert controller.freed_slots == {"orphan"}


class TestSendCheckpoint:
    def test_healing_before_the_first_train_step_is_refused_without_sending(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Healing before the first train step is refused without transferring a checkpoint."""
        train_actor = object.__new__(actor_module.MegatronTrainRayActor)
        train_actor.args = Namespace(keep_old_actor=False)
        train_actor._last_rollout_id = None
        train_actor.model = object()
        train_actor.optimizer = object()
        train_actor.opt_param_scheduler = object()
        checkpoint_transfer_attempted = False

        def record_checkpoint_transfer(**_kwargs: object) -> None:
            nonlocal checkpoint_transfer_attempted
            checkpoint_transfer_attempted = True

        monkeypatch.setattr(actor_module, "get_parallel_state", lambda: SimpleNamespace(indep_dp=object()))
        monkeypatch.setattr(actor_module, "_send_ckpt", record_checkpoint_transfer)

        with pytest.raises(AssertionError, match="healing before the first train step is unsupported"):
            train_actor.send_ckpt(dst_rank=1)

        assert not checkpoint_transfer_attempted


@pytest.mark.parametrize(
    "enabled,role,outcome,representative,published",
    [
        (True, "actor", "NORMAL", True, True),
        (False, "actor", "NORMAL", True, False),
        (True, "critic", "NORMAL", True, False),
        (True, "actor", "DISCARDED_SHOULD_RETRY", True, False),
        (True, "actor", "NORMAL", False, False),
    ],
)
def test_training_witness_publishes_only_normal_actor_representatives(
    actor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    role: str,
    outcome: str,
    representative: bool,
    published: bool,
) -> None:
    """Only a normal actor result on the local representative publishes a snapshot."""
    from datetime import datetime, timezone

    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(enable_sample_ownership_checker=enabled)
    actor.role = role
    actor.model = []
    actor._cell_index = 2
    calls = []
    monkeypatch.setattr(actor_module, "is_local_replica_megatron_main_rank", lambda: representative)
    monkeypatch.setattr(
        actor_module.SampleOwnershipRecorder,
        "publish_cpu_witness",
        lambda *args, **kwargs: calls.append(kwargs) or "published-token",
    )
    result = actor._publish_training_witness(
        rollout_id=7,
        attempt=3,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        result=actor_module.TrainStepOutput(outcome=actor_module.TrainStepOutcome[outcome]),
    )
    assert bool(calls) is published
    assert result.sample_ownership_snapshot_id == ("published-token" if published else None)
    if published:
        assert calls[0]["replica_id"] == "cell-2"
        assert calls[0]["rollout_id"] == 7
        assert calls[0]["attempt"] == 3
