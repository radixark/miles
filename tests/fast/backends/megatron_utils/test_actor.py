import importlib
import inspect
import sys
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

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
