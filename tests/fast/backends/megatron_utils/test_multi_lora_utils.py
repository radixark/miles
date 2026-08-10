import logging
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from miles.backends.megatron_utils import multi_lora_optimizer, multi_lora_utils
from miles.utils.adapter_config import AdapterRun, AdapterRunConfig

_MULTI_LORA_UTILS_LOGGER = "miles.backends.megatron_utils.multi_lora_utils"
_MULTI_LORA_LAYERS_MODULE = "megatron.bridge.peft.multi_lora_layers"


def _install_multi_lora_layers_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType(_MULTI_LORA_LAYERS_MODULE)
    module.MultiLoRALinear = type("_FakeMultiLoRALinear", (), {})
    module._iter_multi_lora_modules = lambda _model: []
    module.clear_adapter_slot = lambda *_args: None
    module.init_adapter_slot = lambda *_args, **_kwargs: None
    module.load_adapter = lambda *_args, **_kwargs: 1
    monkeypatch.setitem(sys.modules, _MULTI_LORA_LAYERS_MODULE, module)


class _FakeSlotScheduler:
    def __init__(self, lr: float) -> None:
        self.optimizer = SimpleNamespace(param_groups=[{"lr": lr}])
        self.increments: list[int] = []

    def step(self, increment: int) -> None:
        self.increments.append(increment)


class _FakeController:
    def __init__(self, snapshot: dict[str, Any] | None = None) -> None:
        self.events: list[tuple[str, Any]] = []
        self.snapshot_value = snapshot or {"active": {}, "retiring": {}}

    async def adapter_step(self, name: str) -> int:
        self.events.append(("adapter_step", name))
        return 7

    async def set_adapter_step(self, name: str, step: int) -> None:
        self.events.append(("set_adapter_step", (name, step)))

    async def free_slot(self, name: str) -> None:
        self.events.append(("free_slot", name))

    async def mark_batch_trained(self, rollout_id: int) -> None:
        self.events.append(("mark_batch_trained", rollout_id))

    async def snapshot(self) -> dict[str, Any]:
        self.events.append(("snapshot", None))
        return self.snapshot_value

    async def record_weight_update(self, names: list[str]) -> None:
        self.events.append(("record_weight_update", names))


def _adapter_lr_messages(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == _MULTI_LORA_UTILS_LOGGER]


class TestStepSteppedAdapterSlots:
    def test_stepped_slots_emit_the_train_tag_with_their_new_learning_rates(self, monkeypatch, caplog):
        """Slots whose adapter batch completes log one train-tagged adapter_lr record per rollout/step."""
        optimizer = SimpleNamespace(miles_slot_schedulers={0: _FakeSlotScheduler(0.5), 1: _FakeSlotScheduler(0.25)})
        monkeypatch.setattr(
            multi_lora_optimizer,
            "step_adapter_slots",
            lambda optimizer, model, step_batch_sizes, clip_grad: {0: 1.5, 1: 0.75},
        )

        with caplog.at_level(logging.INFO, logger=_MULTI_LORA_UTILS_LOGGER):
            max_grad_norm = multi_lora_utils.step_stepped_adapter_slots(
                SimpleNamespace(clip_grad=1.0),
                [],
                optimizer,
                {"step_adapter_batch_sizes": {0: 64, 1: 32}},
                rollout_id=4,
                step_id=9,
            )

        assert max_grad_norm == 1.5
        assert optimizer.miles_slot_schedulers[0].increments == [64]
        assert optimizer.miles_slot_schedulers[1].increments == [32]
        assert _adapter_lr_messages(caplog) == ["train op=adapter_lr rollout=4 step=9 slot_0=0.5 slot_1=0.25"]


class TestIndependentControllerBookkeeping:
    def test_cleanup_saves_the_controller_step_and_releases_the_slot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cleanup saves the committed step, clears local state, and releases the controller slot."""
        controller = _FakeController()
        adapter = AdapterRun(name="alpha", slot=2, config=AdapterRunConfig(data="data", rank=4, alpha=8))
        model = [torch.nn.Module()]
        optimizer = SimpleNamespace(
            reload_model_params=lambda: None,
            miles_slot_schedulers={2: object()},
            miles_slot_child_indices={2: []},
            chained_optimizers=[],
        )
        saved_steps: list[dict[str, int]] = []
        monkeypatch.setattr(multi_lora_utils, "get_multi_lora_controller", lambda: controller)
        monkeypatch.setattr(
            "miles.backends.megatron_utils.initialize.is_first_replica_megatron_main_rank", lambda: True
        )
        monkeypatch.setattr(
            multi_lora_utils,
            "save_multi_lora_checkpoints",
            lambda _args, _model, steps, _adapters: saved_steps.append(steps),
        )
        _install_multi_lora_layers_stub(monkeypatch)

        assert multi_lora_utils.cleanup_adapters(SimpleNamespace(save_interval=1), model, optimizer, [adapter]) == 1

        assert saved_steps == [{"alpha": 7}]
        assert ("adapter_step", "alpha") in controller.events
        assert ("free_slot", "alpha") in controller.events
        assert optimizer.miles_slot_schedulers == {}

    def test_loading_a_fresh_adapter_installs_real_scheduler_without_controller_resume(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fresh adapter installs its slot and scheduler without overwriting controller progress."""
        controller = _FakeController()
        adapter = AdapterRun(
            name="fresh",
            slot=0,
            config=AdapterRunConfig(data="data", rank=4, alpha=8, rollout_batch_size=2, n_samples_per_prompt=2),
        )
        args = SimpleNamespace(
            lr_warmup_fraction=None,
            lr_warmup_iters=0,
            lr_warmup_init=0.0,
            lr=1e-4,
            min_lr=0.0,
            lr_decay_style="constant",
            start_weight_decay=0.0,
            end_weight_decay=0.0,
            weight_decay_incr_style="constant",
            lr_wsd_decay_iters=None,
            lr_wsd_decay_style="linear",
        )
        optimizer = SimpleNamespace(
            reload_model_params=lambda: None,
            miles_slot_child_indices={0: []},
            chained_optimizers=[],
        )
        monkeypatch.setattr(multi_lora_utils, "get_multi_lora_controller", lambda: controller)
        monkeypatch.setattr(
            "miles.backends.megatron_utils.initialize.is_first_replica_megatron_main_rank", lambda: True
        )
        _install_multi_lora_layers_stub(monkeypatch)

        assert multi_lora_utils.load_adapters(args, object(), optimizer, [adapter]) == 1

        assert 0 in optimizer.miles_slot_schedulers
        assert not any(event[0] == "set_adapter_step" for event in controller.events)

    def test_trained_batch_commit_updates_pending_push_and_controller(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A committed trained batch schedules its adapters for push and advances controller bookkeeping."""
        controller = _FakeController()
        monkeypatch.setattr(multi_lora_utils, "get_multi_lora_controller", lambda: controller)
        monkeypatch.setattr(
            "miles.backends.megatron_utils.initialize.is_first_replica_megatron_main_rank", lambda: True
        )

        pending: set[str] = set()
        multi_lora_utils.commit_trained_batch({"step_adapter_names": ["alpha"]}, 12, pending)

        assert pending == {"alpha"}
        assert ("mark_batch_trained", 12) in controller.events

    def test_weight_push_commit_records_the_updated_adapters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A main-rank weight push commits exactly the adapters whose weights changed."""
        controller = _FakeController()
        monkeypatch.setattr(multi_lora_utils, "get_multi_lora_controller", lambda: controller)

        multi_lora_utils.commit_weight_push(["alpha"], is_main_rank=True)

        assert ("record_weight_update", ["alpha"]) in controller.events

    def test_due_checkpoint_uses_the_controller_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A due adapter from the controller snapshot is saved at its committed step."""
        due = SimpleNamespace(step=6, config=SimpleNamespace(save="/nonexistent/checkpoint-root"))
        controller = _FakeController(snapshot={"active": {"due": due}, "retiring": {}})
        saved: list[tuple[dict[str, int], dict[str, Any]]] = []
        monkeypatch.setattr(multi_lora_utils, "get_multi_lora_controller", lambda: controller)
        monkeypatch.setattr(
            "miles.backends.megatron_utils.initialize.is_first_replica_megatron_main_rank", lambda: True
        )
        monkeypatch.setattr(
            multi_lora_utils,
            "save_multi_lora_checkpoints",
            lambda _args, _model, steps, adapters: saved.append((steps, adapters)),
        )

        assert multi_lora_utils.save_due_adapter_checkpoints(SimpleNamespace(save_interval=3), object())

        assert ("snapshot", None) in controller.events
        assert saved == [({"due": 6}, {"due": due})]

    def test_guarded_false_branches_do_not_contact_the_controller(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Disabled saves, empty version lists, and non-main ranks leave controller state untouched."""
        controller = _FakeController()
        monkeypatch.setattr(multi_lora_utils, "get_multi_lora_controller", lambda: controller)
        monkeypatch.setattr(
            "miles.backends.megatron_utils.initialize.is_first_replica_megatron_main_rank", lambda: False
        )

        pending: set[str] = set()
        multi_lora_utils.commit_trained_batch({}, 2, pending)
        multi_lora_utils.commit_weight_push([], is_main_rank=True)
        multi_lora_utils.commit_weight_push(["alpha"], is_main_rank=False)
        assert not multi_lora_utils.save_due_adapter_checkpoints(SimpleNamespace(save_interval=None), object())

        assert controller.events == []
