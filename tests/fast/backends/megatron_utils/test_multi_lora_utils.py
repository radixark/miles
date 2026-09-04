import logging
from types import SimpleNamespace

from miles.backends.megatron_utils import multi_lora_optimizer, multi_lora_utils

_MULTI_LORA_UTILS_LOGGER = "miles.backends.megatron_utils.multi_lora_utils"


class _FakeSlotScheduler:
    def __init__(self, lr: float) -> None:
        self.optimizer = SimpleNamespace(param_groups=[{"lr": lr}])
        self.increments: list[int] = []

    def step(self, increment: int) -> None:
        self.increments.append(increment)


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
