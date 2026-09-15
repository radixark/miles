"""Slot steps preserve nonfinite skips and propagate unknown execution failures."""

import pytest

from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer, step_slot_optimizers


class _FakeSlotOptimizer(SlotOptimizer):
    """Duck instance: exercises the choreography without a megatron build."""

    def __init__(self, slot, *, prepare_error=None, step_outcome=None):
        self.slot = slot
        self._prepare_error = prepare_error
        self._step_outcome = step_outcome or {"grad_norm": 1.5}
        self.adam_params = None
        self.stepped = False
        self.zeroed = False
        self.param_gathers = 0

    def apply_adam_params(self, adam_params):
        self.adam_params = adam_params

    def prepare_grads(self):
        if self._prepare_error is not None:
            raise self._prepare_error

    def clip_and_step(self, clip_grad):
        if "error" in self._step_outcome:
            raise RuntimeError(self._step_outcome["error"])
        self.stepped = "grad_norm" in self._step_outcome
        return self._step_outcome

    def zero_grads(self):
        self.zeroed = True

    def allgather_params(self):
        self.param_gathers += 1


def _step(slot_optimizers):
    adam = {slot: {"learning_rate": 1e-4} for slot in slot_optimizers}
    return step_slot_optimizers(slot_optimizers, adam, clip_grad=1.0)


def test_an_unknown_step_failure_escapes_before_success_is_reported():
    healthy = _FakeSlotOptimizer(0)
    failing = _FakeSlotOptimizer(1, step_outcome={"error": "boom"})
    with pytest.raises(RuntimeError, match="boom"):
        _step({0: healthy, 1: failing})
    assert healthy.stepped
    assert healthy.param_gathers == 0


def test_a_nonfinite_grad_norm_skips_the_step():
    """BF16 has no grad scaler, so the all-reduced norm is the only inf/nan gate."""
    skipped = _FakeSlotOptimizer(0, step_outcome={"skipped_nonfinite": 1.0})
    outcomes = _step({0: skipped})
    assert outcomes[0] == {"skipped_nonfinite": 1.0}
    assert not skipped.stepped and skipped.zeroed and skipped.param_gathers == 0


def test_an_unknown_preparation_failure_escapes():
    broken = _FakeSlotOptimizer(0, prepare_error=RuntimeError("prep died"))
    healthy = _FakeSlotOptimizer(1)
    with pytest.raises(RuntimeError, match="prep died"):
        _step({0: broken, 1: healthy})
    assert not broken.stepped and not healthy.stepped
