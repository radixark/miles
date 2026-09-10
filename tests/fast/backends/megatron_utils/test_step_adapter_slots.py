"""step_slot_optimizers settles each slot identically on every rank and always consumes its grads."""

from types import SimpleNamespace

from miles.backends.megatron_utils.lora import optimizer as optimizer_module
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


def _step(monkeypatch, slot_optimizers):
    monkeypatch.setattr(optimizer_module.dist, "is_initialized", lambda: False)
    adam = {slot: {"learning_rate": 1e-4} for slot in slot_optimizers}
    return step_slot_optimizers(slot_optimizers, adam, clip_grad=1.0)


def test_each_slot_settles_on_its_own(monkeypatch):
    healthy = _FakeSlotOptimizer(0)
    failing = _FakeSlotOptimizer(1, step_outcome={"error": "boom"})
    outcomes = _step(monkeypatch, {0: healthy, 1: failing})
    assert outcomes[0] == {"grad_norm": 1.5} and healthy.stepped
    assert outcomes[1] == {"error": "RuntimeError: boom"}
    assert healthy.zeroed and failing.zeroed, "grads are consumed whether or not the step landed"
    assert healthy.param_gathers == 1 and failing.param_gathers == 0


def test_a_nonfinite_grad_norm_skips_the_step(monkeypatch):
    """BF16 has no grad scaler, so the all-reduced norm is the only inf/nan gate."""
    skipped = _FakeSlotOptimizer(0, step_outcome={"skipped_nonfinite": 1.0})
    outcomes = _step(monkeypatch, {0: skipped})
    assert outcomes[0] == {"skipped_nonfinite": 1.0}
    assert not skipped.stepped and skipped.zeroed and skipped.param_gathers == 0


def test_a_preparation_failure_skips_the_slots_collectives_everywhere(monkeypatch):
    """A rank that failed before the norm all-reduce must not strand its peers in it."""
    broken = _FakeSlotOptimizer(0, prepare_error=RuntimeError("prep died"))
    healthy = _FakeSlotOptimizer(1)
    outcomes = _step(monkeypatch, {0: broken, 1: healthy})
    assert outcomes[0] == {"error": "RuntimeError: prep died"} and not broken.stepped
    assert outcomes[1] == {"grad_norm": 1.5} and healthy.param_gathers == 1


def test_a_rank_local_error_fails_the_slot_on_every_rank(monkeypatch):
    """The healthy rank must not enter the parameter allgather alone."""
    remote_outcomes = {0: {"error": "RuntimeError: died on rank 1"}}

    def gather_outcomes(per_rank, local_outcomes, group):
        per_rank[:] = [local_outcomes, remote_outcomes]

    monkeypatch.setattr(
        optimizer_module,
        "dist",
        SimpleNamespace(is_initialized=lambda: True, get_world_size=lambda: 2, all_gather_object=gather_outcomes),
    )
    monkeypatch.setattr(optimizer_module, "get_gloo_group", lambda: None)
    merged = optimizer_module._merge_outcomes_across_ranks({0: {"grad_norm": 1.5}})
    assert merged == remote_outcomes
