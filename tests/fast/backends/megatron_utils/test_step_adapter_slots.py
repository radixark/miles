"""step_adapter_slots settles each slot identically on every rank and always consumes its grads."""

from types import SimpleNamespace

from miles.backends.megatron_utils.lora import optimizer as optimizer_module


class _FakeChild:
    def __init__(self, step_error=None):
        self._step_error = step_error
        self.stepped = False

    def prepare_grads(self):
        return False  # BF16 runs have no grad scaler; Megatron never flags inf here

    def get_parameters(self):
        return []

    def get_grads_for_grad_norm(self):
        return []

    def step_with_ready_grads(self):
        if self._step_error is not None:
            raise self._step_error
        self.stepped = True


class _FakeOptimizer:
    def __init__(self):
        self.param_gathers = 0

    def allgather_params(self):
        self.param_gathers += 1


def _step_slots(monkeypatch, children_by_slot, grad_norm=1.5):
    optimizer = _FakeOptimizer()
    zeroed = []
    monkeypatch.setattr(optimizer_module, "_slot_children", lambda _optimizer, slot: children_by_slot[slot])
    monkeypatch.setattr(optimizer_module, "zero_adapter_slot_grads", lambda _model, slot: zeroed.append(slot))
    monkeypatch.setattr(optimizer_module, "get_grad_norm_fp32", lambda *_args, **_kwargs: grad_norm)
    monkeypatch.setattr(optimizer_module, "clip_grad_by_total_norm_fp32", lambda *_args: None)
    monkeypatch.setattr(optimizer_module.dist, "is_initialized", lambda: False)
    outcomes = optimizer_module.step_adapter_slots(
        optimizer, model=None, step_batch_sizes={slot: 1 for slot in children_by_slot}, clip_grad=1.0
    )
    return outcomes, zeroed, optimizer


def test_each_slot_settles_on_its_own(monkeypatch):
    healthy, failing = _FakeChild(), _FakeChild(step_error=RuntimeError("boom"))
    outcomes, zeroed, optimizer = _step_slots(monkeypatch, {0: [healthy], 1: [failing]})
    assert outcomes[0] == {"grad_norm": 1.5} and healthy.stepped
    assert outcomes[1] == {"error": "RuntimeError: boom"}
    assert zeroed == [0, 1], "grads are consumed whether or not the step landed"
    assert optimizer.param_gathers == 1


def test_a_nonfinite_grad_norm_skips_the_step(monkeypatch):
    """BF16 has no grad scaler, so the all-reduced norm is the only inf/nan gate."""
    child = _FakeChild()
    outcomes, zeroed, optimizer = _step_slots(monkeypatch, {0: [child]}, grad_norm=float("inf"))
    assert outcomes[0] == {"skipped_nonfinite": 1.0}
    assert not child.stepped and zeroed == [0]
    assert optimizer.param_gathers == 0


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
