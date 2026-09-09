"""step_adapter_slots settles each slot identically on every rank and always consumes its grads."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from miles.backends.megatron_utils.lora import optimizer as optimizer_module

_MOD = "miles.backends.megatron_utils.lora.optimizer"


class _Child:
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


def _run(children_by_slot, grad_norm=1.5):
    optimizer = SimpleNamespace(allgather_params=MagicMock())
    zeroed = []
    with (
        patch(f"{_MOD}._slot_children", side_effect=lambda _opt, slot: children_by_slot[slot]),
        patch(f"{_MOD}.zero_adapter_slot_grads", side_effect=lambda _model, slot: zeroed.append(slot)),
        patch(f"{_MOD}.get_grad_norm_fp32", return_value=grad_norm),
        patch(f"{_MOD}.clip_grad_by_total_norm_fp32"),
    ):
        outcomes = optimizer_module.step_adapter_slots(
            optimizer, model=None, step_batch_sizes={slot: 1 for slot in children_by_slot}, clip_grad=1.0
        )
    return outcomes, zeroed, optimizer


def test_each_slot_settles_on_its_own():
    ok, bad = _Child(), _Child(step_error=RuntimeError("boom"))
    outcomes, zeroed, optimizer = _run({0: [ok], 1: [bad]})
    assert outcomes[0] == {"grad_norm": 1.5} and ok.stepped
    assert outcomes[1] == {"error": "RuntimeError: boom"}
    assert zeroed == [0, 1], "grads are consumed whether or not the step landed"
    optimizer.allgather_params.assert_called_once()


def test_a_nonfinite_grad_norm_skips_the_step():
    """BF16 has no grad scaler, so the all-reduced norm is the only inf/nan gate."""
    child = _Child()
    outcomes, zeroed, optimizer = _run({0: [child]}, grad_norm=float("inf"))
    assert outcomes[0] == {"skipped_nonfinite": 1.0}
    assert not child.stepped and zeroed == [0]
    optimizer.allgather_params.assert_not_called()


def test_a_rank_local_error_fails_the_slot_on_every_rank():
    """The healthy rank must not enter the parameter allgather alone."""
    remote = {0: {"error": "RuntimeError: died on rank 1"}}
    with (
        patch(f"{_MOD}.dist") as dist,
        patch(f"{_MOD}.get_gloo_group"),
    ):
        dist.is_initialized.return_value = True
        dist.get_world_size.return_value = 2

        def fake_gather(out, _local, group=None):
            out[0] = {0: {"grad_norm": 1.5}}  # this rank succeeded
            out[1] = remote

        dist.all_gather_object.side_effect = fake_gather
        merged = optimizer_module._merge_outcomes_across_ranks({0: {"grad_norm": 1.5}})
    assert merged == remote
