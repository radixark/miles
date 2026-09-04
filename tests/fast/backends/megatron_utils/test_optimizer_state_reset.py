"""reset_optimizer_states must reach every leaf optimizer under Megatron's wrappers and clear its
history by class, not by the --optimizer string (which Megatron rewrites before training starts)."""

import sys
from types import SimpleNamespace

import pytest
import torch
from megatron.core.optimizer.emerging_optimizers import TensorParallelAdaptiveMuon, TensorParallelMuon

from miles.backends.megatron_utils.optimizer_state_reset import UnsupportedOptimizerState, reset_optimizer_states


def stepped_adamw(*shapes):
    params = [torch.nn.Parameter(torch.ones(*shape)) for shape in shapes]
    for param in params:
        param.grad = torch.ones_like(param)
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    optimizer.step()
    return optimizer


class GroupStepAdamW(torch.optim.AdamW):
    """Shaped like TE FusedAdam: the step clock lives in param_groups, not in state."""

    def step(self, closure=None):
        super().step(closure)
        for group in self.param_groups:
            group["step"] = 50
        for state in self.state.values():
            del state["step"]


def muon_with_momentum(*shapes, cls=TensorParallelMuon, buffers=("momentum_buffer",)):
    optimizer = cls.__new__(cls)
    optimizer.param_groups = [{"params": [torch.nn.Parameter(torch.ones(*shape)) for shape in shapes]}]
    optimizer.state = {p: {name: torch.ones_like(p) for name in buffers} for p in optimizer.param_groups[0]["params"]}
    return optimizer


def wrap(leaf):
    return SimpleNamespace(optimizer=leaf)


def chain(*children):
    return SimpleNamespace(chained_optimizers=list(children))


def assert_adam_reset(optimizer):
    for group in optimizer.param_groups:
        if "step" in group:
            assert group["step"] == 0
    for state in optimizer.state.values():
        assert not state["exp_avg"].any()
        assert not state["exp_avg_sq"].any()
        if "step" in state:
            assert state["step"] == 0


def test_layer_wise_muon_tree_resets_both_leaves():
    muon, adam = muon_with_momentum((4, 4)), stepped_adamw((4,))
    # dist_muon: LayerWiseDistributedOptimizer(ChainedOptimizer) holding Float16 wrappers of Muon + Adam,
    # itself a child of the top-level ChainedOptimizer
    reset_optimizer_states(chain(chain(wrap(muon), wrap(adam))))

    assert all(not state["momentum_buffer"].any() for state in muon.state.values())
    assert_adam_reset(adam)


def test_adaptive_muon_resets_both_moment_buffers():
    adaptive = muon_with_momentum(
        (4, 4), cls=TensorParallelAdaptiveMuon, buffers=("momentum_buffer", "moment2_buffer")
    )

    reset_optimizer_states(chain(wrap(adaptive)))

    for state in adaptive.state.values():
        assert not state["momentum_buffer"].any()
        assert not state["moment2_buffer"].any()


def test_adam_reset_does_not_need_emerging_optimizers(monkeypatch):
    monkeypatch.setitem(sys.modules, "megatron.core.optimizer.emerging_optimizers", None)
    adam = stepped_adamw((4,))

    reset_optimizer_states(chain(wrap(adam)))

    assert_adam_reset(adam)
    with pytest.raises(ImportError):
        reset_optimizer_states(chain(wrap(muon_with_momentum((4, 4)))))


def test_hybrid_device_optimizer_resets_cpu_and_gpu_children():
    cpu_child, gpu_child = stepped_adamw((4,)), GroupStepAdamW([torch.nn.Parameter(torch.ones(4))], lr=1e-3)
    gpu_child.param_groups[0]["params"][0].grad = torch.ones(4)
    gpu_child.step()
    hybrid = SimpleNamespace(sub_optimizers=[cpu_child, gpu_child], state={}, param_groups=[])

    reset_optimizer_states(chain(wrap(hybrid)))

    assert_adam_reset(cpu_child)
    assert_adam_reset(gpu_child)


def test_tensor_valued_group_step_resets_in_place():
    adam = stepped_adamw((4,))
    clock = torch.tensor(50)
    adam.param_groups[0]["step"] = clock

    reset_optimizer_states(chain(wrap(adam)))

    assert clock.item() == 0


def test_wrapper_without_an_optimizer_is_skipped():
    adam = stepped_adamw((4,))
    reset_optimizer_states(chain(wrap(None), wrap(adam)))
    assert_adam_reset(adam)


def test_reset_before_the_first_step_is_a_no_op():
    reset_optimizer_states(chain(wrap(torch.optim.AdamW([torch.nn.Parameter(torch.ones(4))], lr=1e-3))))


def test_unknown_adam_state_key_is_rejected():
    adam = stepped_adamw((4,))
    next(iter(adam.state.values()))["max_exp_avg_sq"] = torch.ones(4)
    with pytest.raises(UnsupportedOptimizerState, match="max_exp_avg_sq"):
        reset_optimizer_states(chain(wrap(adam)))


def test_unknown_muon_state_key_is_rejected():
    muon = muon_with_momentum((4, 4))
    next(iter(muon.state.values()))["moment2_buffer"] = torch.ones(4, 4)
    with pytest.raises(UnsupportedOptimizerState, match="moment2_buffer"):
        reset_optimizer_states(chain(wrap(muon)))


def test_unknown_optimizer_class_is_rejected():
    sgd = torch.optim.SGD([torch.nn.Parameter(torch.ones(4))], lr=1e-3, momentum=0.9)
    with pytest.raises(UnsupportedOptimizerState, match="SGD"):
        reset_optimizer_states(chain(wrap(sgd)))
