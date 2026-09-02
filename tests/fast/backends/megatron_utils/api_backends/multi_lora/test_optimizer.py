import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import miles.backends.megatron_utils.api_backends.multi_lora.optimizer as multi_lora_optimizer
from miles.backends.megatron_utils.api_backends.multi_lora.optimizer import (
    _found_inf_anywhere,
    apply_adam_params_to_slot,
    build_multi_lora_operation_optimizer,
    step_adapter_slots,
)


class FakeChild:
    def __init__(self, grads, found_inf=False):
        self.params = [torch.nn.Parameter(torch.zeros(len(g))) for g in grads]
        for param, grad in zip(self.params, grads, strict=True):
            param.grad = torch.tensor(grad, dtype=torch.float32)
        self.found_inf = found_inf
        self.stepped = 0
        self.param_groups = [{"params": self.params, "lr": 0.0}]

    def prepare_grads(self):
        return self.found_inf

    def get_parameters(self):
        return self.params

    def get_main_grads_for_grad_norm(self):
        return [p.grad for p in self.params]

    def step_with_ready_grads(self):
        self.stepped += 1


class FakeChained:
    def __init__(self, children_by_slot):
        self.chained_optimizers = [child for children in children_by_slot.values() for child in children]
        self.miles_slot_child_indices, i = {}, 0
        for slot, children in children_by_slot.items():
            self.miles_slot_child_indices[slot] = list(range(i, i + len(children)))
            i += len(children)
        self.allgathered = 0

    def allgather_params(self):
        self.allgathered += 1


@pytest.fixture()
def torch_clip_grads(monkeypatch):
    fake = ModuleType("megatron.core.optimizer.clip_grads")

    def get_grad_norm_fp32(grads, grad_stats_parallel_group=None):
        return torch.norm(torch.stack([torch.norm(g) for g in grads])).item() if grads else 0.0

    def clip_grad_by_total_norm_fp32(params, max_norm, total_norm, _):
        coeff = max_norm / (total_norm + 1e-6)
        if coeff < 1.0:
            for p in params:
                p.grad.mul_(coeff)

    fake.get_grad_norm_fp32 = get_grad_norm_fp32
    fake.clip_grad_by_total_norm_fp32 = clip_grad_by_total_norm_fp32
    monkeypatch.setitem(sys.modules, "megatron.core.optimizer.clip_grads", fake)
    return fake


@pytest.fixture()
def no_slot_traversal(monkeypatch):
    monkeypatch.setattr(multi_lora_optimizer, "named_adapter_slot_parameters", lambda model, slot: iter(()))


class TestAdamParams:
    def test_lands_on_every_group_of_the_slot_only(self):
        mine, other = FakeChild([[1.0]]), FakeChild([[1.0]])
        chained = FakeChained({0: [mine], 1: [other]})
        apply_adam_params_to_slot(chained, 0, {"learning_rate": 5e-5, "beta1": 0.8, "weight_decay": 0.01})
        group = mine.param_groups[0]
        assert group["lr"] == 5e-5 and group["betas"] == (0.8, 0.95) and group["weight_decay"] == 0.01
        assert other.param_groups[0]["lr"] == 0.0


class TestStep:
    def test_gradient_sum_is_never_count_normalized(self, torch_clip_grads, no_slot_traversal):
        child = FakeChild([[3.0, 4.0]])
        chained = FakeChained({0: [child]})
        norms, vetoed, norm_blind = step_adapter_slots(chained, model=None, adam_params_by_slot={0: {}})
        assert vetoed == set()
        assert norms[0] == pytest.approx(5.0)
        assert child.stepped == 1 and chained.allgathered == 1

    def test_per_call_clip_scales_the_update(self, torch_clip_grads, no_slot_traversal):
        child = FakeChild([[3.0, 4.0]])
        chained = FakeChained({0: [child]})
        norms, _, _ = step_adapter_slots(chained, None, {0: {"grad_clip_norm": 1.0}})
        assert norms[0] == pytest.approx(5.0)
        assert torch.allclose(child.params[0].grad, torch.tensor([0.6, 0.8]), atol=1e-4)

    def test_zero_clip_means_no_clip(self, torch_clip_grads, no_slot_traversal):
        child = FakeChild([[30.0, 40.0]])
        chained = FakeChained({0: [child]})
        step_adapter_slots(chained, None, {0: {"grad_clip_norm": 0.0}})
        assert torch.allclose(child.params[0].grad, torch.tensor([30.0, 40.0]))

    def test_nonfinite_slot_is_vetoed_neighbours_step(self, torch_clip_grads, no_slot_traversal):
        bad = FakeChild([[float("nan"), 1.0]])
        good = FakeChild([[1.0, 0.0]])
        chained = FakeChained({0: [bad], 1: [good]})
        norms, vetoed, _ = step_adapter_slots(chained, None, {0: {}, 1: {}})
        assert vetoed == {0} and bad.stepped == 0
        assert list(norms) == [1] and good.stepped == 1
        assert chained.allgathered == 1

    def test_found_inf_from_prepare_grads_vetoes(self, torch_clip_grads, no_slot_traversal):
        child = FakeChild([[1.0]], found_inf=True)
        chained = FakeChained({0: [child]})
        norms, vetoed, _ = step_adapter_slots(chained, None, {0: {}})
        assert vetoed == {0} and norms == {} and child.stepped == 0
        assert chained.allgathered == 0

    def test_untouched_slots_retain_grads(self, torch_clip_grads, no_slot_traversal):
        stepped, retained = FakeChild([[1.0]]), FakeChild([[7.0]])
        chained = FakeChained({0: [stepped], 1: [retained]})
        step_adapter_slots(chained, None, {0: {}})
        assert retained.stepped == 0
        assert torch.allclose(retained.params[0].grad, torch.tensor([7.0]))

    def test_norm_blind_slot_is_refused_not_silently_stepped(self, torch_clip_grads, no_slot_traversal):
        class NormBlindChild(FakeChild):
            def get_main_grads_for_grad_norm(self):
                return []

        child = NormBlindChild([[3.0, 4.0]])
        chained = FakeChained({0: [child]})
        norms, vetoed, norm_blind = step_adapter_slots(chained, None, {0: {"grad_clip_norm": 1.0}})
        assert norm_blind == {0} and vetoed == set() and norms == {}
        assert child.stepped == 0

    def test_truly_zero_gradients_step_with_a_truthful_zero_norm(self, torch_clip_grads, no_slot_traversal):
        class NormBlindChild(FakeChild):
            def get_main_grads_for_grad_norm(self):
                return []

        child = NormBlindChild([[0.0, 0.0]])
        chained = FakeChained({0: [child]})
        norms, vetoed, norm_blind = step_adapter_slots(chained, None, {0: {}})
        assert norms == {0: 0.0} and vetoed == set() and norm_blind == set()
        assert child.stepped == 1


def test_found_inf_passthrough_without_dist():
    assert _found_inf_anywhere(True) is True
    assert _found_inf_anywhere(False) is False


class TestBuildGuards:
    def make(self, **overrides):
        config = SimpleNamespace(use_distributed_optimizer=False, fp16=False, bf16=True, optimizer="adam")
        config.__dict__.update(overrides)
        args = SimpleNamespace(multi_lora_n_adapters=2, use_gloo_process_groups=False)
        return args, config

    def test_rejects_distributed_optimizer_fp16_and_non_adam(self):
        for overrides, message in [
            (dict(use_distributed_optimizer=True), "use_distributed_optimizer=False"),
            (dict(fp16=True), "bf16"),
            (dict(optimizer="sgd"), "Adam semantics"),
        ]:
            args, config = self.make(**overrides)
            with pytest.raises(AssertionError, match=message):
                build_multi_lora_operation_optimizer(args, config, model_chunks=[])
