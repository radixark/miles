from types import SimpleNamespace

import pytest
import torch


class _FakeInnerOptimizer:
    def __init__(self, *, step: int) -> None:
        self.param = torch.nn.Parameter(torch.zeros(2))
        self.param.grad = torch.ones(2)
        self.state = {self.param: {"exp_avg": torch.ones(2), "exp_avg_sq": torch.ones(2)}}
        self.param_groups = [{"step": step, "params": [self.param]}]


class _FakeChainedMember:
    def __init__(self, *, is_stub_optimizer: bool = False, init_state_fn=None) -> None:
        self.is_stub_optimizer = is_stub_optimizer
        self.optimizer = _FakeInnerOptimizer(step=7)
        self.init_state_fn = init_state_fn
        self.config = "the optimizer config"
        self.zero_grad_calls: list[bool] = []

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.zero_grad_calls.append(set_to_none)
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                param.grad = None


class _RealChainedMember:
    def __init__(self, optimizer: torch.optim.Optimizer, *, init_state_fn=None) -> None:
        self.is_stub_optimizer = False
        self.optimizer = optimizer
        self.init_state_fn = init_state_fn
        self.config = None

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)


def _zero_adam_state(inner, config) -> None:
    for param_group in inner.param_groups:
        for param in param_group["params"]:
            inner.state[param] = {"exp_avg": torch.zeros_like(param.data), "exp_avg_sq": torch.zeros_like(param.data)}


def _chained_optimizer(*members) -> object:
    optimizer_module = pytest.importorskip("megatron.core.optimizer.optimizer")
    optimizer = optimizer_module.ChainedOptimizer.__new__(optimizer_module.ChainedOptimizer)
    optimizer.chained_optimizers = list(members)
    optimizer.config = SimpleNamespace(offload_optimizer_states=False)
    return optimizer


def _reset(monkeypatch, *members, using_pytorch_optimizer: bool = True) -> None:
    optimizer_utils = pytest.importorskip("miles.backends.megatron_utils.optimizer_utils")
    monkeypatch.setattr(optimizer_utils, "USING_PYTORCH_OPTIMIZER", using_pytorch_optimizer)
    optimizer_utils.reset_optimizer_state(_chained_optimizer(*members))


def _adam_with_history(*, steps: int) -> tuple[torch.nn.Parameter, torch.optim.Adam]:
    param = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.Adam([param], lr=0.1)
    for _ in range(steps):
        _step_on_fixed_gradient(param, optimizer)
    return param, optimizer


def _step_on_fixed_gradient(param: torch.nn.Parameter, optimizer: torch.optim.Adam) -> None:
    param.grad = torch.full_like(param, 0.5)
    optimizer.step()


def _assert_same_optimizer_state(left: torch.optim.Adam, right: torch.optim.Adam) -> None:
    left_state_dict, right_state_dict = left.state_dict(), right.state_dict()
    assert left_state_dict["param_groups"] == right_state_dict["param_groups"]
    assert list(left_state_dict["state"]) == list(right_state_dict["state"])
    for key, left_state in left_state_dict["state"].items():
        right_state = right_state_dict["state"][key]
        assert list(left_state) == list(right_state)
        for name, left_value in left_state.items():
            right_value = right_state[name]
            if isinstance(left_value, torch.Tensor):
                assert torch.equal(left_value, right_value)
            else:
                assert left_value == right_value


class TestResettingALiveOptimizer:
    def test_the_adam_moments_are_dropped(self, monkeypatch):
        """A trainer put back at the start of the run must not keep moments from the rollouts it is discarding."""
        member = _FakeChainedMember()

        _reset(monkeypatch, member)

        assert member.optimizer.state == {}

    def test_the_step_counter_a_fused_adam_keeps_on_the_param_group_is_dropped(self, monkeypatch):
        """TE and apex fused adam count steps there rather than in per-param state, so clearing state misses it."""
        member = _FakeChainedMember()

        _reset(monkeypatch, member)

        assert "step" not in member.optimizer.param_groups[0]

    def test_the_state_is_reallocated_when_the_optimizer_knows_how(self, monkeypatch):
        """A fused optimizer that saves a checkpoint before its first step would otherwise write no moments at all."""
        member = _FakeChainedMember(init_state_fn=_zero_adam_state)

        _reset(monkeypatch, member, using_pytorch_optimizer=False)

        assert set(next(iter(member.optimizer.state.values()))) == {"exp_avg", "exp_avg_sq"}

    def test_the_state_is_left_empty_for_a_torch_optimizer(self, monkeypatch):
        """The initializer writes the moments and not the step, and torch only fills a state it finds empty."""
        member = _FakeChainedMember(init_state_fn=_zero_adam_state)

        _reset(monkeypatch, member, using_pytorch_optimizer=True)

        assert member.optimizer.state == {}

    def test_the_gradients_are_dropped_too(self, monkeypatch):
        """A gradient computed against the weights being thrown away would be applied to the restored ones."""
        member = _FakeChainedMember()

        _reset(monkeypatch, member)

        assert member.zero_grad_calls == [True]

    def test_a_stub_optimizer_is_left_alone(self, monkeypatch):
        """A rank with no parameters of its own carries a stub whose inner optimizer is not there to reset."""
        stub = _FakeChainedMember(is_stub_optimizer=True)

        _reset(monkeypatch, stub, _FakeChainedMember())

        assert stub.optimizer.state != {} and stub.zero_grad_calls == []


class TestResettingARealAdam:
    def test_a_reset_optimizer_holds_what_a_freshly_built_one_holds(self, monkeypatch):
        """The fakes above only prove the calls happen; a real Adam has to end up where a new one starts."""
        _, optimizer = _adam_with_history(steps=3)
        assert optimizer.state_dict()["state"]

        _reset(monkeypatch, _RealChainedMember(optimizer))

        _assert_same_optimizer_state(optimizer, _adam_with_history(steps=0)[1])

    def test_a_reset_optimizer_then_steps_like_a_freshly_built_one(self, monkeypatch):
        """Matching state dicts could still diverge on the next step, which is the thing a rolled-back run does next."""
        param, optimizer = _adam_with_history(steps=3)
        _reset(monkeypatch, _RealChainedMember(optimizer))
        fresh_param, fresh_optimizer = _adam_with_history(steps=0)

        _step_on_fixed_gradient(param, optimizer)
        _step_on_fixed_gradient(fresh_param, fresh_optimizer)

        _assert_same_optimizer_state(optimizer, fresh_optimizer)

    def test_a_fused_adam_reallocates_the_moments_a_megatron_optimizer_expects_to_find(self, monkeypatch):
        """Outside the torch path mcore keeps the moments allocated, and a step onto an absent state raises."""
        param, optimizer = _adam_with_history(steps=3)

        _reset(
            monkeypatch,
            _RealChainedMember(optimizer, init_state_fn=_zero_adam_state),
            using_pytorch_optimizer=False,
        )

        state = optimizer.state[param]
        assert set(state) == {"exp_avg", "exp_avg_sq"}
        assert torch.count_nonzero(state["exp_avg"]) == 0 and torch.count_nonzero(state["exp_avg_sq"]) == 0

    def test_a_reset_drops_the_gradient_a_rolled_back_trainer_still_holds(self, monkeypatch):
        """That gradient was computed against the weights being thrown away, so applying it would corrupt the load."""
        param, optimizer = _adam_with_history(steps=1)

        _reset(monkeypatch, _RealChainedMember(optimizer))

        assert param.grad is None


class TestTheSelfCheckAResetRuns:
    def test_it_passes_over_a_fused_optimizer_the_reset_just_reinitialized(self, monkeypatch):
        """The fused path leaves zeroed moments rather than nothing, and both are what a fresh optimizer holds."""
        _reset(monkeypatch, _FakeChainedMember(init_state_fn=_zero_adam_state), using_pytorch_optimizer=False)

    def test_a_moment_left_behind_is_caught(self, monkeypatch):
        """A reset that missed a moment would let the run continue on the very state it meant to discard."""
        member = _FakeChainedMember(
            init_state_fn=lambda inner, config: inner.state.update({inner.param: {"exp_avg": torch.ones(2)}})
        )

        with pytest.raises(AssertionError):
            _reset(monkeypatch, member, using_pytorch_optimizer=False)

    def test_a_step_counter_left_on_the_param_group_is_caught(self, monkeypatch):
        """A fused adam keeps its step there, and a reset that leaves it resumes the schedule mid-run."""
        member = _FakeChainedMember(init_state_fn=lambda inner, config: inner.param_groups[0].update(step=7))

        with pytest.raises(AssertionError):
            _reset(monkeypatch, member, using_pytorch_optimizer=False)

    def test_a_gradient_left_behind_is_caught(self, monkeypatch):
        """That gradient was computed against the weights being thrown away."""
        member = _FakeChainedMember()
        monkeypatch.setattr(member, "zero_grad", lambda set_to_none=True: None)

        with pytest.raises(AssertionError):
            _reset(monkeypatch, member)

    def test_an_optimizer_that_reset_nothing_at_all_is_caught(self, monkeypatch):
        """A chain of nothing but stubs would let every assertion above pass without ever seeing a parameter."""
        with pytest.raises(AssertionError):
            _reset(monkeypatch, _FakeChainedMember(is_stub_optimizer=True))

    def test_a_state_holding_keys_no_adam_uses_is_caught(self, monkeypatch):
        """The value checks say nothing about a key they never see, so the set of keys is checked as well."""
        member = _FakeChainedMember(
            init_state_fn=lambda inner, config: inner.state.update({inner.param: {"momentum": torch.zeros(2)}})
        )

        with pytest.raises(AssertionError):
            _reset(monkeypatch, member, using_pytorch_optimizer=False)
