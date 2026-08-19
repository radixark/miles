import torch
from megatron.core.optimizer import USING_PYTORCH_OPTIMIZER
from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer


def reset_optimizer_state(optimizer: MegatronOptimizer) -> None:
    assert isinstance(optimizer, ChainedOptimizer)

    for chained in optimizer.chained_optimizers:
        if chained.is_stub_optimizer:
            continue

        inner = chained.optimizer
        inner.state.clear()
        for param_group in inner.param_groups:
            param_group.pop("step", None)

        if chained.init_state_fn is not None and not USING_PYTORCH_OPTIMIZER:
            chained.init_state_fn(inner, chained.config)

        chained.zero_grad(set_to_none=True)

    _check_optimizer_state_empty(optimizer)


def _check_optimizer_state_empty(optimizer: MegatronOptimizer) -> None:
    checked_optimizers = 0
    checked_params = 0
    checked_state_keys: set[str] = set()

    for chained in optimizer.chained_optimizers:
        if chained.is_stub_optimizer:
            continue

        checked_optimizers += 1
        inner = chained.optimizer
        for param_group in inner.param_groups:
            assert "step" not in param_group
            for param in param_group["params"]:
                checked_params += 1
                assert param.grad is None

        for state in inner.state.values():
            checked_state_keys |= set(state)
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    assert torch.count_nonzero(value) == 0
                else:
                    assert value == 0

    assert checked_optimizers > 0
    assert checked_params > 0
    # if you use a non-adam optimizer, add the asserts for its state keys here
    assert checked_state_keys == (set() if USING_PYTORCH_OPTIMIZER else {"exp_avg", "exp_avg_sq"})
