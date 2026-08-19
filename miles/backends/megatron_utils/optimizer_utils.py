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
