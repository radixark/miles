import torch

from miles.backends.megatron_utils.predictive_router_replay import (
    clear_predictive_optimizer_grads,
    disable_predictive_param_groups,
    restore_predictive_param_groups,
)


class _FakeOptimizer:
    def __init__(self, param_groups):
        self.param_groups = param_groups


def test_disable_predictive_param_groups_falls_back_to_high_lr_group():
    base_param = torch.nn.Parameter(torch.zeros(1))
    predictor_shard = torch.nn.Parameter(torch.zeros(1))
    optimizer = _FakeOptimizer(
        [
            {"params": [base_param], "lr": 2e-6, "max_lr": 2e-6, "weight_decay": 0.1},
            {"params": [predictor_shard], "lr": 2e-3, "max_lr": 2e-3, "weight_decay": 0.1},
        ]
    )

    saved_states = disable_predictive_param_groups(optimizer)

    assert len(saved_states) == 1
    assert optimizer.param_groups[0]["lr"] == 2e-6
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["lr"] == 0.0
    assert optimizer.param_groups[1]["weight_decay"] == 0.0

    restore_predictive_param_groups(saved_states)

    assert optimizer.param_groups[1]["lr"] == 2e-3
    assert optimizer.param_groups[1]["weight_decay"] == 0.1


def test_clear_predictive_optimizer_grads_clears_main_param_gradients():
    predictor_param = torch.nn.Parameter(torch.zeros(1))
    predictor_param.is_bias_predictor = True
    predictor_param.grad = torch.ones(1)
    predictor_param.main_grad = torch.ones(1)

    main_param = torch.nn.Parameter(torch.zeros(1))
    main_param.grad = torch.ones(1)
    predictor_param.main_param = main_param

    optimizer = _FakeOptimizer([{"params": [predictor_param], "lr": 1e-3, "weight_decay": 0.1}])

    clear_predictive_optimizer_grads(optimizer)

    assert predictor_param.grad is None
    assert torch.equal(predictor_param.main_grad, torch.zeros(1))
    assert main_param.grad is None
