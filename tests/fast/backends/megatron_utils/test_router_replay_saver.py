import sys
import types

import torch

if "megatron.core" not in sys.modules:
    parallel_state_module = types.ModuleType("megatron.core.parallel_state")
    parallel_state_module.get_data_parallel_rank = lambda *args, **kwargs: 0
    parallel_state_module.get_tensor_model_parallel_rank = lambda *args, **kwargs: 0
    parallel_state_module.get_pipeline_model_parallel_rank = lambda *args, **kwargs: 0
    parallel_state_module.get_data_parallel_world_size = lambda *args, **kwargs: 1
    parallel_state_module.get_tensor_model_parallel_world_size = lambda *args, **kwargs: 1

    core_module = types.ModuleType("megatron.core")
    core_module.parallel_state = parallel_state_module
    megatron_module = types.ModuleType("megatron")
    megatron_module.core = core_module

    sys.modules["megatron"] = megatron_module
    sys.modules["megatron.core"] = core_module
    sys.modules["megatron.core.parallel_state"] = parallel_state_module

from miles.backends.megatron_utils.router_replay_saver import (
    _truncate_predictive_metric_tensor_payload_tokens,
    _truncate_router_replay_save_dict_tokens,
)


def test_truncate_router_replay_save_dict_tokens_limits_token_aligned_tensors():
    save_dict = {
        "compute_log_prob": {0: torch.randn(12, 4)},
        "training": {1: torch.randn(9, 4)},
        "predictive_bias": {2: torch.randn(15, 1, 4)},
        "global_token_ids": torch.arange(20, dtype=torch.long),
        "router_weights": {0: torch.randn(4, 4)},
    }

    truncated = _truncate_router_replay_save_dict_tokens(save_dict, max_tokens=10)

    assert truncated["compute_log_prob"][0].shape[0] == 10
    assert truncated["training"][1].shape[0] == 9
    assert truncated["predictive_bias"][2].shape[0] == 10
    assert truncated["global_token_ids"].shape[0] == 10
    assert truncated["router_weights"][0].shape == (4, 4)


def test_truncate_predictive_metric_tensor_payload_tokens_limits_each_layer():
    payload = {
        "layers": {
            "0": {
                "old_logits": torch.randn(14, 8),
                "current_logits": torch.randn(14, 8),
            },
            "3": {
                "predicted_delta_logits": torch.randn(7, 8),
            },
        }
    }

    truncated = _truncate_predictive_metric_tensor_payload_tokens(payload, max_tokens=10)

    assert truncated["layers"]["0"]["old_logits"].shape[0] == 10
    assert truncated["layers"]["0"]["current_logits"].shape[0] == 10
    assert truncated["layers"]["3"]["predicted_delta_logits"].shape[0] == 7
