from argparse import Namespace
from types import SimpleNamespace

import torch

from miles.backends.training_utils import cp_utils, log_utils


def test_every_tinker_conversion_key_is_handled(monkeypatch):
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        cp=SimpleNamespace(size=1),
        intra_dp=SimpleNamespace(size=1),
        is_pp_last_stage=True,
    )
    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(log_utils, "gather_log_data", lambda *a, **k: None)

    rollout_data = {
        "tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
        "response_lengths": [2],
        "rewards": [0.0],
        "raw_reward": [0.0],
        "truncated": [0],
        "loss_masks": [torch.tensor([1, 1], dtype=torch.int32)],
        "sample_indices": [0],
        "rollout_ids": [0],
        "rollout_mask_sums": torch.tensor([2]),
        "loss_weights": [torch.tensor([1.0, 1.0])],
        "advantages": [torch.tensor([0.0, 0.0])],
        "adapter_slots": [0],
        "batch_kind": "tinker",
        "tinker_operation_lanes": [0],
        "tinker_loss_by_lane": {0: {"loss_fn": "cross_entropy"}},
        "operation_by_lane": {0: "op-A"},
        "registration_by_lane": {0: ("A", "r-A")},
        "batch_execution_lease": {
            "dispatch_id": "d",
            "bindings_by_operation": [["op-A", ["A", "r-A", 0]]],
        },
        "tinker_forward_only": True,
        "tinker_logprob_collector": {},
        "dynamic_global_batch_size": 1,
        "n_adapters": 2,
    }

    # Every conversion key must be accepted without raising.
    log_utils.log_rollout_data(
        0,
        Namespace(
            ci_test=False,
            ci_disable_logprobs_checker=True,
            true_on_policy_mode=False,
            qkv_format="thd",
            log_multi_turn=False,
            log_passrate=False,
            log_correct_samples=False,
        ),
        rollout_data,
    )
