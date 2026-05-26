from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils import data as data_utils
from miles.backends.training_utils import log_utils


def test_true_on_policy_rollout_logprob_dtype_follows_training_precision():
    assert (
        data_utils._rollout_logprob_dtype(Namespace(true_on_policy_mode=True, bf16=True, fp16=False)) is torch.bfloat16
    )
    assert (
        data_utils._rollout_logprob_dtype(Namespace(true_on_policy_mode=True, bf16=False, fp16=True)) is torch.float16
    )
    assert (
        data_utils._rollout_logprob_dtype(Namespace(true_on_policy_mode=False, bf16=True, fp16=False)) is torch.float32
    )


def test_true_on_policy_log_checker_passes_when_values_and_dtype_match(monkeypatch):
    captured = {}
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        cp=SimpleNamespace(size=1),
        is_pp_last_stage=True,
    )

    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(
        log_utils,
        "gather_log_data",
        lambda metric_name, args, rollout_id, log_dict: captured.setdefault("log_dict", log_dict),
    )

    rollout_data = {
        "tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 1], dtype=torch.int32)],
        "log_probs": [torch.tensor([-13.25, -13.5], dtype=torch.bfloat16)],
        "rollout_log_probs": [torch.tensor([-13.25, -13.5], dtype=torch.bfloat16)],
    }

    log_utils.log_rollout_data(
        1,
        Namespace(
            ci_test=True,
            ci_disable_logprobs_checker=False,
            true_on_policy_mode=True,
            qkv_format="thd",
            log_multi_turn=False,
            log_passrate=False,
            log_correct_samples=False,
        ),
        rollout_data,
    )

    assert captured["log_dict"]["log_probs"] == captured["log_dict"]["rollout_log_probs"]


def test_true_on_policy_train_step_checker_passes_on_exact_logprob_match():
    log_dict = log_utils.log_train_step(
        args=Namespace(ci_test=True, true_on_policy_mode=True),
        loss_dict={"train_rollout_logprob_abs_diff": 0.0},
        grad_norm=0.0,
        rollout_id=0,
        step_id=0,
        num_steps_per_rollout=1,
        should_log=False,
    )

    assert log_dict["train/train_rollout_logprob_abs_diff"] == 0.0


def test_true_on_policy_train_step_checker_rejects_nonzero_logprob_diff():
    with pytest.raises(AssertionError, match="exact train/rollout logprob equality"):
        log_utils.log_train_step(
            args=Namespace(ci_test=True, true_on_policy_mode=True),
            loss_dict={"train_rollout_logprob_abs_diff": 1e-6},
            grad_norm=0.0,
            rollout_id=0,
            step_id=0,
            num_steps_per_rollout=1,
            should_log=False,
        )


def test_rollout_logging_handles_scalar_tensor_lists(monkeypatch):
    captured = {}
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        cp=SimpleNamespace(size=1),
        is_pp_last_stage=True,
    )

    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(
        log_utils,
        "gather_log_data",
        lambda metric_name, args, rollout_id, log_dict: captured.setdefault("log_dict", log_dict),
    )

    rollout_data = {
        "tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3, 3],
        "response_lengths": [2, 2],
        "loss_masks": [torch.tensor([1, 1], dtype=torch.int32), torch.tensor([1, 1], dtype=torch.int32)],
        "log_probs": [
            torch.tensor([-13.25, -13.5], dtype=torch.bfloat16),
            torch.tensor([-13.25, -13.5], dtype=torch.bfloat16),
        ],
        "rollout_log_probs": [
            torch.tensor([-13.25, -13.5], dtype=torch.bfloat16),
            torch.tensor([-13.25, -13.5], dtype=torch.bfloat16),
        ],
        "moe_aux_loss": [torch.tensor(1.0), torch.tensor(3.0)],
    }

    log_utils.log_rollout_data(
        1,
        Namespace(
            ci_test=False,
            ci_disable_logprobs_checker=False,
            true_on_policy_mode=True,
            qkv_format="thd",
            log_multi_turn=False,
            log_passrate=False,
            log_correct_samples=False,
        ),
        rollout_data,
    )

    assert captured["log_dict"]["moe_aux_loss"] == 2.0
