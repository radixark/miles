from argparse import Namespace

import pytest

from miles.backends.training_utils.train_iters import compute_train_iters


def test_eval_only_num_rollout_zero_is_one_iter():
    args = Namespace(num_rollout=0, rollout_batch_size=8, n_samples_per_prompt=8, global_batch_size=16)
    assert compute_train_iters(args) == 1


def test_normal_run_uses_integer_division():
    args = Namespace(num_rollout=4, rollout_batch_size=8, n_samples_per_prompt=8, global_batch_size=16)
    assert compute_train_iters(args) == 16


def test_too_few_samples_raises():
    args = Namespace(num_rollout=1, rollout_batch_size=1, n_samples_per_prompt=1, global_batch_size=16)
    with pytest.raises(ValueError, match="global_batch_size"):
        compute_train_iters(args)
