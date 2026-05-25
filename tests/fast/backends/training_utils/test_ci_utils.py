from argparse import Namespace

import pytest

from tests.ci.ci_register import register_cpu_ci

from miles.backends.training_utils.ci_utils import check_kl

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


def _make_args(**overrides):
    args = {
        "multi_latent_attention": False,
        "lora_rank": 0,
        "use_rollout_routing_replay": True,
        "ci_max_train_rollout_logprob_abs_diff": None,
        "ci_max_kl_loss": None,
    }
    args.update(overrides)
    return Namespace(**args)


def test_check_kl_accepts_configured_r3_metric_bounds():
    check_kl(
        _make_args(
            ci_max_train_rollout_logprob_abs_diff=0.028,
            ci_max_kl_loss=0.008,
        ),
        {
            "train/train_rollout_logprob_abs_diff": 0.027,
            "train/kl_loss": 0.007,
        },
        step_id=0,
        accumulated_step_id=0,
    )


def test_check_kl_rejects_train_rollout_logprob_bound():
    with pytest.raises(AssertionError, match="train/train_rollout_logprob_abs_diff"):
        check_kl(
            _make_args(ci_max_train_rollout_logprob_abs_diff=0.028),
            {
                "train/train_rollout_logprob_abs_diff": 0.028,
            },
            step_id=0,
            accumulated_step_id=0,
        )


def test_check_kl_rejects_missing_configured_metric():
    with pytest.raises(AssertionError, match="train/kl_loss missing"):
        check_kl(
            _make_args(ci_max_kl_loss=0.008),
            {},
            step_id=0,
            accumulated_step_id=0,
        )
