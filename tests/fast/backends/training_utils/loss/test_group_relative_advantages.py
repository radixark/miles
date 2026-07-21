"""GRPO group-relative advantage with the rollout (not the training sample) as the unit.

Pins the reward-attribution contract for fanned agent rollouts: a rollout split into
several segments (compaction / sub-agent / fork branches) that share a ``rollout_id``
counts once in the per-prompt baseline, and every segment carries the rollout's advantage.
In miles ``rollout_id`` is ``Sample.index`` and ``prompt_id`` is ``Sample.group_index`` --
fanned segments are ``deepcopy`` of the same sample and therefore share both. With prompt
ids the baseline is bucketed per prompt even when rollout counts are uneven; without them
the legacy positional grouping is preserved verbatim.

These tests are CPU-only (torch on CPU, no GPU) and live under ``tests/fast`` so the CI
collector assigns them to the stage-a-cpu suite implicitly.
"""

import pytest
import torch

from miles.backends.training_utils.loss_hub.math_utils import group_relative_advantages


def test_default_one_sample_per_rollout_is_plain_per_prompt_group_norm():
    # 2 prompts x 2 rollouts, one sample each (unique rollout_ids) -> identity reduction.
    advantages = group_relative_advantages(
        [1.0, 3.0, 5.0, 11.0],
        rollout_ids=[0, 1, 2, 3],
        prompt_ids=[0, 0, 1, 1],
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        std_normalization=False,
    )
    assert advantages == pytest.approx([-1.0, 1.0, -3.0, 3.0])


def test_rigid_input_is_bit_identical_to_legacy_reshape_group_norm():
    # On the rigid default path (every prompt has exactly n_samples_per_prompt rollouts, in
    # prompt-major order) the prompt buckets are exactly the legacy reshape rows, so the
    # result must equal the previous inline reshape/mean/std computation bit for bit.
    raw_rewards = [0.13, 0.71, 1.37, 2.93, 0.29, 1.11, 3.71, 0.017]
    advantages = group_relative_advantages(
        raw_rewards,
        rollout_ids=list(range(8)),
        prompt_ids=[0, 0, 0, 0, 1, 1, 1, 1],
        n_samples_per_prompt=4,
        rollout_batch_size=2,
        std_normalization=True,
    )

    legacy = torch.tensor(raw_rewards, dtype=torch.float).reshape(-1, 4)
    legacy = legacy - legacy.mean(dim=-1, keepdim=True)
    legacy = legacy / (legacy.std(dim=-1, keepdim=True) + 1e-6)
    assert advantages == legacy.flatten().tolist()


def test_fanned_rollout_counts_once_and_segments_share_the_advantage():
    # Rollout 1 (rollout_id 1) fans into two segments that repeat its outcome reward 3.0.
    # The baseline is over the 4 rollouts, NOT the 5 samples, so each prompt's mean is
    # unchanged by the fan-out: the two segments share rollout 1's advantage (+1), and
    # rollout 0 stays at -1. A double-counting / global baseline would subtract
    # mean([1,3,3,5,11]) or mean([1,3,5,11]) and shift these values.
    advantages = group_relative_advantages(
        [1.0, 3.0, 3.0, 5.0, 11.0],
        rollout_ids=[0, 1, 1, 2, 3],
        prompt_ids=[0, 0, 0, 1, 1],
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        std_normalization=False,
    )
    assert advantages == pytest.approx([-1.0, 1.0, 1.0, -3.0, 3.0])


def test_std_normalization_divides_by_per_prompt_group_std():
    advantages = group_relative_advantages(
        [0.0, 2.0, 10.0, 14.0],
        rollout_ids=[0, 1, 2, 3],
        prompt_ids=[0, 0, 1, 1],
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        std_normalization=True,
    )
    # Each prompt has two rollouts symmetric about the mean, so both prompts normalize to
    # +-1/sqrt(2) regardless of scale (unbiased std), confirming per-prompt std division.
    assert advantages == pytest.approx([-0.70710, 0.70710, -0.70710, 0.70710], abs=1e-4)


def test_uneven_groups_get_per_prompt_baselines():
    # 3 rollouts on prompt 0, 2 on prompt 1 (5 != 2 * 2): the legacy positional path could
    # only fall back to one global group (mean 6.0 -> [-5, -4, -3, 4, 8]); prompt ids
    # recover the true per-prompt centering instead.
    advantages = group_relative_advantages(
        [1.0, 2.0, 3.0, 10.0, 14.0],
        rollout_ids=[0, 1, 2, 3, 4],
        prompt_ids=[0, 0, 0, 1, 1],
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        std_normalization=False,
    )
    assert advantages == pytest.approx([-1.0, 0.0, 1.0, -2.0, 2.0])
    assert advantages != pytest.approx([-5.0, -4.0, -3.0, 4.0, 8.0])


def test_without_prompt_ids_uneven_counts_fall_back_to_one_global_group():
    # 3 rollouts when 4 are expected and no prompt metadata -> single global group
    # (the documented legacy fallback for custom rollout paths).
    advantages = group_relative_advantages(
        [1.0, 2.0, 3.0],
        rollout_ids=[0, 1, 2],
        prompt_ids=None,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        std_normalization=False,
    )
    assert advantages == pytest.approx([-1.0, 0.0, 1.0])


def test_none_rollout_id_raises():
    # Sample.index defaults to None; without the guard, the dedup would silently merge all
    # None-id samples into one rollout and zero out their advantages. prompt_ids is absent
    # here (the custom-rollout scenario that hits this), so only the guard can catch it.
    with pytest.raises(ValueError, match="rollout id"):
        group_relative_advantages(
            [1.0, 3.0, 5.0, 11.0],
            rollout_ids=[None, None, None, None],
            prompt_ids=None,
            n_samples_per_prompt=2,
            rollout_batch_size=2,
            std_normalization=False,
        )


def test_rollout_id_spanning_two_prompts_raises():
    with pytest.raises(ValueError, match="exactly one prompt"):
        group_relative_advantages(
            [1.0, 3.0, 5.0, 11.0],
            rollout_ids=[0, 1, 1, 2],
            prompt_ids=[0, 0, 1, 1],
            n_samples_per_prompt=2,
            rollout_batch_size=2,
            std_normalization=False,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
