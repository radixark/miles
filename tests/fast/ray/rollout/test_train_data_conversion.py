from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import ray
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.fast.ray.rollout.conftest import make_args, make_sample, make_samples_grouped

from miles.ray.rollout.train_data_conversion import (
    _post_process_rewards,
    can_schedule_on_rollout_side,
    convert_samples_to_train_data,
    split_train_data_by_dp,
    split_train_data_by_dp_raw,
    split_train_data_by_dp_scheduled_raw,
)
from miles.utils import object_store
from miles.utils.types import Sample


@pytest.fixture(scope="module", autouse=True)
def _ray_minicluster():
    """split_train_data_by_dp uses ray.put(...) so we need Ray."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    yield


# ----------------------------- convert_samples_to_train_data -----------------------------


class TestConvertSamplesToTrainData:
    def test_default_path_produces_required_keys(self):
        args = make_args(advantage_estimator="grpo", rewards_normalization=False)
        samples = make_samples_grouped(n_groups=2, group_size=4)
        out = convert_samples_to_train_data(
            args,
            samples,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        for key in (
            "tokens",
            "response_lengths",
            "rewards",
            "raw_reward",
            "truncated",
            "sample_indices",
            "loss_masks",
        ):
            assert key in out, f"missing required key {key}"
        assert len(out["tokens"]) == len(samples)

    def test_loss_mask_none_filled_with_ones(self):
        args = make_args(rewards_normalization=False)
        s = make_sample(response_length=5)
        s.loss_mask = None
        out = convert_samples_to_train_data(
            args,
            [s],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["loss_masks"][0] == [1] * 5

    def test_remove_sample_zeroes_loss_mask(self):
        args = make_args(rewards_normalization=False)
        s = make_sample(response_length=4)
        s.loss_mask = [1, 1, 1, 1]
        s.remove_sample = True
        out = convert_samples_to_train_data(
            args,
            [s],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["loss_masks"][0] == [0, 0, 0, 0]

    def test_loss_mask_length_mismatch_asserts(self):
        args = make_args(rewards_normalization=False)
        s = make_sample(response_length=4)
        s.loss_mask = [1, 1]  # wrong length
        with pytest.raises(AssertionError):
            convert_samples_to_train_data(
                args,
                [s],
                metadata={},
                custom_convert_samples_to_train_data_func=None,
                custom_reward_post_process_func=None,
            )

    def test_truncated_status_marked(self):
        args = make_args(rewards_normalization=False)
        s = make_sample(status=Sample.Status.TRUNCATED)
        out = convert_samples_to_train_data(
            args,
            [s],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["truncated"][0] == 1

    def test_optional_field_rollout_log_probs_passed_through(self):
        args = make_args(rewards_normalization=False)
        s = make_sample()
        s.rollout_log_probs = [-0.1, -0.2, -0.3, -0.4]
        out = convert_samples_to_train_data(
            args,
            [s],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["rollout_log_probs"][0] == [-0.1, -0.2, -0.3, -0.4]

    def test_truncated_sampling_mask_passed_through(self):
        args = make_args(rewards_normalization=False, rollout_top_p=0.95, rollout_top_k=32)
        s = make_sample()
        s.rollout_sampling_mask_ids = [0, 7, 1, 8, 2, 9, 3, 10]
        s.rollout_sampling_mask_offsets = [0, 2, 4, 6, 8]
        out = convert_samples_to_train_data(
            args,
            [s],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["rollout_sampling_mask_ids"][0] == s.rollout_sampling_mask_ids
        assert out["rollout_sampling_mask_offsets"][0] == s.rollout_sampling_mask_offsets

    def test_truncated_sampling_requires_mask_for_every_sample(self):
        args = make_args(rewards_normalization=False, rollout_top_p=0.95, rollout_top_k=32)

        with pytest.raises(
            ValueError,
            match=r"requires sampling-mask data for every training sample.*sample_index=0",
        ):
            convert_samples_to_train_data(
                args,
                [make_sample()],
                metadata={},
                custom_convert_samples_to_train_data_func=None,
                custom_reward_post_process_func=None,
            )

    def test_removed_sample_still_requires_native_sampling_mask(self):
        args = make_args(rewards_normalization=False, rollout_top_p=0.95, rollout_top_k=32)
        removed = make_sample(index=9)
        removed.remove_sample = True

        with pytest.raises(ValueError, match=r"training sample.*sample_index=9"):
            convert_samples_to_train_data(
                args,
                [removed],
                metadata={},
                custom_convert_samples_to_train_data_func=None,
                custom_reward_post_process_func=None,
            )

    def test_optional_field_round_number_from_metadata(self):
        args = make_args(rewards_normalization=False)
        s = make_sample()
        s.metadata = {"round_number": 7}
        out = convert_samples_to_train_data(
            args,
            [s],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["round_number"][0] == 7

    def test_optional_field_raw_reward_overridden_from_metadata(self):
        args = make_args(rewards_normalization=False)
        s = make_sample(reward=1.0)
        s.metadata = {"raw_reward": 9.0}
        out = convert_samples_to_train_data(
            args,
            [s],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["raw_reward"][0] == 9.0

    def test_rollout_ids_default_to_sample_index(self):
        args = make_args(rewards_normalization=False)
        samples = [make_sample(index=i) for i in range(3)]
        out = convert_samples_to_train_data(
            args,
            samples,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["rollout_ids"] == [0, 1, 2]

    def test_rollout_ids_use_explicit_rollout_id_when_set(self):
        args = make_args(rewards_normalization=False)
        samples = [make_sample(index=i) for i in range(4)]
        # two compact siblings sharing one rollout execution
        samples[1].rollout_id = samples[2].rollout_id = 1
        out = convert_samples_to_train_data(
            args,
            samples,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["rollout_ids"] == [0, 1, 1, 3]

    def test_custom_convert_func_short_circuits(self):
        args = make_args()
        sentinel = {"foo": "bar"}
        out = convert_samples_to_train_data(
            args,
            [make_sample()],
            metadata={},
            custom_convert_samples_to_train_data_func=lambda a, s: sentinel,
            custom_reward_post_process_func=None,
        )
        assert out is sentinel

    def test_dynamic_global_batch_size_metadata_must_match(self):
        args = make_args(use_dynamic_global_batch_size=True, rewards_normalization=False)
        with pytest.raises(AssertionError):
            convert_samples_to_train_data(
                args,
                [make_sample()],
                metadata={},
                custom_convert_samples_to_train_data_func=None,
                custom_reward_post_process_func=None,
            )

    def test_dynamic_global_batch_size_metadata_passed_through(self):
        args = make_args(use_dynamic_global_batch_size=True, rewards_normalization=False)
        out = convert_samples_to_train_data(
            args,
            [make_sample()],
            metadata={"dynamic_global_batch_size": 16},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert out["dynamic_global_batch_size"] == 16


# ----------------------------- _post_process_rewards -----------------------------


class TestPostProcessRewards:
    def test_ppo_path_returns_raw_rewards_unchanged(self):
        args = make_args(advantage_estimator="ppo", rewards_normalization=True)
        samples = make_samples_grouped(2, 4, rewards=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        raw, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)
        assert raw == processed

    def test_grpo_normalizes_each_group_to_zero_mean(self):
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=4,
            rollout_batch_size=2,
        )
        samples = make_samples_grouped(2, 4, rewards=[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)
        # group means: 2.5 and 25.0 → centered values
        g1 = processed[:4]
        g2 = processed[4:]
        assert abs(sum(g1) / 4) < 1e-5
        assert abs(sum(g2) / 4) < 1e-5

    def test_grpo_with_std_normalization_unit_variance(self):
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=True,
            n_samples_per_prompt=4,
            rollout_batch_size=1,
        )
        samples = make_samples_grouped(1, 4, rewards=[1.0, 2.0, 3.0, 4.0])
        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)
        # Unit std with epsilon — torch.std uses N-1 by default; centered
        # values have std exactly 1.0 after dividing by their own std.
        import numpy as np

        assert abs(np.std(processed, ddof=1) - 1.0) < 1e-4

    def test_gspo_uses_grpo_normalization_path(self):
        args = make_args(
            advantage_estimator="gspo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=4,
            rollout_batch_size=1,
        )
        samples = make_samples_grouped(1, 4, rewards=[1.0, 2.0, 3.0, 4.0])
        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)
        assert abs(sum(processed) / 4) < 1e-5

    def test_reinforce_plus_plus_baseline_only_zero_mean_no_std(self):
        args = make_args(
            advantage_estimator="reinforce_plus_plus_baseline",
            rewards_normalization=True,
            grpo_std_normalization=True,  # should be IGNORED on this path
            n_samples_per_prompt=4,
            rollout_batch_size=1,
        )
        samples = make_samples_grouped(1, 4, rewards=[1.0, 2.0, 3.0, 4.0])
        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)
        # Mean is zero (centering happens) and std is the std of the centered
        # input (no normalization applied) — pin this exactly so a regression
        # that re-enabled the std division is caught.
        assert abs(sum(processed) / 4) < 1e-5
        import numpy as np

        expected_std = float(np.std([-1.5, -0.5, 0.5, 1.5]))
        assert abs(np.std(processed) - expected_std) < 1e-5

    def test_irregular_group_size_uses_explicit_group_index(self):
        """Explicit group identity keeps an irregularly sized group together."""
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=8,
            rollout_batch_size=2,
        )
        # rewards length 4 — does not match 8 * 2 = 16; trigger view branch
        samples = make_samples_grouped(1, 4, rewards=[2.0, 4.0, 6.0, 8.0])
        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)
        # mean is 5.0, after centering: -3, -1, 1, 3
        assert abs(sum(processed)) < 1e-5

    def test_grpo_normalizes_unique_rollouts_with_unequal_fanout(self):
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=2,
            rollout_batch_size=2,
        )
        samples = [
            make_sample(group_index=0, index=0, rollout_id=10, reward=0.0),
            make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
            make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
            make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
            make_sample(group_index=1, index=2, rollout_id=20, reward=2.0),
            make_sample(group_index=1, index=2, rollout_id=20, reward=2.0),
            make_sample(group_index=1, index=3, rollout_id=21, reward=4.0),
        ]

        raw, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        assert raw == [0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0]
        assert processed == pytest.approx([-0.5, 0.5, 0.5, 0.5, -1.0, -1.0, 1.0])

    def test_grpo_broadcasts_std_normalized_rollout_advantage(self):
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=True,
            n_samples_per_prompt=2,
            rollout_batch_size=1,
        )
        samples = [
            make_sample(group_index=0, index=0, rollout_id=10, reward=0.0),
            make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
            make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
        ]

        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        assert processed == pytest.approx([-(2**-0.5), 2**-0.5, 2**-0.5], abs=1e-5)

    def test_grpo_rejects_different_sibling_rewards(self):
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=2,
            rollout_batch_size=1,
        )
        samples = [
            make_sample(group_index=0, index=0, rollout_id=10, reward=0.0, loss_mask=[1, 1, 1, 1]),
            make_sample(group_index=0, index=1, rollout_id=11, reward=2.0, loss_mask=[1, 0, 0, 0]),
            make_sample(group_index=0, index=1, rollout_id=11, reward=6.0, loss_mask=[1, 1, 1, 0]),
        ]

        with pytest.raises(
            ValueError,
            match=r"all samples in rollout 11 must share one reward; rows \[1, 2\] have rewards \[2.0, 6.0\]",
        ):
            _post_process_rewards(args, samples, custom_reward_post_process_func=None)

    def test_grpo_shared_reward_ignores_final_training_mask(self):
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=2,
            rollout_batch_size=1,
        )
        samples = [
            make_sample(
                group_index=0,
                index=0,
                rollout_id=10,
                reward=2.0,
                response_length=8,
                remove_sample=True,
            ),
            make_sample(group_index=0, index=0, rollout_id=10, reward=2.0, response_length=4),
            make_sample(group_index=0, index=1, rollout_id=11, reward=6.0, loss_mask=[1, 1, 0, 0]),
        ]

        train_data = convert_samples_to_train_data(
            args,
            samples,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )

        assert train_data["raw_reward"] == [2.0, 2.0, 6.0]
        assert train_data["rewards"] == pytest.approx([-2.0, -2.0, 2.0])
        assert train_data["loss_masks"] == [[0] * 8, [1] * 4, [1, 1, 0, 0]]

    def test_grpo_shared_reward_uses_selected_reward_key(self):
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            reward_key="score",
            n_samples_per_prompt=2,
            rollout_batch_size=1,
        )
        samples = [
            make_sample(group_index=0, index=0, rollout_id=10, reward={"score": 2.0, "detail": "first"}),
            make_sample(group_index=0, index=0, rollout_id=10, reward={"score": 2.0, "detail": "second"}),
            make_sample(group_index=0, index=1, rollout_id=11, reward={"score": 6.0}),
        ]

        raw, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        assert raw == [2.0, 2.0, 6.0]
        assert processed == pytest.approx([-2.0, -2.0, 2.0])

    def test_prompt_group_sizes_override_reused_group_index(self):
        args = make_args(advantage_estimator="grpo", rewards_normalization=True)
        samples = [
            make_sample(group_index=0, rollout_id=10, reward=0.0),
            make_sample(group_index=0, rollout_id=11, reward=2.0),
            make_sample(group_index=0, rollout_id=20, reward=10.0),
            make_sample(group_index=0, rollout_id=21, reward=14.0),
        ]

        _, processed = _post_process_rewards(
            args,
            samples,
            custom_reward_post_process_func=None,
            prompt_group_sizes=[2, 2],
        )

        assert processed == pytest.approx([-1.0, 1.0, -2.0, 2.0])

    def test_noncontiguous_group_indices_share_reward_group(self):
        args = make_args(advantage_estimator="grpo", rewards_normalization=True)
        samples = [
            make_sample(group_index=0, rollout_id=10, reward=0.0),
            make_sample(group_index=1, rollout_id=20, reward=10.0),
            make_sample(group_index=0, rollout_id=11, reward=2.0),
            make_sample(group_index=1, rollout_id=21, reward=14.0),
        ]

        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        assert processed == pytest.approx([-1.0, -2.0, 1.0, 2.0])

    @pytest.mark.parametrize(
        ("rewards", "expected"),
        [
            ([0.0, 2.0, 10.0, 14.0], [-1.0, 1.0, -2.0, 2.0]),
            ([0.0, 2.0, 4.0], [-2.0, 0.0, 2.0]),
        ],
    )
    def test_missing_group_indices_use_legacy_boundaries(self, rewards, expected):
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            n_samples_per_prompt=2,
            rollout_batch_size=2,
        )
        samples = [
            make_sample(group_index=None, rollout_id=rollout_id, reward=reward)
            for rollout_id, reward in enumerate(rewards)
        ]

        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        assert processed == pytest.approx(expected)

    def test_rows_without_rollout_identity_stay_distinct(self):
        args = make_args(advantage_estimator="grpo", rewards_normalization=True)
        samples = [
            make_sample(group_index=0, index=None, rollout_id=None, reward=0.0),
            make_sample(group_index=0, index=None, rollout_id=None, reward=2.0),
        ]

        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        assert processed == pytest.approx([-1.0, 1.0])

    def test_empty_rewards_stay_empty(self):
        args = make_args(advantage_estimator="grpo", rewards_normalization=True)

        raw, processed = _post_process_rewards(args, [], custom_reward_post_process_func=None)

        assert raw == processed == []

    def test_custom_reward_post_process_short_circuits(self):
        args = make_args(advantage_estimator="grpo", rewards_normalization=True)
        sentinel = ([0.0], [1.0])
        raw, processed = _post_process_rewards(
            args, [make_sample()], custom_reward_post_process_func=lambda a, s: sentinel
        )
        assert (raw, processed) == sentinel


class TestPostProcessRewardsProperties:
    """Hypothesis-driven invariants for the GRPO normalization path.

    The point-tests above pin specific shapes; these guarantee the math holds
    across arbitrary group counts, group sizes, and reward distributions —
    catching bugs (e.g. wrong reshape axis, sign flip) that a fixed example
    might miss."""

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
        max_examples=40,
    )
    @given(
        n_groups=st.integers(min_value=1, max_value=6),
        group_size=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_grpo_zero_mean_invariant(self, n_groups, group_size, seed):
        """After GRPO centering, every group's mean must be ≈ 0 regardless of
        input distribution."""
        import random

        rng = random.Random(seed)
        rewards_list = [rng.uniform(-1000, 1000) for _ in range(n_groups * group_size)]
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=False,
            n_samples_per_prompt=group_size,
            rollout_batch_size=n_groups,
        )
        samples = make_samples_grouped(n_groups, group_size, rewards=rewards_list)
        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        assert len(processed) == n_groups * group_size
        for g in range(n_groups):
            chunk = processed[g * group_size : (g + 1) * group_size]
            # Tolerance scales with magnitude; 1e-3 of mean(|input|) covers fp32 drift.
            scale = max(abs(min(rewards_list)), abs(max(rewards_list)), 1.0)
            assert abs(sum(chunk) / group_size) < 1e-3 * scale, f"group {g} mean is not zero: chunk={chunk}"

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
        max_examples=40,
    )
    @given(
        n_groups=st.integers(min_value=1, max_value=6),
        group_size=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_grpo_unit_variance_invariant(self, n_groups, group_size, seed):
        """With grpo_std_normalization=True, each group's processed std → 1.

        We construct rewards whose per-group std is well above the 1e-6 epsilon
        floor; otherwise the epsilon-stabilized division produces ≈0 not ≈1."""
        import random

        import numpy as np

        rng = random.Random(seed)
        rewards_list: list[float] = []
        for _g in range(n_groups):
            base = rng.uniform(-100, 100)
            spread = rng.uniform(0.5, 50.0)  # >> epsilon
            for k in range(group_size):
                rewards_list.append(base + spread * (k - (group_size - 1) / 2))

        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=True,
            n_samples_per_prompt=group_size,
            rollout_batch_size=n_groups,
        )
        samples = make_samples_grouped(n_groups, group_size, rewards=rewards_list)
        _, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)

        for g in range(n_groups):
            chunk = processed[g * group_size : (g + 1) * group_size]
            std_val = float(np.std(chunk, ddof=1))
            assert abs(std_val - 1.0) < 1e-3, f"group {g} std={std_val}, chunk={chunk}"

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
        max_examples=30,
    )
    @given(
        n=st.integers(min_value=1, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_ppo_path_is_identity(self, n, seed):
        """PPO never normalizes rewards regardless of any other flag."""
        import random

        rng = random.Random(seed)
        rewards_list = [rng.uniform(-50, 50) for _ in range(n)]
        args = make_args(
            advantage_estimator="ppo",
            rewards_normalization=True,
            grpo_std_normalization=True,
            n_samples_per_prompt=n,
            rollout_batch_size=1,
        )
        samples = make_samples_grouped(1, n, rewards=rewards_list)
        raw, processed = _post_process_rewards(args, samples, custom_reward_post_process_func=None)
        assert raw == processed == rewards_list


# ----------------------------- split_train_data_by_dp -----------------------------


class TestSplitTrainDataByDp:
    @pytest.fixture(autouse=True)
    def _init_object_store(self):
        """split_train_data_by_dp puts through the object store singleton."""
        object_store.init_instance(make_args())

    def test_strided_partition_when_balance_data_off(self):
        args = make_args(balance_data=False)
        data = {
            "tokens": [[1, 2], [3, 4, 5], [6], [7, 8, 9, 10]],
            "response_lengths": [2, 3, 1, 4],
            "rewards": [0.1, 0.2, 0.3, 0.4],
            "truncated": [0, 0, 0, 1],
            "loss_masks": [[1, 1]] * 4,
            "sample_indices": [0, 1, 2, 3],
        }
        refs = split_train_data_by_dp(args, data, {"dp_size": 2})
        parts = [ray.get(r.inner) for r in refs]
        # stride: dp=0 takes [0, 2], dp=1 takes [1, 3]
        assert list(parts[0]["partition"]) == [0, 2]
        assert list(parts[1]["partition"]) == [1, 3]
        assert parts[0]["tokens"] == [[1, 2], [6]]

    def test_balanced_partition_when_balance_data_on(self):
        args = make_args(balance_data=True)
        # lengths chosen to force grouping: 1 + 4 vs 2 + 3 are balanced
        data = {
            "tokens": [[1], [2, 3], [4, 5, 6], [7, 8, 9, 10]],
            "response_lengths": [1, 2, 3, 4],
            "rewards": [0, 0, 0, 0],
            "truncated": [0, 0, 0, 0],
            "loss_masks": [[1] * n for n in (1, 2, 3, 4)],
            "sample_indices": [0, 1, 2, 3],
        }
        refs = split_train_data_by_dp(args, data, {"dp_size": 2})
        parts = [ray.get(r.inner) for r in refs]
        sizes = [len(p["tokens"]) for p in parts]
        assert max(sizes) - min(sizes) <= 1

    def test_optional_keys_propagated_when_present(self):
        args = make_args(balance_data=False)
        data = {
            "tokens": [[1], [2]],
            "response_lengths": [1, 1],
            "rewards": [0, 0],
            "truncated": [0, 0],
            "loss_masks": [[1], [1]],
            "sample_indices": [0, 1],
            "rollout_log_probs": [[-0.1], [-0.2]],
            "round_number": [1, 2],
        }
        refs = split_train_data_by_dp(args, data, {"dp_size": 2})
        parts = [ray.get(r.inner) for r in refs]
        assert "rollout_log_probs" in parts[0]
        assert "round_number" in parts[0]

    def test_shared_keys_not_split(self):
        """raw_reward, total_lengths, dynamic_global_batch_size are shared, not split."""
        args = make_args(balance_data=False)
        data = {
            "tokens": [[1], [2], [3], [4]],
            "response_lengths": [1, 1, 1, 1],
            "rewards": [0, 0, 0, 0],
            "truncated": [0, 0, 0, 0],
            "loss_masks": [[1]] * 4,
            "sample_indices": [0, 1, 2, 3],
            "raw_reward": [9.0, 8.0, 7.0, 6.0],
            "dynamic_global_batch_size": 4,
        }
        refs = split_train_data_by_dp(args, data, {"dp_size": 2})
        parts = [ray.get(r.inner) for r in refs]
        for p in parts:
            assert p["raw_reward"] == [9.0, 8.0, 7.0, 6.0]
            assert p["dynamic_global_batch_size"] == 4

    def test_partition_indices_form_a_partition(self):
        """All partition indices together cover [0, N) exactly once."""
        args = make_args(balance_data=False)
        n = 12
        data = {
            "tokens": [[i] for i in range(n)],
            "response_lengths": [1] * n,
            "rewards": [0] * n,
            "truncated": [0] * n,
            "loss_masks": [[1]] * n,
            "sample_indices": list(range(n)),
        }
        refs = split_train_data_by_dp(args, data, {"dp_size": 4})
        parts = [ray.get(r.inner) for r in refs]
        all_indices = sorted(i for p in parts for i in p["partition"])
        assert all_indices == list(range(n))


class TestSplitTrainDataRaw:
    def test_witness_ids_split_across_dp(self) -> None:
        tokens = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10, 11]]
        witness_ids = [
            torch.tensor([0, 0, 0]),
            torch.tensor([1, 1]),
            torch.tensor([2, 2, 2, 2]),
            torch.tensor([3, 3]),
        ]

        data = {
            "tokens": tokens,
            "seq_witness_ids": witness_ids,
            "response_lengths": [1, 1, 1, 1],
            "loss_masks": [[0, 0, 1], [0, 1], [0, 0, 0, 1], [0, 1]],
        }

        args = MagicMock()
        args.balance_data = False

        result = split_train_data_by_dp_raw(args, data, dp_size=2)

        assert len(result) == 2
        assert "seq_witness_ids" in result[0]
        assert "seq_witness_ids" in result[1]
        assert len(result[0]["seq_witness_ids"]) == 2
        assert len(result[1]["seq_witness_ids"]) == 2

    def test_indexer_topk_and_opd_reverse_kl_split_across_dp(self) -> None:
        """Keys from the rollout-side split (rollout_indexer_topk, opd_reverse_kl) partition per sample."""
        data = {
            "tokens": [[1, 2], [3, 4], [5, 6], [7, 8]],
            "response_lengths": [1, 1, 1, 1],
            "loss_masks": [[0, 1], [0, 1], [0, 1], [0, 1]],
            "rollout_indexer_topk": [torch.tensor([i]) for i in range(4)],
            "opd_reverse_kl": [[float(i)] for i in range(4)],
        }

        args = MagicMock()
        args.balance_data = False

        result = split_train_data_by_dp_raw(args, data, dp_size=2)

        assert len(result) == 2
        for part in result:
            assert len(part["rollout_indexer_topk"]) == 2
            assert len(part["opd_reverse_kl"]) == 2

    def test_no_witness_ids_when_absent(self) -> None:
        tokens = [[1, 2], [3, 4]]
        data = {
            "tokens": tokens,
            "response_lengths": [1, 1],
            "loss_masks": [[0, 1], [0, 1]],
        }

        args = MagicMock()
        args.balance_data = False

        result = split_train_data_by_dp_raw(args, data, dp_size=1)
        assert "seq_witness_ids" not in result[0]


FULL_SCHEDULE_CONFIG = {
    "dp_size": 2,
    "cp_size": 1,
    "vpp_size": 1,
    "microbatch_group_size_per_vp_stage": None,
}


def _make_split_data(n: int, *, lengths: list[int] | None = None, rollout_ids: list[int] | None = None) -> dict:
    lengths = lengths or [2] * n
    assert len(lengths) == n
    return {
        "tokens": [list(range(length)) for length in lengths],
        "response_lengths": [1] * n,
        "rewards": [0.0] * n,
        "truncated": [0] * n,
        "loss_masks": [[1] * length for length in lengths],
        "sample_indices": list(range(n)),
        "rollout_ids": rollout_ids if rollout_ids is not None else list(range(n)),
    }


class TestCanScheduleOnRolloutSide:
    def test_eligible_with_full_megatron_config(self):
        args = make_args(balance_data=False, micro_batch_size=1, use_dynamic_batch_size=False, multi_lora=False)
        assert can_schedule_on_rollout_side(args, _make_split_data(8), FULL_SCHEDULE_CONFIG)

    def test_rejects_partial_config(self):
        """fsdp / torchtitan advertise only dp_size; indep_dp advertises {}."""
        args = make_args(balance_data=False, micro_batch_size=1, multi_lora=False)
        assert not can_schedule_on_rollout_side(args, _make_split_data(8), {"dp_size": 2})
        assert not can_schedule_on_rollout_side(args, _make_split_data(8), {})
        assert not can_schedule_on_rollout_side(args, _make_split_data(8), None)

    def test_rejects_multi_lora(self):
        args = make_args(balance_data=False, micro_batch_size=1, multi_lora=True)
        assert not can_schedule_on_rollout_side(args, _make_split_data(8), FULL_SCHEDULE_CONFIG)

    def test_rejects_multimodal(self):
        args = make_args(balance_data=False, micro_batch_size=1, multi_lora=False)
        data = _make_split_data(8)
        data["multimodal_train_inputs"] = [None] * 8
        assert not can_schedule_on_rollout_side(args, data, FULL_SCHEDULE_CONFIG)

    def test_rejects_fewer_rollouts_than_gbs(self):
        args = make_args(balance_data=False, micro_batch_size=1, multi_lora=False)  # global_batch_size=8
        assert not can_schedule_on_rollout_side(args, _make_split_data(6), FULL_SCHEDULE_CONFIG)

    def test_accepts_trailing_partial_step(self):
        """Extra rollouts beyond a full step are fine — the schedule drops them."""
        args = make_args(balance_data=False, micro_batch_size=1, multi_lora=False)  # global_batch_size=8
        assert can_schedule_on_rollout_side(args, _make_split_data(10), FULL_SCHEDULE_CONFIG)

    def test_dynamic_gbs_overrides_args_gbs(self):
        args = make_args(balance_data=False, micro_batch_size=1, multi_lora=False)  # global_batch_size=8
        data = _make_split_data(6)
        data["dynamic_global_batch_size"] = 6
        assert can_schedule_on_rollout_side(args, data, FULL_SCHEDULE_CONFIG)


class TestSplitTrainDataByDpScheduled:
    def test_static_shards_cover_all_samples(self):
        """Static path: every sample lands in exactly one shard row, the schedule
        tiles each shard's rows exactly, and shard rows match their partition."""
        args = make_args(balance_data=False, micro_batch_size=2, use_dynamic_batch_size=False)
        data = _make_split_data(8)
        scheduled = split_train_data_by_dp_scheduled_raw(args, dict(data), train_parallel_config=FULL_SCHEDULE_CONFIG)

        assert len(scheduled) == 2
        seen = []
        for new in scheduled:
            partition = list(new["partition"])
            seen.extend(partition)
            assert new["tokens"] == [data["tokens"][j] for j in partition]
            # global_batch_size=8, dp=2, mbs=2 -> 4 mbs total, 2 per rank, 1 step
            assert new["num_microbatches"] == [2]
            assert new["num_rollouts"] == [8]
            flat = [i for mbs in new["micro_batch_indices"] for i in mbs]
            assert flat == list(range(len(new["tokens"])))
        assert sorted(seen) == list(range(8))

    def test_compact_rollout_gbs_counts_rollouts_not_samples(self):
        """gbs counts rollouts: 4 rollouts over 6 samples with gbs=2 -> 2 steps,
        and every sample is still covered exactly once across shards."""
        args = make_args(balance_data=False, micro_batch_size=1, use_dynamic_batch_size=True, max_tokens_per_gpu=8)
        rollout_ids = [0, 1, 1, 1, 2, 3]  # rollout 1 emits 3 samples
        data = _make_split_data(6, rollout_ids=rollout_ids)
        data["dynamic_global_batch_size"] = 2
        shards = split_train_data_by_dp_scheduled_raw(args, dict(data), train_parallel_config=FULL_SCHEDULE_CONFIG)

        assert shards[0]["num_rollouts"] == [2, 2]
        assert len(shards[0]["num_microbatches"]) == 2
        seen = sorted(j for shard in shards for j in shard["partition"])
        assert seen == list(range(6))

    def test_dynamic_schedule_respects_token_cap(self):
        args = make_args(
            balance_data=False,
            micro_batch_size=1,
            use_dynamic_batch_size=True,
            max_tokens_per_gpu=6,
        )
        lengths = [5, 1, 4, 2, 3, 3, 2, 4]
        data = _make_split_data(8, lengths=lengths)
        shards = split_train_data_by_dp_scheduled_raw(args, data, train_parallel_config=FULL_SCHEDULE_CONFIG)

        nmb = shards[0]["num_microbatches"]
        for shard in shards:
            assert shard["num_microbatches"] == nmb, "num_microbatches must be identical on every rank"
            assert len(shard["micro_batch_indices"]) == sum(nmb)
            partition = list(shard["partition"])
            for mbs in shard["micro_batch_indices"]:
                total = sum(lengths[partition[i]] for i in mbs)
                assert total <= 6 or len(mbs) == 1

    def test_dynamic_gbs_multi_step(self):
        """16 samples with dynamic_global_batch_size=8 -> 2 steps."""
        args = make_args(balance_data=False, micro_batch_size=2, use_dynamic_batch_size=False)
        data = _make_split_data(16)
        data["dynamic_global_batch_size"] = 8
        shards = split_train_data_by_dp_scheduled_raw(args, data, train_parallel_config=FULL_SCHEDULE_CONFIG)

        assert shards[0]["num_microbatches"] == [2, 2]
        assert shards[0]["dynamic_global_batch_size"] == 8
        assert shards[0]["num_rollouts"] == [8, 8]
