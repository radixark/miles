"""CPU tests for ragged-safe pass@k in miles.utils.metric_utils."""

from __future__ import annotations

import math

import numpy as np
import pytest

from miles.utils.metric_utils import _estimate_pass_at_k, compute_pass_rate


def _legacy_pass_rate(flat_rewards: list[float], group_size: int, num_groups: int) -> dict[str, float]:
    """The pre-fix implementation: assert exact multiple, reshape, average pass@k."""
    if group_size == 1:
        return {}
    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]
    assert len(flat_rewards) == num_groups * group_size
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)
    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)
        log_dict[f"pass@{k}"] = np.mean(_estimate_pass_at_k(num_samples, num_correct, k))
    return log_dict


class TestLegacyEquivalence:
    """group_ids=None default path is bit-identical to the legacy reshape for
    well-formed (exact-multiple) input."""

    @pytest.mark.parametrize(
        "rewards,group_size,num_groups",
        [
            ([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], 4, 2),
            ([1.0, 1.0, 0.0, 0.0], 2, 2),
            ([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], 4, 2),
            ([1.0] * 8, 8, 1),
        ],
    )
    def test_default_path_matches_legacy(self, rewards, group_size, num_groups):
        legacy = _legacy_pass_rate(rewards, group_size, num_groups)
        new = compute_pass_rate(rewards, group_size=group_size)
        assert new.keys() == legacy.keys()
        for k in legacy:
            assert new[k] == pytest.approx(legacy[k])

    def test_group_size_one_short_circuits(self):
        assert compute_pass_rate([1.0, 0.0, 1.0], group_size=1) == {}


class TestRaggedDoesNotCrash:
    def test_train_oversampled_flat_count_no_longer_asserts(self):
        """TRAIN reproduction: 9 rewards with group_size=4 is not an exact
        multiple (the legacy reshape raised). New code chunks into
        [1,0,1,1] (c=3,n=4), [0,0,1,0] (c=1,n=4), [1] (c=1,n=1)."""
        rewards = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
        with pytest.raises(AssertionError):
            # legacy asserted len == num_groups * group_size (9 != 2 * 4)
            _legacy_pass_rate(rewards, group_size=4, num_groups=2)

        out = compute_pass_rate(rewards, group_size=4)
        # pass@1 = mean(3/4, 1/4, 1) over all three groups.
        assert out["pass@1"] == pytest.approx(np.mean([3 / 4, 1 / 4, 1.0]))
        # pass@2 eligible groups (size>=2): [1,0,1,1]->1.0, [0,0,1,0]->0.5.
        assert out["pass@2"] == pytest.approx(np.mean([1.0, 0.5]))
        # pass@4 eligible groups (size>=4): both fully sampled -> 1.0 each.
        assert out["pass@4"] == pytest.approx(1.0)

    def test_eval_list_expanded_group_ids_bucket_correctly(self):
        """EVAL reproduction: list-expanded multi-turn samples give a ragged,
        non-multiple reward count. Bucketing by per-prompt group_ids must group
        them by prompt regardless of expansion."""
        # prompt 0 -> 3 samples (one generation list-expanded), prompt 1 -> 2 samples.
        rewards = [1.0, 0.0, 1.0, 0.0, 1.0]
        group_ids = [0, 0, 0, 1, 1]
        out = compute_pass_rate(rewards, group_size=2, group_ids=group_ids)
        assert set(out.keys()) == {"pass@1", "pass@2"}
        # prompt 0 -> [1,0,1] (c=2,n=3), prompt 1 -> [0,1] (c=1,n=2).
        # pass@1: P(>=1 correct in 1 draw) = c/n per group, averaged.
        assert out["pass@1"] == pytest.approx(np.mean([2 / 3, 1 / 2]))
        # pass@2: every group has >=1 correct, so each group's pass@2 is 1.0.
        assert out["pass@2"] == pytest.approx(1.0)

    def test_group_ids_length_mismatch_asserts(self):
        with pytest.raises(AssertionError):
            compute_pass_rate([1.0, 0.0, 1.0], group_size=2, group_ids=[0, 0])


class TestRungEligibility:
    def test_high_rung_dropped_when_all_groups_undersized(self):
        """Every group has 1 sample, so pass@2 (and above) has no eligible group
        and that rung is dropped; pass@1 survives."""
        rewards = [1.0, 0.0, 1.0]
        group_ids = [0, 1, 2]
        out = compute_pass_rate(rewards, group_size=4, group_ids=group_ids)
        assert "pass@1" in out
        assert "pass@2" not in out
        assert "pass@4" not in out

    def test_rung_averages_only_over_eligible_groups(self):
        """One full group (size 4) and one undersized group (size 1): pass@4 is
        computed only over the eligible size-4 group, not dragged by the
        singleton."""
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0]
        group_ids = [0, 0, 0, 0, 1]
        out = compute_pass_rate(rewards, group_size=4, group_ids=group_ids)
        assert out["pass@4"] == pytest.approx(1.0)
        assert out["pass@1"] == pytest.approx(0.5)
