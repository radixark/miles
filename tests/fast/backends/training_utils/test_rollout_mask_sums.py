"""CPU tests for per-rollout loss denominators (``rollout_mask_sums``).

Covers:
- the rollout-side precomputation in ``convert_samples_to_train_data``;
- ``get_sum_of_sample_mean(denominators=...)`` — legacy per-sample mean when
  ``None``, per-rollout token-weighted mean with precomputed denominators, and
  the strict no-op equivalence between the two when 1 rollout = 1 sample;
- ``aggregate_train_losses(num_rollouts=...)`` per-rollout-mean reduction.
"""

from __future__ import annotations

import math

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.backends.training_utils import log_utils
from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data


@pytest.fixture(autouse=True)
def _parallel_state():
    make_parallel_state()


def _convert(samples):
    return convert_samples_to_train_data(
        make_args(rewards_normalization=False),
        samples,
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )


class TestRolloutMaskSumsPrecompute:
    def test_defaults_to_per_sample_mask_sums(self):
        samples = [make_sample(index=i, response_length=n) for i, n in enumerate((2, 3, 4))]
        out = _convert(samples)
        assert out["rollout_mask_sums"] == [2, 3, 4]

    def test_compact_siblings_share_whole_rollout_total(self):
        samples = [make_sample(index=i, response_length=n) for i, n in enumerate((2, 3, 4))]
        # samples 0 and 1 come from the same rollout execution
        samples[0].rollout_id = samples[1].rollout_id = 0
        out = _convert(samples)
        assert out["rollout_mask_sums"] == [5, 5, 4]

    def test_masked_out_tokens_excluded(self):
        s = make_sample(index=0, response_length=4)
        s.loss_mask = [1, 0, 1, 0]
        out = _convert([s])
        assert out["rollout_mask_sums"] == [2]


class TestDenominators:
    def _masks(self, lens):
        return [torch.ones(n, dtype=torch.int) for n in lens]

    def test_none_matches_legacy_per_sample_mean(self):
        lens = [2, 3]
        x = torch.tensor([1.0, 3.0, 2.0, 2.0, 5.0])
        som = get_sum_of_sample_mean(list(lens), list(lens), self._masks(lens))
        # sample means: (1+3)/2 = 2, (2+2+5)/3 = 3
        assert math.isclose(som(x).item(), 5.0)

    def test_rollout_denoms_give_token_weighted_rollout_mean(self):
        lens = [2, 3]
        x = torch.tensor([1.0, 3.0, 2.0, 2.0, 5.0])
        # both samples belong to one rollout with 5 mask tokens total
        denoms = torch.tensor([5.0, 5.0])
        som = get_sum_of_sample_mean(list(lens), list(lens), self._masks(lens), denominators=denoms)
        # (1+3)/5 + (2+2+5)/5 = 13/5 — one token-weighted mean for the rollout
        assert math.isclose(som(x).item(), 13.0 / 5, rel_tol=1e-6)

    def test_per_sample_denominators_equal_legacy(self):
        """1 rollout = 1 sample: rollout_mask_sums equals each mask's own sum -> strict no-op."""
        lens = [2, 3, 4]
        x = torch.randn(sum(lens))
        masks = self._masks(lens)
        legacy = get_sum_of_sample_mean(list(lens), list(lens), masks)
        denoms = torch.tensor([float(n) for n in lens])
        new = get_sum_of_sample_mean(list(lens), list(lens), masks, denominators=denoms)
        assert torch.allclose(legacy(x), new(x))


class TestAggregateTrainLossesNumRollouts:
    @pytest.fixture(autouse=True)
    def _no_allreduce(self, monkeypatch):
        monkeypatch.setattr(log_utils.MultiPGUtil, "all_reduce", staticmethod(lambda *a, **k: None))

    def _mb(self, num_samples, values):
        return {"keys": list(values.keys()), "values": torch.tensor([num_samples, *values.values()])}

    def test_num_rollouts_overrides_sample_count(self):
        losses = [self._mb(2, {"loss": 2.0}), self._mb(2, {"loss": 6.0})]
        out = log_utils.aggregate_train_losses(losses, num_rollouts=4)
        assert math.isclose(out["loss"], 2.0)

    def test_legacy_reduction_without_divisor(self):
        losses = [self._mb(2, {"loss": 2.0}), self._mb(2, {"loss": 6.0})]
        out = log_utils.aggregate_train_losses(losses)
        assert math.isclose(out["loss"], 2.0)
