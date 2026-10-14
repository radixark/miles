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

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.metrics import log_utils
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


class TestSplitRolloutsLoseNothing:
    """A rollout split into several samples trains the same loss as the
    unsplit rollout once the legacy DP split stamps ``num_rollouts``
    (``train_data_conversion._rollouts_per_step``): the per-sample sums over
    the whole rollout's mask total add up to one token-weighted mean per
    rollout, and the step divides by rollouts, not samples."""

    def _masks(self, lens):
        return [torch.ones(n, dtype=torch.int) for n in lens]

    def test_two_rollouts_in_five_samples_match_the_unsplit_loss(self):
        from types import SimpleNamespace

        from miles.ray.rollout.train_data_conversion import _rollouts_per_step

        # rollout A: 6 response tokens, cut into samples of 1, 2 and 3; rollout B: 4 tokens, cut 1 + 3
        per_token = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0])
        unsplit_lens = [6, 4]
        unsplit = get_sum_of_sample_mean(list(unsplit_lens), list(unsplit_lens), self._masks(unsplit_lens))
        expected = unsplit(per_token).item() / 2  # (21/6 + 100/4) / 2 rollouts

        split_lens = [1, 2, 3, 1, 3]
        rollout_ids = [7, 7, 7, 9, 9]
        rollout_mask_sums = torch.tensor([6.0, 6.0, 6.0, 4.0, 4.0])
        split = get_sum_of_sample_mean(
            list(split_lens), list(split_lens), self._masks(split_lens), denominators=rollout_mask_sums
        )
        data = {"tokens": [[0]] * 5, "rollout_ids": rollout_ids, "dynamic_global_batch_size": 5}
        (num_rollouts,) = _rollouts_per_step(SimpleNamespace(global_batch_size=5), data)
        assert num_rollouts == 2
        assert math.isclose(split(per_token).item() / num_rollouts, expected, rel_tol=1e-6)
        # the fallback divisor (the sample count) scales the same loss by 2/5
        assert math.isclose(split(per_token).item() / 5, expected * 2 / 5, rel_tol=1e-6)


class TestTheTrainerDividesByTheStampedRolloutCount:
    """End to end through the trainer's own code: the legacy DP split stamps
    ``num_rollouts`` on the shard, ``get_num_rollouts`` reads it, and
    ``loss_function`` (objective.py) divides the summed loss by it."""

    def test_objective_divides_by_rollouts_from_the_legacy_split(self, monkeypatch):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from tests.fast.backends.training_utils.loss.loss_test_utils import make_args as make_loss_args
        from tests.fast.backends.training_utils.loss.loss_test_utils import make_batch, make_inputs

        from miles.backends.training_utils.data.rollout import get_num_rollouts
        from miles.backends.training_utils.loss import objective as loss_module
        from miles.ray.rollout.train_data_conversion import split_train_data_by_dp_raw

        # 2 rollouts in 5 samples, one optimizer step, dp 1: the split stamps [2]
        split_args = SimpleNamespace(global_batch_size=5, balance_data=False)
        data = {
            "tokens": [[1, 2]] * 5,
            "rollout_ids": [7, 7, 7, 9, 9],
            "dynamic_global_batch_size": 5,
            "response_lengths": [1] * 5,
        }
        (shard,) = split_train_data_by_dp_raw(split_args, data, dp_size=1)
        assert shard["num_rollouts"] == [2]
        (num_rollouts,) = get_num_rollouts(split_args, shard, num_steps=1)
        assert num_rollouts == 2
        # the same batch without the stamp falls back to the sample count
        del shard["num_rollouts"]
        assert get_num_rollouts(split_args, shard, num_steps=1) == [5]

        # loss_function with a loss that sums to 3.0: divided by 2 rollouts, not 5 samples
        args = make_loss_args(use_dynamic_global_batch_size=True, calculate_per_token_loss=False)
        inputs = make_inputs(seed=5, batch_size=1, prompt_lens=[3], response_lens=[2], vocab_size=8, args=args)
        batch = make_batch(inputs, "policy_loss")
        batch["dynamic_global_batch_size"] = 5
        logits = inputs["policy_logits"]
        monkeypatch.setattr(
            loss_module,
            "get_loss_function",
            lambda _args, _loss_fn=None: Mock(return_value=(torch.tensor(3.0), {"loss": torch.tensor(0.0)})),
        )
        stamped, _, _ = loss_module.loss_function(args, batch, 1, logits, num_rollouts=num_rollouts)
        fallback, _, _ = loss_module.loss_function(args, batch, 1, logits, num_rollouts=None)
        assert math.isclose(stamped.item(), 3.0 / 2, rel_tol=1e-6)
        assert math.isclose(fallback.item(), 3.0 / 5, rel_tol=1e-6)
