from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

from miles.ray.rollout.metrics import _compute_metrics_from_samples, _compute_zero_std_metrics


class TestComputeZeroStdMetrics:
    def test_returns_empty_for_ppo_regardless_of_reward_distribution(self):
        args = make_args(advantage_estimator="ppo")
        out = _compute_zero_std_metrics(args, make_samples_grouped(2, 4, rewards=[1.0] * 8))
        assert out == {}

    def test_grpo_mixed_rewards_yield_zero_percentages_and_no_buckets(self):
        """Happy path: every group has reward variation → no group is zero-std →
        no bucket counts; the all_zero/all_one percentages are 0."""
        args = make_args(advantage_estimator="grpo", reward_key=None)
        samples = make_samples_grouped(2, 4, rewards=[0.0, 0.5, 1.0, 0.7, 0.2, 0.8, 0.3, 0.6])
        out = _compute_zero_std_metrics(args, samples)
        assert out == {"zero_std/all_zero_percentage": 0.0, "zero_std/all_one_percentage": 0.0}

    def test_grpo_zero_std_groups_produce_bucket_counts_and_percentages(self):
        """1 group all-1, 1 group all-0, 1 group mixed → bucket counts plus the
        all_zero/all_one percentages over total groups."""
        args = make_args(advantage_estimator="grpo", reward_key=None)
        samples = make_samples_grouped(3, 4, rewards=[1.0] * 4 + [0.0] * 4 + [0.0, 1.0, 0.0, 1.0])
        out = _compute_zero_std_metrics(args, samples)
        assert out["zero_std/count_1.0"] == 1
        assert out["zero_std/count_0.0"] == 1
        assert out["zero_std/all_zero_percentage"] == pytest.approx(1 / 3)
        assert out["zero_std/all_one_percentage"] == pytest.approx(1 / 3)

    def test_grpo_uniform_non_binary_reward_gets_its_own_bucket(self):
        """Every group zero-std at reward=0.5 → bucket count_0.5=2, but
        all_zero/all_one percentages stay 0 because they only count 0.0 and 1.0."""
        args = make_args(advantage_estimator="grpo", reward_key=None)
        samples = make_samples_grouped(2, 4, rewards=[0.5] * 8)
        out = _compute_zero_std_metrics(args, samples)
        assert out["zero_std/count_0.5"] == 2
        assert out["zero_std/all_zero_percentage"] == 0.0
        assert out["zero_std/all_one_percentage"] == 0.0

    def test_empty_samples_does_not_crash(self):
        args = make_args(advantage_estimator="grpo", reward_key=None)
        out = _compute_zero_std_metrics(args, [])
        # No groups → no all_zero/all_one keys (the function guards on total_groups>0).
        assert "zero_std/all_zero_percentage" not in out
        assert "zero_std/all_one_percentage" not in out


class TestTitoMismatchMetrics:
    def test_no_tito_metadata_emits_no_tito_keys(self):
        args = make_args(advantage_estimator="ppo", ci_test=False, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        out = _compute_metrics_from_samples(args, samples)
        assert "tito_session_mismatch_rate" not in out

    def test_clean_tito_metadata_yields_zero_rates_per_mismatch_type(self):
        args = make_args(advantage_estimator="ppo", ci_test=False, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        for s in samples:
            s.metadata = {"tito_session_mismatch": []}
        out = _compute_metrics_from_samples(args, samples)
        assert out["tito_session_mismatch_rate"] == 0.0
        for mtype in ("special_token_count", "special_token_type", "non_assistant_text", "assistant_text"):
            assert out[f"tito_session_mismatch_rate/{mtype}"] == 0.0

    def test_strict_mismatch_raises_under_ci_test(self):
        """Under ci_test=True, a non-zero rate on the strict mismatch types
        (special_token_count / special_token_type / non_assistant_text) must
        hard-fail — these signal a TITO algorithm or chat-template bug."""
        args = make_args(advantage_estimator="ppo", ci_test=True, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        samples[0].metadata = {"tito_session_mismatch": [{"type": "special_token_count"}]}
        for s in samples[1:]:
            s.metadata = {"tito_session_mismatch": []}
        with pytest.raises(AssertionError, match="special_token_count"):
            _compute_metrics_from_samples(args, samples)

    def test_assistant_text_mismatch_does_not_raise_under_ci_test(self):
        """assistant_text mismatch is non-critical (tokens inherited from the
        pretokenized prefix) — even under ci_test, must not raise."""
        args = make_args(advantage_estimator="ppo", ci_test=True, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        samples[0].metadata = {"tito_session_mismatch": [{"type": "assistant_text"}]}
        for s in samples[1:]:
            s.metadata = {"tito_session_mismatch": []}
        out = _compute_metrics_from_samples(args, samples)
        assert out["tito_session_mismatch_rate/assistant_text"] > 0
