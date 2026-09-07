from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sample, make_samples_grouped

from miles.ray.rollout.metrics import (
    _compute_episode_response_length_metrics,
    _compute_metrics_from_samples,
    _compute_passrate_from_samples,
    _compute_training_sample_metrics,
    _compute_zero_std_metrics,
    log_eval_rollout_data,
    log_rollout_data,
)
from miles.utils.types import AdapterRef, Sample, WeightVersionSpan, WeightVersionsPerCall


class TestEpisodeResponseLengthMetrics:
    def test_compacted_siblings_are_summed_before_computing_statistics(self):
        samples = [
            make_sample(group_index=0, index=0, rollout_id=10, response_length=5, loss_mask=[1, 1, 0, 0, 0]),
            make_sample(group_index=0, index=0, rollout_id=10, response_length=7, loss_mask=[1, 1, 1, 0, 0, 0, 0]),
            make_sample(group_index=0, index=1, rollout_id=11, response_length=4, loss_mask=[1, 1, 1, 1]),
            make_sample(group_index=1, index=2, rollout_id=10, response_length=8, loss_mask=[1, 1, 1, 1, 1, 1, 0, 0]),
        ]

        out = _compute_episode_response_length_metrics(samples)

        assert out == {
            "episode_response_length/mean": pytest.approx(5.0),
            "episode_response_length/median": pytest.approx(5.0),
            "episode_response_length/max": pytest.approx(6.0),
            "episode_response_length/min": pytest.approx(4.0),
            "episode_total_response_length/mean": pytest.approx(8.0),
        }

    def test_single_sample_rollouts_match_sample_level_statistics(self):
        samples = [
            make_sample(index=0, rollout_id=10, response_length=5, loss_mask=[1, 1, 0, 0, 0]),
            make_sample(index=1, rollout_id=11, response_length=7, loss_mask=[1, 1, 1, 0, 0, 0, 0]),
            make_sample(index=2, rollout_id=12, response_length=4, loss_mask=[1, 1, 1, 1]),
        ]

        out = _compute_metrics_from_samples(make_args(advantage_estimator="ppo"), samples)

        for statistic in ("mean", "median", "max", "min"):
            assert out[f"episode_response_length/{statistic}"] == out[f"response_len/{statistic}"]

    def test_empty_samples_emit_no_episode_length_metrics(self):
        assert _compute_episode_response_length_metrics([]) == {}

    def test_total_length_counts_masked_and_unmasked_tokens_in_every_sample(self):
        samples = [
            make_sample(index=0, rollout_id=10, response_length=5, loss_mask=[1, 1, 0, 0, 0]),
            make_sample(index=0, rollout_id=10, response_length=7, loss_mask=[1, 1, 1, 0, 0, 0, 0]),
            make_sample(index=1, rollout_id=11, response_length=4, loss_mask=[1, 1, 1, 1]),
        ]

        out = _compute_episode_response_length_metrics(samples)

        assert out["episode_response_length/mean"] == pytest.approx(4.5)
        assert out["episode_total_response_length/mean"] == pytest.approx(8.0)

    def test_multi_lora_samples_emit_no_episode_length_metrics(self):
        samples = [
            make_sample(index=0, rollout_id=10, adapter=AdapterRef(name="adapter-a", slot=0)),
            make_sample(index=0, rollout_id=10, adapter=AdapterRef(name="adapter-b", slot=1)),
        ]

        assert _compute_episode_response_length_metrics(samples) == {}
        out = _compute_metrics_from_samples(make_args(advantage_estimator="ppo"), samples)
        assert not any(key.startswith("episode_response_length/") for key in out)
        assert "episode_total_response_length/mean" not in out
        assert out["response_len/mean"] == pytest.approx(4.0)

    def test_removed_sample_has_zero_effective_length_but_keeps_total_length(self):
        sample = make_sample(
            index=0,
            rollout_id=10,
            response_length=5,
            loss_mask=[1, 1, 1, 1, 1],
            remove_sample=True,
        )

        out = _compute_episode_response_length_metrics([sample])

        assert out["episode_response_length/mean"] == pytest.approx(0.0)
        assert out["episode_total_response_length/mean"] == pytest.approx(5.0)


class TestTrainingSampleMetrics:
    def test_compacted_rollouts_are_counted_as_samples_but_rewarded_as_episodes(self):
        args = make_args(reward_key=None)
        samples = [
            make_sample(index=0, rollout_id=10, reward=1.0),
            make_sample(index=0, rollout_id=10, reward=1.0),
            make_sample(index=0, rollout_id=10, reward=1.0),
            make_sample(index=1, rollout_id=11, reward=0.0),
        ]

        out = _compute_training_sample_metrics(args, samples)

        assert out["num_training_samples"] == 4
        assert out["episode_raw_reward"] == pytest.approx(0.5)

    def test_different_sibling_rewards_are_averaged_within_rollout_first(self):
        args = make_args(reward_key=None)
        samples = [
            make_sample(index=0, rollout_id=10, reward=0.0),
            make_sample(index=0, rollout_id=10, reward=1.0),
            make_sample(index=1, rollout_id=11, reward=1.0),
        ]

        out = _compute_training_sample_metrics(args, samples)

        assert out["episode_raw_reward"] == pytest.approx(0.75)

    def test_rollout_ids_are_scoped_by_prompt_group(self):
        args = make_args(reward_key=None)
        samples = [
            make_sample(group_index=0, index=0, rollout_id=10, reward=1.0),
            make_sample(group_index=0, index=0, rollout_id=10, reward=1.0),
            make_sample(group_index=1, index=1, rollout_id=10, reward=0.0),
        ]

        out = _compute_training_sample_metrics(args, samples)

        assert out["episode_raw_reward"] == pytest.approx(0.5)

    def test_rollout_ids_are_scoped_by_adapter(self):
        args = make_args(reward_key=None)
        adapter_a = AdapterRef(name="adapter-a", slot=0)
        adapter_b = AdapterRef(name="adapter-b", slot=1)
        samples = [
            make_sample(group_index=0, rollout_id=10, adapter=adapter_a, reward=1.0),
            make_sample(group_index=0, rollout_id=10, adapter=adapter_a, reward=1.0),
            make_sample(group_index=0, rollout_id=10, adapter=adapter_a, reward=1.0),
            make_sample(group_index=0, rollout_id=10, adapter=adapter_b, reward=0.0),
        ]

        out = _compute_training_sample_metrics(args, samples)

        assert out["episode_raw_reward"] == pytest.approx(0.5)

    def test_metadata_raw_reward_and_fallback_identities(self):
        args = make_args(reward_key=None)
        samples = [
            make_sample(index=5, reward=0.0),
            make_sample(index=None, reward=0.0),
        ]
        samples[0].metadata = {"raw_reward": 1.0}
        samples[1].metadata = {"raw_reward": 0.0}

        out = _compute_training_sample_metrics(args, samples)

        assert out == {"num_training_samples": 2, "episode_raw_reward": pytest.approx(0.5)}

    def test_empty_samples(self):
        assert _compute_training_sample_metrics(make_args(), []) == {
            "num_training_samples": 0,
            "episode_raw_reward": 0.0,
        }


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
        assert not any(key.startswith("tito_session_mismatch_rate") for key in out)

    @pytest.mark.parametrize(
        ("configured_version", "metric_version"),
        [(True, "v1"), ("v1", "v1"), ("v2", "v2")],
    )
    def test_clean_tito_metadata_yields_zero_rates_per_mismatch_type(self, configured_version, metric_version):
        args = make_args(
            advantage_estimator="ppo",
            ci_test=False,
            log_passrate=False,
            use_session_server=configured_version,
        )
        samples = make_samples_grouped(1, 4)
        for s in samples:
            s.metadata = {"tito_session_mismatch": []}
        out = _compute_metrics_from_samples(args, samples)
        metric_prefix = f"tito_session_mismatch_rate/{metric_version}"
        tito_keys = {
            metric_prefix,
            f"{metric_prefix}/special_token_count",
            f"{metric_prefix}/special_token_type",
            f"{metric_prefix}/non_assistant_text",
            f"{metric_prefix}/assistant_text",
        }
        assert {key for key in out if key.startswith("tito_session_mismatch_rate")} == tito_keys
        assert all(out[key] == 0.0 for key in tito_keys)

    def test_strict_mismatch_raises_under_ci_test(self):
        """Under ci_test=True, a non-zero rate on the strict mismatch types
        (special_token_count / special_token_type / non_assistant_text) must
        hard-fail — these signal a TITO algorithm or chat-template bug."""
        args = make_args(
            advantage_estimator="ppo",
            ci_test=True,
            log_passrate=False,
            use_session_server="v1",
        )
        samples = make_samples_grouped(1, 4)
        samples[0].metadata = {"tito_session_mismatch": [{"type": "special_token_count"}]}
        for s in samples[1:]:
            s.metadata = {"tito_session_mismatch": []}
        with pytest.raises(
            AssertionError,
            match=r"tito_session_mismatch_rate/v1/special_token_count=0\.2500",
        ):
            _compute_metrics_from_samples(args, samples)

    def test_assistant_text_mismatch_does_not_raise_under_ci_test(self):
        """assistant_text mismatch is non-critical (tokens inherited from the
        pretokenized prefix) — even under ci_test, must not raise."""
        args = make_args(
            advantage_estimator="ppo",
            ci_test=True,
            log_passrate=False,
            use_session_server="v2",
        )
        samples = make_samples_grouped(1, 4)
        samples[0].metadata = {"tito_session_mismatch": [{"type": "assistant_text"}]}
        for s in samples[1:]:
            s.metadata = {"tito_session_mismatch": []}
        out = _compute_metrics_from_samples(args, samples)
        assert out["tito_session_mismatch_rate/v2/assistant_text"] == 0.25
        assert "tito_session_mismatch_rate/assistant_text" not in out

    def test_tito_metadata_requires_session_server_version(self):
        args = make_args(advantage_estimator="ppo", ci_test=False, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        for sample in samples:
            sample.metadata = {"tito_session_mismatch": []}

        with pytest.raises(AssertionError, match="session server v1 or v2"):
            _compute_metrics_from_samples(args, samples)

    def test_rollout_log_fans_out_versioned_tito_keys(self, monkeypatch):
        args = make_args(
            advantage_estimator="ppo",
            ci_test=False,
            log_passrate=False,
            use_session_server="v2",
        )
        samples = make_samples_grouped(1, 4)
        samples[0].metadata = {"tito_session_mismatch": [{"type": "assistant_text"}]}
        for sample in samples[1:]:
            sample.metadata = {"tito_session_mismatch": []}
        logged = {}
        monkeypatch.setattr(
            "miles.ray.rollout.metrics.tracking.log",
            lambda _args, metrics, **_kwargs: logged.update(metrics),
        )

        log_rollout_data(0, args, samples, None, 1.0)

        assert logged["rollout/num_training_samples"] == 4
        assert logged["rollout/episode_raw_reward"] == pytest.approx(1.5)
        assert logged["rollout/episode_response_length/mean"] == pytest.approx(4.0)
        assert logged["rollout/episode_total_response_length/mean"] == pytest.approx(4.0)
        assert logged["rollout/tito_session_mismatch_rate/v2/assistant_text"] == 0.25
        assert "rollout/tito_session_mismatch_rate/assistant_text" not in logged


class TestComputePassrateFromSamples:
    def test_returns_empty_when_group_size_is_one(self):
        args = make_args(n_samples_per_prompt=1)
        samples = make_samples_grouped(4, 1, rewards=[1.0, 0.0, 1.0, 0.0])

        assert _compute_passrate_from_samples(args, samples) == {}

    @pytest.mark.parametrize("reward, expected", [(1.0, 1.0), (0.0, 0.0)])
    def test_uniform_rewards(self, reward, expected):
        args = make_args(n_samples_per_prompt=4, reward_key=None)
        samples = make_samples_grouped(2, 4, rewards=[reward] * 8)

        out = _compute_passrate_from_samples(args, samples)

        assert out == {
            "pass@1": pytest.approx(expected),
            "pass@2": pytest.approx(expected),
            "pass@4": pytest.approx(expected),
        }

    def test_mixed_rewards_pass_at_k_increases_with_k(self):
        args = make_args(n_samples_per_prompt=4, reward_key=None)
        rewards = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        samples = make_samples_grouped(2, 4, rewards=rewards)

        out = _compute_passrate_from_samples(args, samples)

        assert out["pass@1"] < out["pass@2"] < out["pass@4"]

    def test_excludes_incomplete_groups(self):
        args = make_args(n_samples_per_prompt=4, reward_key=None)
        samples = make_samples_grouped(2, 4, rewards=[1.0] * 4 + [0.0] * 4)
        samples.pop()

        out = _compute_passrate_from_samples(args, samples)

        assert out == {
            "pass@1": pytest.approx(1.0),
            "pass@2": pytest.approx(1.0),
            "pass@4": pytest.approx(1.0),
        }


class TestWeightVersionMetrics:
    def test_all_numeric_inputs_keep_the_existing_decode_metric_keys_and_values(self):
        """All-numeric decode spans retain the complete pre-existing metric surface."""
        samples = [
            _make_versioned_sample(["4"], index=0),
            _make_versioned_sample(["5", "6"], index=1),
        ]

        out = _compute_metrics_from_samples(make_args(), samples)

        assert {key: value for key, value in out.items() if key.startswith("weight_version/")} == {
            "weight_version/mean": 4.5,
            "weight_version/median": 4.5,
            "weight_version/max": 5,
            "weight_version/min": 4,
            "weight_version/mixed_version_ratio": 0.5,
        }

    def test_a_placeholder_and_numeric_decode_mix_is_mixed(self):
        """Any two distinct version labels count as mixed, placeholders included."""
        sample = _make_versioned_sample(["default", "4"], index=0)

        out = _compute_metrics_from_samples(make_args(), [sample])

        assert out["weight_version/min"] == 4
        assert out["weight_version/mixed_version_ratio"] == 1.0

    def test_the_mixed_ratio_denominator_is_the_sample_count_not_the_call_count(self):
        """Mixedness is a per-sample verdict, so one sample spanning three calls still weighs one sample."""
        samples = [
            _make_versioned_sample(["4", "5", "6"], index=0),
            _make_versioned_sample(["7"], index=1),
        ]

        out = _compute_metrics_from_samples(make_args(), samples)

        assert out["weight_version/mixed_version_ratio"] == 0.5

    def test_no_version_metrics_when_no_span_carries_a_numeric_version(self):
        """Placeholder-only labels leave nothing to average, so the whole series stays absent."""
        samples = [_make_versioned_sample(["default", "mock-v0"], index=0)]

        out = _compute_metrics_from_samples(make_args(), samples)

        assert not any(key.startswith("weight_version/") for key in out)

    def test_a_call_spanning_no_update_is_not_mixed(self):
        """Two calls that both saw the same version must not count as mixed."""
        samples = [_make_versioned_sample(["7", "7"], index=0)]

        out = _compute_metrics_from_samples(make_args(), samples)

        assert out["weight_version/mixed_version_ratio"] == 0.0

    def test_a_single_call_spanning_two_versions_counts_as_mixed(self):
        """A weight update landing mid-call makes that single call mixed, just like two calls seeing two versions."""
        sample = make_sample(index=0, group_index=0)
        sample.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan("3", 0, 2), WeightVersionSpan("4", 2, 4)])
        ]

        out = _compute_metrics_from_samples(make_args(), [sample])

        assert out["weight_version/mixed_version_ratio"] == 1.0

    def test_no_version_metrics_when_nothing_was_stamped(self):
        """SFT-style batches carry no versions and must not synthesise the series."""
        out = _compute_metrics_from_samples(make_args(), [make_sample(index=0, group_index=0)])

        assert not any(key.startswith("weight_version/") for key in out)


def _make_versioned_sample(versions: list[str], *, index: int) -> Sample:
    sample = make_sample(index=index, group_index=0)
    sample.weight_versions = [
        WeightVersionsPerCall(spans=[WeightVersionSpan(version, i, i + 1)]) for i, version in enumerate(versions)
    ]
    return sample


def _prefilled_sample(*turns: tuple[list[tuple[str, int]], list[str]], index: int = 0) -> Sample:
    calls: list[WeightVersionsPerCall] = []
    generated_end = 0
    for prefill, decode in turns:
        prefill_spans: list[WeightVersionSpan] = []
        output_start = 0
        for version, num_tokens in prefill:
            prefill_spans.append(WeightVersionSpan(version, output_start, output_start + num_tokens))
            output_start += num_tokens
        assert output_start >= generated_end, "each turn's prompt must cover everything generated so far"
        spans = [
            WeightVersionSpan(version, output_start + i, output_start + i + 1) for i, version in enumerate(decode)
        ]
        calls.append(WeightVersionsPerCall(spans=spans, prefill_spans=prefill_spans, output_start=output_start))
        generated_end = output_start + len(decode)

    sample = make_sample(
        index=index,
        group_index=0,
        response_length=generated_end - calls[0].output_start,
        tokens=list(range(generated_end)),
        weight_versions=calls,
    )
    sample.validate()
    return sample


class TestPrefillWeightVersionMetrics:
    def test_stale_token_ratio_counts_prompt_tokens_older_than_the_decode_version(self):
        """Prompt tokens whose KV predates the call's decode version are stale; the ratio is over all prompt tokens."""
        first = _prefilled_sample(([("1", 3), ("5", 5)], ["5"]), index=0)
        second = _prefilled_sample(([("5", 2)], ["5"]), index=1)

        out = _compute_metrics_from_samples(make_args(), [first, second])

        assert out["weight_version/prefill_stale_token_ratio"] == pytest.approx(3 / 10)

    def test_lag_is_the_decode_version_minus_the_oldest_prompt_version_per_call(self):
        """Each call contributes its own lag, and the batch reports the worst one."""
        first = _prefilled_sample(([("1", 3), ("5", 5)], ["5"]), ([("4", 4), ("6", 5)], ["6"]), index=0)
        second = _prefilled_sample(([("6", 2)], ["6"]), index=1)

        out = _compute_metrics_from_samples(make_args(), [first, second])

        assert out["weight_version/prefill_lag_max"] == 4

    def test_mixed_ratio_is_the_share_of_samples_with_a_multi_version_prompt(self):
        """Prefill mixedness and oldest-version statistics span every call in a sample."""
        mixed = _prefilled_sample(([("5", 2)], ["5"]), ([("1", 4)], ["5"]), index=0)
        uniform = _prefilled_sample(([("5", 2)], ["5"]), ([("5", 4)], ["5"]), index=1)

        out = _compute_metrics_from_samples(make_args(), [mixed, uniform])

        assert out["weight_version/prefill_min"] == 1
        assert out["weight_version/prefill_max"] == 5
        assert out["weight_version/prefill_mean"] == 3
        assert out["weight_version/prefill_median"] == 3
        assert out["weight_version/prefill_mixed_version_ratio"] == 0.5

    def test_a_call_spanning_an_update_measures_lag_against_its_newest_decode_version(self):
        """Under in_place a call can decode across an update; its current version is the newest one it saw."""
        sample = _prefilled_sample(([("3", 4)], ["4", "5"]))

        out = _compute_metrics_from_samples(make_args(), [sample])

        assert out["weight_version/prefill_lag_max"] == 2
        assert out["weight_version/prefill_stale_token_ratio"] == 1.0

    def test_stale_and_lag_only_cover_comparable_calls(self):
        """Calls with a placeholder version anywhere stay out of the stale and lag figures entirely."""
        sample = _prefilled_sample(
            ([("default", 2), ("4", 2)], ["5"]),
            ([("2", 5)], ["mock-v0"]),
            ([("3", 6)], ["5"]),
        )

        out = _compute_metrics_from_samples(make_args(), [sample])

        assert out["weight_version/prefill_lag_max"] == 2
        assert out["weight_version/prefill_stale_token_ratio"] == 1.0

    def test_a_prompt_spanning_two_labels_is_mixed_without_being_comparable(self):
        """Mixedness reads the raw labels, so it survives calls that stale and lag must skip."""
        mixed = _prefilled_sample(([("3", 2), ("4", 2)], ["mock-v0"]), index=0)
        uniform = _prefilled_sample(([("default", 4)], ["5"]), index=1)

        out = _compute_metrics_from_samples(make_args(), [mixed, uniform])

        assert out["weight_version/prefill_mixed_version_ratio"] == 0.5
        assert not any(
            key in out for key in ("weight_version/prefill_stale_token_ratio", "weight_version/prefill_lag_max")
        )

    def test_no_prefill_metrics_when_no_call_carries_prefill_spans(self):
        """Engines without prefill weight versions must not synthesise the prefill series."""
        out = _compute_metrics_from_samples(make_args(), [_make_versioned_sample(["4", "5"], index=0)])

        assert "weight_version/min" in out
        assert not any(key.startswith("weight_version/prefill_") for key in out)

    def test_prefill_spans_do_not_move_the_oldest_weight_version_series(self):
        """Prompt KV versions are reported separately and leave weight_version/min to the output spans."""
        sample = _prefilled_sample(([("1", 3)], ["5"]))

        out = _compute_metrics_from_samples(make_args(), [sample])

        assert out["weight_version/min"] == 5
        assert out["weight_version/mixed_version_ratio"] == 0.0


class TestCiPrefillLagMetrics:
    @pytest.mark.parametrize("overrides", [{"ci_test": False, "ci_assert_prefill_lag_max": 1}, {"ci_test": True}])
    def test_lag_is_not_bounded_without_the_ci_gate(self, overrides: dict[str, object]) -> None:
        """Large lag is reported without enforcement unless CI and its bound are both enabled."""
        sample = _prefilled_sample(([("2", 3)], ["5"]))

        metrics = _compute_metrics_from_samples(make_args(**overrides), [sample])

        assert metrics["weight_version/prefill_lag_max"] == 3

    @pytest.mark.parametrize(("prefill_version", "expected_lag", "expected_ratio"), [("4", 1, 1.0), ("5", 0, 0.0)])
    def test_prefill_metrics_within_the_bound_are_reported(
        self, prefill_version: str, expected_lag: int, expected_ratio: float
    ) -> None:
        """CI reports exact freshness metrics for fresh and one-version-old prompts."""
        sample = _prefilled_sample(([(prefill_version, 3)], ["5"]))

        metrics = _compute_metrics_from_samples(make_args(ci_test=True, ci_assert_prefill_lag_max=1), [sample])

        assert metrics["weight_version/prefill_lag_max"] == expected_lag
        assert metrics["weight_version/prefill_stale_token_ratio"] == expected_ratio

    def test_prefill_metrics_beyond_the_bound_fail(self) -> None:
        """CI rejects reported prompt KV lag beyond the configured bound."""
        sample = _prefilled_sample(([("2", 3)], ["5"]))

        with pytest.raises(AssertionError, match="lag metric 3"):
            _compute_metrics_from_samples(make_args(ci_test=True, ci_assert_prefill_lag_max=1), [sample])

    def test_missing_prefill_metrics_fail_instead_of_passing_vacuously(self) -> None:
        """CI fails when an engine reports decode versions but no prompt KV versions."""
        sample = _make_versioned_sample(["5"], index=0)

        with pytest.raises(AssertionError, match="CI requires prompt KV lag"):
            _compute_metrics_from_samples(make_args(ci_test=True, ci_assert_prefill_lag_max=1), [sample])

    def test_prefill_spans_without_comparable_decode_versions_fail(self) -> None:
        """CI fails when prompt versions exist but no call can produce freshness metrics."""
        sample = _prefilled_sample(([("4", 3)], ["default"]))

        with pytest.raises(AssertionError, match="CI requires prompt KV lag"):
            _compute_metrics_from_samples(make_args(ci_test=True, ci_assert_prefill_lag_max=1), [sample])


class TestLogRolloutData:
    def test_the_model_id_comes_from_the_caller_not_from_the_args(self, monkeypatch):
        """One rollout executor serves every policy, so the id must travel with the call, not with the run."""
        calls: list[tuple[dict, str]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.metrics.tracking.log",
            lambda _args, payload, step_key: calls.append((payload, step_key)),
        )
        args = make_args(advantage_estimator="ppo", ci_test=False, log_passrate=False, trainer_model_id=None)

        log_rollout_data(0, args, make_samples_grouped(1, 4), None, 1.0, trainer_model_id="alpha")

        [(payload, step_key)] = calls
        assert step_key == "alpha/rollout/step"
        assert all(key.startswith("alpha/") for key in payload)


class TestEvalMetrics:
    def test_eval_metrics_are_not_namespaced_by_policy(self, monkeypatch):
        """Pinning the status quo: a run training several policies is refused an eval, so eval keeps one step axis."""
        calls: list[tuple[dict, str]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.metrics.tracking.log",
            lambda _args, payload, step_key: calls.append((payload, step_key)),
        )
        args = make_args(log_passrate=False, trainer_model_id="alpha")

        log_eval_rollout_data(0, args, {"gsm8k": {"rewards": [1.0, 0.0]}})

        [(payload, step_key)] = calls
        assert step_key == "eval/step"
        assert payload["eval/gsm8k"] == 0.5
        assert not any(key.startswith("alpha/") for key in payload)
