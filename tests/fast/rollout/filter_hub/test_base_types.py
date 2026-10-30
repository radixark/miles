from argparse import Namespace

from miles.rollout.filter_hub.base_types import MetricGatherer
from miles.rollout.filter_hub.common_filters import apply_reward_nonzero_std_filter
from miles.utils.types import Sample


class TestMetricGathererRawReward:
    def test_the_mean_covers_kept_and_dropped_groups_alike(self):
        """The gatherer sees every group before the filter, so the mean must not depend on the keep decision."""
        gatherer = MetricGatherer()

        kept = []
        for rewards in [[1.0, 1.0]] * 8 + [[0.0, 0.0], [1.0, 0.0]]:
            group = [_sample(reward=reward) for reward in rewards]
            gatherer.on_group_before_dynamic_filter(_args(), group)
            output = apply_reward_nonzero_std_filter(_args(), group)
            if output.keep:
                kept.extend(group)
            else:
                gatherer.on_dynamic_filter_drop(output.reason)

        assert [sample.reward for sample in kept] == [1.0, 0.0]
        assert gatherer.collect() == {
            "rollout/raw_reward": 0.85,
            "rollout/dynamic_filter/drop_zero_std_1.0": 8,
            "rollout/dynamic_filter/drop_zero_std_0.0": 1,
        }

    def test_a_nested_group_contributes_every_inner_sample(self):
        """Multi-turn groups arrive as lists of lists; counting outer items would weight turns unevenly."""
        gatherer = MetricGatherer()

        gatherer.on_group_before_dynamic_filter(
            _args(), [[_sample(reward=1.0)], [_sample(reward=0.0), _sample(reward=0.5)]]
        )

        assert gatherer.collect()["rollout/raw_reward"] == 0.5

    def test_a_dict_reward_is_read_through_the_reward_key(self):
        """Custom generate functions store dict rewards; the metric follows --reward-key."""
        gatherer = MetricGatherer()

        gatherer.on_group_before_dynamic_filter(
            _args(reward_key="reward_value"), [_sample(reward={"reward_value": 0.25, "outcome": "x"})]
        )

        assert gatherer.collect()["rollout/raw_reward"] == 0.25

    def test_an_unkeyed_structured_reward_does_not_report_a_raw_reward(self):
        """Structured custom-RM payloads have no unambiguous scalar value to average."""
        gatherer = MetricGatherer()

        gatherer.on_group_before_dynamic_filter(
            _args(), [_sample(reward={"teacher": {"reward": 0.25}, "student": {"reward": 0.5}})]
        )

        assert "rollout/raw_reward" not in gatherer.collect()

    def test_no_offered_group_reports_no_metric(self):
        """A window without generation must yield no point rather than a fabricated zero."""
        gatherer = MetricGatherer()

        assert "rollout/raw_reward" not in gatherer.collect()


def _args(reward_key: str | None = None) -> Namespace:
    return Namespace(reward_key=reward_key)


def _sample(reward: float | dict | None) -> Sample:
    return Sample(group_index=0, index=0, prompt="p", label="l", reward=reward)


class TestMetricGathererUnscoredSamples:
    def test_an_unscored_sample_is_left_out_of_the_mean_instead_of_crashing_it(self):
        """A late-aborted group-RM group arrives COMPLETED with reward None; it carries no score to average."""
        gatherer = MetricGatherer()

        gatherer.on_group_before_dynamic_filter(_args(), [_sample(reward=1.0), _sample(reward=None)])

        assert gatherer.collect()["rollout/raw_reward"] == 1.0
