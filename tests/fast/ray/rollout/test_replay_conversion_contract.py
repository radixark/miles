import numpy as np
import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data


class TestReplayConversionContract:
    @pytest.mark.parametrize(
        "flag,field",
        [
            ("use_rollout_routing_replay", "rollout_routed_experts"),
            ("use_rollout_indexer_replay", "rollout_indexer_topk"),
        ],
    )
    def test_enabled_replay_rejects_missing_engine_capture(self, flag: str, field: str) -> None:
        """Training cannot silently proceed when requested replay metadata is absent."""
        args = make_args(rewards_normalization=False, **{flag: True})

        with pytest.raises(ValueError, match=field):
            convert_samples_to_train_data(
                args,
                [make_sample()],
                metadata={},
                custom_convert_samples_to_train_data_func=None,
                custom_reward_post_process_func=None,
            )

    def test_captured_replay_arrays_keep_each_samples_values(self) -> None:
        """Routing and indexer captures reach training without swapping sample rows."""
        samples = [make_sample(index=7), make_sample(index=8)]
        for index, sample in enumerate(samples):
            sample.rollout_routed_experts = np.full((2, 3, 2), index + 1, dtype=np.int32)
            sample.rollout_indexer_topk = np.full((2, 1, 4), index + 11, dtype=np.int32)

        out = convert_samples_to_train_data(
            make_args(rewards_normalization=False, use_rollout_routing_replay=True, use_rollout_indexer_replay=True),
            samples,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )

        for index in range(2):
            np.testing.assert_array_equal(out["rollout_routed_experts"][index], samples[index].rollout_routed_experts)
            np.testing.assert_array_equal(out["rollout_indexer_topk"][index], samples[index].rollout_indexer_topk)
