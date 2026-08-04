import msgpack
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.utils.types import WeightVersionSpan, WeightVersionsPerCall


class TestWeightVersionsAreSerializable:
    def test_weight_version_spans_leave_as_plain_data(self):
        """The column is msgpack-encoded on its way to the trainer, and no dataclass survives that."""
        sample = make_sample(index=0, reward=1.0)
        sample.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan(version="3", abs_start=0, abs_end=2)])
        ]

        train_data = convert_samples_to_train_data(
            make_args(),
            [sample],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )

        assert train_data["weight_versions"] == [[[{"version": "3", "abs_start": 0, "abs_end": 2}]]]
        msgpack.packb(train_data["weight_versions"])
