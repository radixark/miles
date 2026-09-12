from miles.ray.rollout.train_data_conversion import ROLLOUT_DATA_TENSOR_DTYPES
from miles.utils.types import Sample


def test_wire_dtypes_keep_binary_mask_and_add_float_channels():
    assert ROLLOUT_DATA_TENSOR_DTYPES["loss_masks"] == "int32"
    assert ROLLOUT_DATA_TENSOR_DTYPES["loss_weights"] == "float32"
    assert ROLLOUT_DATA_TENSOR_DTYPES["advantages"] == "float32"


def test_sample_round_trips_the_channels():
    sample = Sample.from_dict(
        {
            "prompt": "p",
            "tokens": [1, 2, 3],
            "response_length": 2,
            "loss_mask": [1, 1],
            "loss_weights": [0.5, -1.0],
            "advantages": [2.0, 0.0],
            "status": "completed",
        }
    )
    assert sample.loss_weights == [0.5, -1.0]
    assert sample.advantages == [2.0, 0.0]
    assert Sample.from_dict(sample.to_dict()).loss_weights == [0.5, -1.0]


def test_merge_pads_the_channels_over_the_observation_span(monkeypatch):
    from miles.rollout.generate_utils.sample_utils import merge_samples

    class _Tok:
        def decode(self, tokens):
            return "obs"

    a = Sample(
        prompt="p",
        status=Sample.Status.COMPLETED,
        tokens=[1, 2, 3],
        response="x",
        response_length=1,
        loss_mask=[1],
        loss_weights=[0.5],
        advantages=[1.0],
        rollout_log_probs=[-0.1],
    )
    b = Sample(
        prompt="p",
        status=Sample.Status.COMPLETED,
        tokens=[1, 2, 3, 9, 4, 5],
        response="y",
        response_length=2,
        loss_mask=[1, 1],
        loss_weights=[1.5, 2.5],
        advantages=[0.0, -1.0],
        rollout_log_probs=[-0.2, -0.3],
    )
    merged = merge_samples([a, b], tokenizer=_Tok())
    assert merged.loss_weights == [0.5, 0.0, 1.5, 2.5]
    assert merged.advantages == [1.0, 0.0, 0.0, -1.0]
