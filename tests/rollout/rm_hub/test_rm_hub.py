from unittest.mock import MagicMock

import pytest

from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils.async_utils import run
from miles.utils.types import Sample


@pytest.fixture
def mock_args():
    args = MagicMock()
    args.custom_rm_path = None
    args.rm_type = None
    args.rm_url = None
    return args


class TestAsyncRm:
    @pytest.mark.parametrize(
        "rm_type,response,label,expected",
        [
            ("math", r"\boxed{42}", "42", 1),
            ("math", r"\boxed{wrong}", "42", 0),
            ("f1", "hello world", "hello world", 1.0),
            ("dapo", "Answer: 42", "42", {"score": 1.0}),
            ("deepscaler", r"</think>\boxed{42}", "42", 1),
            ("gpqa", "Answer: A", "A", 1.0),
            ("boxed_f1", r"Final answer is \boxed{hello world}", "hello world", 1.0),
        ],
    )
    def test_rm_types(self, mock_args, rm_type, response, label, expected):
        mock_args.rm_type = rm_type
        sample = Sample(prompt="", response=response, label=label)
        reward = run(async_rm(mock_args, sample))
        if isinstance(expected, dict):
            for k, v in expected.items():
                assert reward[k] == v
        else:
            assert reward == expected

    def test_f1_rm_partial(self, mock_args):
        mock_args.rm_type = "f1"
        sample = Sample(prompt="", response="hello", label="hello world")
        reward = run(async_rm(mock_args, sample))
        assert 0 < reward < 1

    def test_random_rm(self, mock_args):
        mock_args.rm_type = "random"
        sample = Sample(prompt="", response="anything", label="anything")
        reward = run(async_rm(mock_args, sample))
        assert reward in [0, 1]

    def test_rm_type_from_metadata(self, mock_args):
        mock_args.rm_type = None
        sample = Sample(prompt="", response=r"\boxed{42}", label="42", metadata={"rm_type": "math"})
        reward = run(async_rm(mock_args, sample))
        assert reward == 1

    @pytest.mark.parametrize(
        "rm_type,match",
        [
            ("unknown_type", "not implemented"),
            ("", "not specified"),
        ],
    )
    def test_invalid_rm_type_raises(self, mock_args, rm_type, match):
        mock_args.rm_type = rm_type
        sample = Sample(prompt="", response="test", label="test")
        with pytest.raises(NotImplementedError, match=match):
            run(async_rm(mock_args, sample))


class TestBatchedAsyncRm:
    @pytest.mark.parametrize(
        "rm_type,samples_data,expected",
        [
            (
                "math",
                [(r"\boxed{42}", "42"), (r"\boxed{100}", "100"), (r"\boxed{wrong}", "42")],
                [1, 1, 0],
            ),
            (
                "f1",
                [("hello world", "hello world"), ("different", "something else")],
                [1.0, 0],
            ),
        ],
    )
    def test_batched_rm(self, mock_args, rm_type, samples_data, expected):
        mock_args.rm_type = rm_type
        samples = [Sample(prompt="", response=r, label=label) for r, label in samples_data]
        rewards = run(batched_async_rm(mock_args, samples))
        assert rewards == expected

    def test_inplace_set_reward_field(self, mock_args):
        mock_args.rm_type = "math"
        samples = [
            Sample(prompt="", response=r"\boxed{42}", label="42"),
            Sample(prompt="", response=r"\boxed{100}", label="100"),
        ]
        result = run(batched_async_rm(mock_args, samples, inplace_set_reward_field=True))
        assert result is None
        assert samples[0].reward == 1
        assert samples[1].reward == 1

    def test_inplace_raises_on_existing_reward(self, mock_args):
        mock_args.rm_type = "math"
        samples = [Sample(prompt="", response=r"\boxed{42}", label="42", reward=0.5)]
        with pytest.raises(AssertionError, match="Overriding"):
            run(batched_async_rm(mock_args, samples, inplace_set_reward_field=True))

    def test_empty_samples(self, mock_args):
        mock_args.rm_type = "math"
        rewards = run(batched_async_rm(mock_args, []))
        assert rewards == []

    def test_mixed_rm_types_via_metadata(self, mock_args):
        mock_args.rm_type = None
        samples = [
            Sample(prompt="", response=r"\boxed{42}", label="42", metadata={"rm_type": "math"}),
            Sample(prompt="", response="hello", label="hello", metadata={"rm_type": "f1"}),
        ]
        rewards = run(batched_async_rm(mock_args, samples))
        assert rewards[0] == 1
        assert rewards[1] == 1.0
