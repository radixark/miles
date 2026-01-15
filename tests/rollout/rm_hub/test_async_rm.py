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
    def test_math_rm(self, mock_args):
        mock_args.rm_type = "math"
        sample = Sample(prompt="", response=r"\boxed{42}", label="42")
        reward = run(async_rm(mock_args, sample))
        assert reward == 1

    def test_math_rm_incorrect(self, mock_args):
        mock_args.rm_type = "math"
        sample = Sample(prompt="", response=r"\boxed{wrong}", label="42")
        reward = run(async_rm(mock_args, sample))
        assert reward == 0

    def test_f1_rm(self, mock_args):
        mock_args.rm_type = "f1"
        sample = Sample(prompt="", response="hello world", label="hello world")
        reward = run(async_rm(mock_args, sample))
        assert reward == 1.0

    def test_f1_rm_partial(self, mock_args):
        mock_args.rm_type = "f1"
        sample = Sample(prompt="", response="hello", label="hello world")
        reward = run(async_rm(mock_args, sample))
        assert 0 < reward < 1

    def test_dapo_rm(self, mock_args):
        mock_args.rm_type = "dapo"
        sample = Sample(prompt="", response="Answer: 42", label="42")
        result = run(async_rm(mock_args, sample))
        assert result["score"] == 1.0

    def test_deepscaler_rm(self, mock_args):
        mock_args.rm_type = "deepscaler"
        sample = Sample(prompt="", response=r"</think>\boxed{42}", label="42")
        reward = run(async_rm(mock_args, sample))
        assert reward == 1

    def test_gpqa_rm(self, mock_args):
        mock_args.rm_type = "gpqa"
        sample = Sample(prompt="", response="Answer: A", label="A")
        reward = run(async_rm(mock_args, sample))
        assert reward == 1.0

    def test_random_rm(self, mock_args):
        mock_args.rm_type = "random"
        sample = Sample(prompt="", response="anything", label="anything")
        reward = run(async_rm(mock_args, sample))
        assert reward in [0, 1]

    def test_boxed_prefix_preprocessing(self, mock_args):
        mock_args.rm_type = "boxed_math"
        sample = Sample(prompt="", response=r"Final answer is \boxed{42}", label="42")
        reward = run(async_rm(mock_args, sample))
        assert reward == 1

    def test_rm_type_from_metadata(self, mock_args):
        mock_args.rm_type = None
        sample = Sample(prompt="", response=r"\boxed{42}", label="42", metadata={"rm_type": "math"})
        reward = run(async_rm(mock_args, sample))
        assert reward == 1

    def test_unknown_rm_type_raises(self, mock_args):
        mock_args.rm_type = "unknown_type"
        sample = Sample(prompt="", response="test", label="test")
        with pytest.raises(NotImplementedError, match="not implemented"):
            run(async_rm(mock_args, sample))

    def test_empty_rm_type_raises(self, mock_args):
        mock_args.rm_type = ""
        sample = Sample(prompt="", response="test", label="test")
        with pytest.raises(NotImplementedError, match="not specified"):
            run(async_rm(mock_args, sample))


class TestBatchedAsyncRm:
    def test_batched_math_rm(self, mock_args):
        mock_args.rm_type = "math"
        samples = [
            Sample(prompt="", response=r"\boxed{42}", label="42"),
            Sample(prompt="", response=r"\boxed{100}", label="100"),
            Sample(prompt="", response=r"\boxed{wrong}", label="42"),
        ]
        rewards = run(batched_async_rm(mock_args, samples))
        assert rewards == [1, 1, 0]

    def test_batched_f1_rm(self, mock_args):
        mock_args.rm_type = "f1"
        samples = [
            Sample(prompt="", response="hello world", label="hello world"),
            Sample(prompt="", response="different", label="something else"),
        ]
        rewards = run(batched_async_rm(mock_args, samples))
        assert rewards[0] == 1.0
        assert rewards[1] == 0

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
