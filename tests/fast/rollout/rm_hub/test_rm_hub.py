from unittest.mock import MagicMock

import pytest

from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils.async_utils import run
from miles.utils.misc import function_registry
from miles.utils.types import Sample

PER_SAMPLE_RM_PATH = "test.per_sample.rm"
GLOBAL_RM_PATH = "test.global.rm"


async def _per_sample_rm(args, sample, **kwargs):
    return 111


async def _global_rm(args, sample, **kwargs):
    return 999


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

    def test_deterministic_random_rm_returns_binary(self, mock_args):
        mock_args.rm_type = "deterministic_random"
        sample = Sample(prompt="", response="hello", label="", tokens=[1, 2, 3])
        reward = run(async_rm(mock_args, sample))
        assert reward in [0, 1]

    def test_deterministic_random_rm_is_deterministic(self, mock_args):
        mock_args.rm_type = "deterministic_random"
        sample = Sample(prompt="", response="hello world", label="", tokens=[10, 20])
        rewards = [run(async_rm(mock_args, sample)) for _ in range(5)]
        assert len(set(rewards)) == 1

    def test_deterministic_random_rm_differs_by_response(self, mock_args):
        mock_args.rm_type = "deterministic_random"
        samples = [Sample(prompt="", response=f"response_{i}", label="", tokens=[1, 2, 3]) for i in range(20)]
        rewards = [run(async_rm(mock_args, s)) for s in samples]
        assert 0 in rewards and 1 in rewards

    def test_deterministic_random_rm_differs_by_tokens(self, mock_args):
        """Same response with different tokens yields both reward values across many samples."""
        mock_args.rm_type = "deterministic_random"
        samples = [Sample(prompt="", response="same", label="", tokens=[i, i + 1, i + 2]) for i in range(20)]
        rewards = [run(async_rm(mock_args, s)) for s in samples]
        assert 0 in rewards and 1 in rewards

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

    def test_per_sample_custom_rm_takes_priority(self, mock_args):
        mock_args.custom_rm_path = GLOBAL_RM_PATH
        sample = Sample(prompt="", response="", label="", custom_rm_path=PER_SAMPLE_RM_PATH)
        with function_registry.temporary(PER_SAMPLE_RM_PATH, _per_sample_rm), function_registry.temporary(
            GLOBAL_RM_PATH, _global_rm
        ):
            reward = run(async_rm(mock_args, sample))
        assert reward == 111

    def test_falls_back_to_global_custom_rm(self, mock_args):
        mock_args.custom_rm_path = GLOBAL_RM_PATH
        sample = Sample(prompt="", response="", label="")
        with function_registry.temporary(GLOBAL_RM_PATH, _global_rm):
            reward = run(async_rm(mock_args, sample))
        assert reward == 999


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
