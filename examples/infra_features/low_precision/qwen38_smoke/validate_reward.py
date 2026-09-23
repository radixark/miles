"""Check DAPO dictionary rewards through the real rollout metrics consumer."""
import asyncio
from argparse import Namespace

from miles.ray.rollout.metrics import _compute_zero_std_metrics
from miles.rollout.rm_hub import async_rm
from miles.utils.types import Sample


async def main() -> None:
    args = Namespace(custom_rm_path=None, rm_type="dapo", reward_key="score", advantage_estimator="grpo")
    samples = []
    for group, answers in enumerate((("42", "42"), ("41", "41"), ("42", "41"))):
        for answer in answers:
            sample = Sample(response=f"Answer: {answer}", label="42", group_index=group)
            sample.reward = await async_rm(args, sample)
            value = sample.get_reward_value(args)
            assert isinstance(value, (int, float)), value
            assert value == (1.0 if answer == "42" else -1.0), sample.reward
            samples.append(sample)
    result = _compute_zero_std_metrics(args, samples)
    assert result["zero_std/count_1.0"] == 1, result
    assert result["zero_std/count_-1.0"] == 1, result
    assert result["zero_std/all_one_percentage"] == 1 / 3, result
    print("REWARD_CONTRACT_PASSED", result)


if __name__ == "__main__":
    asyncio.run(main())
