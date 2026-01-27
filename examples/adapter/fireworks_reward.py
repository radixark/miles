import asyncio
from argparse import Namespace
from typing import Any

from eval_protocol.models import EvaluationRow, Message

from miles.utils.misc import load_function
from miles.utils.types import Sample


async def calc_single_reward(
    args: Namespace, messages: list[dict[str, Any]], label: str | None, reward_function_path: str, **kwargs
) -> float:
    row = EvaluationRow(
        messages=[Message.model_validate(message) for message in messages], ground_truth=label, **kwargs
    )
    reward_function = load_function(reward_function_path)
    row = reward_function(row, **kwargs)
    return row.evaluation_result.score


async def calc_rewards(args: Namespace, samples: list[Sample], reward_function_path: str, **kwargs) -> list[float]:
    tasks = [
        calc_single_reward(args, sample.messages, sample.label, reward_function_path, **kwargs) for sample in samples
    ]
    rewards = await asyncio.gather(*tasks)
    for sample, reward in zip(samples, rewards, strict=True):
        sample.reward = reward
    return samples
