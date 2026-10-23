"""TMax's reward function and pure OPD's reward post-process (generate.py)."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "tmax"))

import generate  # noqa: E402

from miles.utils.types import Sample  # noqa: E402

ARGS = SimpleNamespace(reward_key=None)


def test_reward_is_the_sandbox_verdict():
    samples = [Sample(metadata={"reward": 1.0}), Sample(metadata={"reward": None}), Sample(metadata={})]
    assert asyncio.run(generate.reward_func(ARGS, samples)) == [1.0, None, None]
    assert asyncio.run(generate.reward_func(ARGS, samples[0])) == 1.0


def test_pure_opd_trains_on_zero_reward_and_logs_the_task_reward():
    samples = [Sample(reward=1.0), Sample(reward=0.0), Sample(reward=1.0)]
    raw_rewards, rewards = generate.opd_pure_rewards(ARGS, samples)
    assert raw_rewards == [1.0, 0.0, 1.0], "raw_reward (rollout/raw_reward) stays the task reward"
    assert rewards == [0.0, 0.0, 0.0], "GRPO's advantage is then 0, leaving the OPD term alone"
