"""The TMax rollout wrapper logs agent/* metrics for groups of Samples and, with session server v2, of list[Sample]."""

import asyncio
import sys
from pathlib import Path

import pytest

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "tmax"))

import generate  # noqa: E402


def _sample(end_reason: str, turns: int) -> Sample:
    return Sample(metadata={"agent_metrics": {"end_reason": end_reason, "turns": turns}})


@pytest.mark.parametrize("v2", [False, True])
def test_agent_metrics_cover_every_episode(monkeypatch, v2):
    episodes = [_sample("submitted", 2), _sample("submitted", 4), _sample("context_full", 6), _sample("max_steps", 8)]
    wrap = (lambda s: [s]) if v2 else (lambda s: s)
    groups = [[wrap(episodes[0]), wrap(episodes[1])], [wrap(episodes[2]), wrap(episodes[3])]]

    async def fake_call_train(self, input):
        return RolloutFnTrainOutput(samples=groups, metrics={"rollout/x": 1.0})

    monkeypatch.setattr(InferenceRolloutFn, "_call_train", fake_call_train)
    rollout_fn = object.__new__(generate.RolloutFn)
    output = asyncio.run(rollout_fn._call_train(type("Input", (), {"rollout_id": 0})()))

    assert output.metrics["rollout/x"] == 1.0
    assert output.metrics["agent/turns_mean"] == 5.0
    assert output.metrics["agent/end_reason/submitted"] == 0.5
    assert output.metrics["agent/end_reason/context_full"] == 0.25
