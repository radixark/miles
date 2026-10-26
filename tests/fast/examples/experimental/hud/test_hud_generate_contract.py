import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from examples.experimental.hud import rollout

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample


@pytest.mark.parametrize(("settings", "expected_width"), [({}, 640), ({"hud_shot_width": 512}, 512)])
def test_hud_screenshot_width_preserves_default_and_explicit_recipe(
    settings: dict[str, int], expected_width: int
) -> None:
    """Omitted screenshot width retains 640 while the 2048 recipe can explicitly select 512."""
    config = rollout.HudConfig(hud_env_dir="/hud", hud_snapshot_name="snapshot", **settings)

    assert config.hud_shot_width == expected_width


@pytest.mark.parametrize("failed", [False, True])
async def test_episode_result_wraps_the_same_sample_and_uses_supplied_state(
    monkeypatch: pytest.MonkeyPatch, failed: bool
) -> None:
    """HUD success and failure both pass caller-owned state through the v2 generation boundary."""
    state = SimpleNamespace(
        args=SimpleNamespace(partial_rollout=False, hud_episode_timeout_s=30),
        tokenizer=SimpleNamespace(decode=Mock(return_value="answer")),
    )
    sample = Sample(prompt="question", metadata={"hud_task": {"id": "task"}})
    run = SimpleNamespace(reward=0.75, trace=SimpleNamespace(status="done"))
    task = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(runs=[run])))
    if failed:
        task.run.side_effect = RuntimeError("provider failed")
    failure = Mock(return_value=sample)
    vision = Mock(return_value={"pixels": "image"})
    stitch = Mock(return_value=([10, 20], [1], [-0.2], 1))
    monkeypatch.setattr(rollout, "_mint_task", lambda meta: task)
    monkeypatch.setattr(rollout, "_build_agent", lambda args, sampling: "agent")
    monkeypatch.setattr(rollout, "_shared_runtime", lambda args: "runtime")
    monkeypatch.setattr(rollout, "_sandbox_gate", lambda args: asyncio.Semaphore(1))
    monkeypatch.setattr(rollout, "_collect_turns", lambda run: ["turn"])
    monkeypatch.setattr(rollout, "_collect_images", lambda run: ["image"])
    monkeypatch.setattr(rollout, "_stitch", stitch)
    monkeypatch.setattr(rollout, "_vision_inputs", vision)
    monkeypatch.setattr(rollout, "_failed", failure)

    output = await rollout.generate(GenerateFnInput(state=state, sample=sample, sampling_params={}, evaluation=False))

    assert isinstance(output, GenerateFnOutput)
    assert output.samples is sample
    task.run.assert_awaited_once_with("agent", runtime="runtime", rollout_timeout=30.0)
    if failed:
        assert failure.call_args.args == (sample, "hud episode errored: RuntimeError", state)
        stitch.assert_not_called()
    else:
        assert stitch.call_args.args[1] is state.tokenizer
        assert vision.call_args.args[0] is state
        assert sample.tokens == [10, 20]
        assert sample.response == "answer"
        assert sample.response_length == 1
        assert sample.metadata["reward"] == 0.75
        assert sample.status == Sample.Status.COMPLETED
