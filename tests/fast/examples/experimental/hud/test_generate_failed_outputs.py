from collections.abc import Callable
from types import SimpleNamespace

import pytest
from examples.experimental.hud import rollout
from tests.fast.examples.experimental.hud.hud_generation_fixtures import failed_hud_episode

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample

__all__ = ["failed_hud_episode"]


class TestHudFailedOutputs:
    @pytest.mark.parametrize("outcome", ["no_run", "no_tokens", "no_screenshot", "vision_mismatch"])
    async def test_unusable_episode_returns_a_wrapped_structurally_valid_writeoff(
        self, failed_hud_episode: Callable[[str], SimpleNamespace], outcome: str
    ) -> None:
        """Every HUD early return preserves the v2 result and the image-bearing failed sample."""
        state = failed_hud_episode(outcome)
        sample = Sample(prompt="question", metadata={"hud_task": {"env": "unit", "id": "task"}})

        result = await rollout.generate(
            GenerateFnInput(state=state, sample=sample, sampling_params={}, evaluation=False)
        )

        assert isinstance(result, GenerateFnOutput)
        assert result.samples is sample
        assert sample.status == Sample.Status.FAILED
        assert sample.remove_sample
        assert sample.tokens == [1, 2, 3]
        assert sample.response_length == 0
        assert sample.response == ""
        assert sample.loss_mask == []
        assert sample.rollout_log_probs == []
        assert set(sample.multimodal_train_inputs) == {"pixel_values", "image_grid_thw"}
