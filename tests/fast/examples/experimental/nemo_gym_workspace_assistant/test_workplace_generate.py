import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import compute_custom_function_config
from miles.utils.args.runtime import RolloutConfig
from miles.utils.types import Sample

_MODULE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[5]
        / "examples/experimental/nemo-gym-workspace-assistant/workplace_generate.py"
    )
)


class TestWorkplaceRoutingReplay:
    @pytest.mark.parametrize("expert_id", [0, 4])
    async def test_real_rollout_config_validates_expert_indices(
        self, monkeypatch: pytest.MonkeyPatch, expert_id: int
    ) -> None:
        """Routing validation retains flat backend fields in the actual rollout hook view."""
        config = parse_megatron_test_config("--num-experts", "4", "--moe-router-topk", "2")
        rollout = RolloutConfig.slice_from(config).model_copy(update={"use_rollout_routing_replay": True})
        args = compute_custom_function_config(rollout, CustomFunctionConfig(path="workplace_generate.generate"))
        sample = Sample(
            tokens=[1, 2],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.5],
            rollout_routed_experts=np.full((1, args.num_layers, 2), expert_id, dtype=np.int32),
            metadata={"workplace_episode_valid": True, "workplace_turns": 1, "workplace_stop": "done"},
        )
        output = GenerateFnOutput(samples=sample)

        async def generate_episode(input: GenerateFnInput) -> GenerateFnOutput:
            return output

        monkeypatch.setitem(_MODULE["generate"].__globals__, "agentic_generate", generate_episode)
        input = GenerateFnInput(state=SimpleNamespace(args=args), sample=sample, sampling_params={}, evaluation=False)
        if expert_id == 4:
            with pytest.raises(ValueError, match="Invalid routing replay expert indices"):
                await _MODULE["generate"](input)
        else:
            assert await _MODULE["generate"](input) is output
