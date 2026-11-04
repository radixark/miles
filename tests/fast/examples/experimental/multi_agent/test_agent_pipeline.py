from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from examples.experimental.multi_agent.agent_system import AgentState, run_agent_system

from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.types import Sample

if TYPE_CHECKING:
    from tests.fast.examples.experimental.multi_agent.conftest import _AgentService


class TestAgentPipelineState:
    @pytest.mark.parametrize(
        "responses,expected_counts,expected_rewards",
        [
            (["malformed"], [1, 0, 0], [0.4]),
            (["reason</think>correct", "malformed"], [1, 1, 0], [0.8, 0.4]),
            (["reason</think>correct", "reason</think>correct", RuntimeError("offline")], [1, 1, 0], [0.8, 0.8]),
            (["reason</think>correct", "reason</think>correct", "malformed"], [1, 1, 1], [0.8, 0.8, 0.0]),
            (
                ["reason</think>correct", "reason</think>correct", "reason</think>Judgment: 9"],
                [1, 1, 1],
                [0.8, 0.8, 0.0],
            ),
            (
                ["reason</think>correct", "reason</think>correct", "reason</think>Judgment: 1"],
                [1, 1, 1],
                [1.2, 1.2, 1.2],
            ),
            (
                ["reason</think>correct", "reason</think>wrong", "reason</think>Judgment: 1"],
                [1, 1, 1],
                [0.8, 0.4, 0.4],
            ),
        ],
    )
    async def test_real_pipeline_routes_each_stage_and_applies_rewards_to_only_its_own_samples(
        self,
        agent_service: _AgentService,
        responses: list[str | Exception],
        expected_counts: list[int],
        expected_rewards: list[float],
    ) -> None:
        """Solver, rewriter and selector failures preserve local stage results and reward ownership."""
        args = ImmutableNamespace(
            num_parallel=1,
            rollout_max_context_len=16,
            sglang_router_ip="localhost",
            sglang_router_port=8000,
            incorrect_reward_weight=0.8,
            correct_reward_weight=1.2,
            custom_rm_path=CustomFunctionConfig(path="test-agent-rewards"),
        )
        original = dict(args)
        sample = Sample(prompt="original", label="answer")
        state = AgentState(
            sample=sample,
            tokenizer=lambda *args, **kwargs: {"input_ids": [1, 2]},
            sampling_params={"max_new_tokens": 20},
        )
        agent_service.responses = list(responses)

        result = await run_agent_system(args, state=state)

        assert [len(state.results[key]) for key in ("solver", "rewriter", "selector")] == expected_counts
        assert result == state.results["solver"] + state.results["rewriter"] + state.results["selector"]
        assert [item.reward for item in result] == pytest.approx(expected_rewards)
        assert all(item.tokens == [1, 2, 7] for item in result)
        assert all(item.status is Sample.Status.COMPLETED for item in result)
        assert all(item is not sample for item in result)
        assert all(payload["sampling_params"]["max_new_tokens"] == 14 for payload in agent_service.payloads)
        assert agent_service.responses == []
        assert sample.prompt == "original"
        assert sample.reward is None
        assert state.sampling_params == {"max_new_tokens": 20}
        assert dict(args) == original
