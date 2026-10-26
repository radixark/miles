import asyncio
from typing import Any

import pytest

from examples.experimental.multi_agent.agent_system import AgentState, generate_response
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.types import Sample


class TestAgentRuntimeIsolation:
    async def test_concurrent_rollouts_share_immutable_config_but_not_samples_or_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrent agent responses keep mutable state local to their rollout."""
        config = ImmutableNamespace(rollout_max_context_len=8, sglang_router_ip="localhost", sglang_router_port=8000)
        entered: list[int] = []
        both_entered = asyncio.Event()

        async def post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
            identity = payload["input_ids"][0]
            entered.append(identity)
            if len(entered) == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), timeout=1)
            return {
                "text": f"reason</think>answer-{identity}",
                "meta_info": {"output_token_logprobs": [[-0.1, identity + 10]], "finish_reason": {"type": "stop"}},
            }

        monkeypatch.setattr("examples.experimental.multi_agent.agent_system.post", post)
        first = AgentState(
            sample=Sample(prompt="first"),
            tokenizer=lambda *args, **kwargs: {"input_ids": [1]},
            sampling_params={"max_new_tokens": 20},
        )
        second = AgentState(
            sample=Sample(prompt="second"),
            tokenizer=lambda *args, **kwargs: {"input_ids": [2]},
            sampling_params={"max_new_tokens": 20},
        )
        original = dict(config)

        results = await asyncio.gather(
            generate_response(config, "first prompt", "solver", state=first),
            generate_response(config, "second prompt", "solver", state=second),
        )

        assert results == ["answer-1", "answer-2"]
        assert first.results["solver"][0].tokens == [1, 11]
        assert second.results["solver"][0].tokens == [2, 12]
        assert first.results["rewriter"] == second.results["rewriter"] == []
        assert first.results["solver"][0].status is Sample.Status.COMPLETED
        assert first.sample.prompt == "first"
        assert second.sample.prompt == "second"
        assert first.sample.response_length == second.sample.response_length == 0
        assert first.sampling_params == second.sampling_params == {"max_new_tokens": 20}
        assert dict(config) == original

    async def test_exhausted_context_does_not_publish_a_partial_sample(self) -> None:
        """A prompt filling the context window leaves the rollout's result lists untouched."""
        config = ImmutableNamespace(rollout_max_context_len=2, sglang_router_ip="localhost", sglang_router_port=8000)
        state = AgentState(
            sample=Sample(prompt="original"),
            tokenizer=lambda *args, **kwargs: {"input_ids": [1, 2]},
            sampling_params={"max_new_tokens": 3},
        )

        assert await generate_response(config, "full", "solver", state=state) is None
        assert state.results == {"solver": [], "rewriter": [], "selector": []}
        assert state.sample.prompt == "original"
