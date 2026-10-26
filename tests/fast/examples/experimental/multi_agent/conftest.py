from collections.abc import Iterator
from typing import Any

import pytest

from miles.utils.function_registry import function_registry
from miles.utils.types import Sample


class _AgentService:
    def __init__(self) -> None:
        self.responses: list[str | Exception] = []
        self.payloads: list[dict[str, Any]] = []
        self.rewarded: list[list[Sample]] = []

    async def post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return {
            "text": response,
            "meta_info": {"output_token_logprobs": [[-0.1, 7]], "finish_reason": {"type": "stop"}},
        }

    async def reward(self, args: Any, samples: list[Sample]) -> list[float]:
        self.rewarded.append(list(samples))
        return [1.0 if sample.response_content == "correct" else 0.5 for sample in samples]


@pytest.fixture
def agent_service(monkeypatch: pytest.MonkeyPatch) -> Iterator[_AgentService]:
    service = _AgentService()
    monkeypatch.setattr("examples.experimental.multi_agent.agent_system.post", service.post)
    with function_registry.temporary("test-agent-rewards", service.reward):
        yield service
