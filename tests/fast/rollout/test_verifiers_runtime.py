import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Verifiers requires Python 3.11+", allow_module_level=True)

pytest.importorskip("verifiers", minversion="0.2.0")
pytest.importorskip("renderers", minversion="0.1.8")

from verifiers.v1.clients.train import TrainClient
from verifiers.v1.dialects import ChatDialect, ResponsesDialect
from verifiers.v1.env import EnvConfig, Environment
from verifiers.v1.types import SamplingConfig

from miles.rollout.verifiers_rollout import MilesSGLangTransport


def _args(**overrides):
    values = {
        "lora_adapter_path": None,
        "lora_rank": 0,
        "rollout_max_context_len": 64,
        "rollout_max_prompt_len": None,
        "rollout_max_response_len": 8,
        "rollout_skip_special_tokens": True,
        "rollout_stop": None,
        "rollout_stop_token_ids": None,
        "sglang_model_routers": None,
        "sglang_router_ip": "127.0.0.1",
        "sglang_router_policy": "round_robin",
        "sglang_router_port": 30000,
        "sglang_tokenizer_path": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_minimal_env_config_uses_the_v1_environment_contract():
    config = EnvConfig.model_validate({"taskset": {"id": "harbor"}})

    environment = Environment(config)

    assert config.is_legacy is False
    assert config.env_id == "harbor"
    assert type(environment.taskset).__name__ == "HarborTaskset"


class _Rendered:
    token_ids = [10, 11]
    multi_modal_data = None
    is_content = [True, True]

    @staticmethod
    def message_token_spans():
        return [(0, 2)]


class _Renderer:
    supports_tools = True

    def render(self, messages, *, tools, add_generation_prompt):
        assert messages == [{"role": "user", "content": "question"}]
        assert tools is None
        assert add_generation_prompt is True
        return _Rendered()

    @staticmethod
    def get_stop_token_ids():
        return [99]

    @staticmethod
    def parse_response(token_ids, *, tools):
        assert token_ids == [20, 21]
        assert tools is None
        return SimpleNamespace(content="answer", reasoning_content=None, tool_calls=[])


@pytest.mark.asyncio
async def test_published_train_client_runs_through_miles_transport(monkeypatch):
    async def fake_post(_url, _payload, headers=None):
        assert headers is None
        return {
            "request_id": "request-id",
            "meta_info": {
                "completion_tokens": 2,
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, 20], [-0.2, 21]],
            },
        }

    monkeypatch.setattr("miles.utils.http_utils.post", fake_post)
    client = TrainClient(MilesSGLangTransport(_args()), renderer_model_name="test/model")
    client._pool = _Renderer()

    response = await client.get_response(
        ChatDialect(),
        {"messages": [{"role": "user", "content": "question"}]},
        "test/model",
        SamplingConfig(temperature=0.2, max_tokens=2),
        session_id="trace-id",
    )

    assert response.message.content == "answer"
    assert response.tokens.prompt_ids == [10, 11]
    assert response.tokens.completion_ids == [20, 21]
    assert response.tokens.completion_logprobs == [-0.1, -0.2]
    ChatDialect().validate_response(response.raw)


@pytest.mark.asyncio
async def test_published_train_client_rejects_non_chat_dialects():
    client = TrainClient(MilesSGLangTransport(_args()), renderer_model_name="test/model")

    with pytest.raises(NotImplementedError, match="chat-completions dialect"):
        await client.get_response(
            ResponsesDialect(),
            {"input": "question"},
            "test/model",
            SamplingConfig(max_tokens=2),
        )
