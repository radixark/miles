import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock

import pytest


def _load_agent_function() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[4] / "examples" / "swe-agent-harbor-docker" / "swe_agent_function.py"
    )
    spec = importlib.util.spec_from_file_location("swe_agent_function", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def agent_function() -> ModuleType:
    return _load_agent_function()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_name", "expected_base_url", "expected_model"),
    [
        (
            "terminus-2",
            "http://10.0.0.1:30000/sessions/session-1/v1",
            "openai/GLM-4.7-Flash",
        ),
        (
            "claude-code",
            "http://10.0.0.1:30000/sessions/session-1",
            "GLM-4.7-Flash",
        ),
    ],
)
async def test_run_builds_agent_specific_api_payload(
    agent_function: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    agent_name: str,
    expected_base_url: str,
    expected_model: str,
) -> None:
    monkeypatch.setenv("AGENT_SERVER_URL", "http://agent-server:11000")
    monkeypatch.setenv("AGENT_MODEL_NAME", "GLM-4.7-Flash")
    monkeypatch.delenv("MILES_ROUTER_EXTERNAL_HOST", raising=False)
    post_agent_server = AsyncMock(return_value={"reward": 1.0})
    monkeypatch.setattr(agent_function, "_post_agent_server", post_agent_server)

    result = await agent_function.run(
        base_url="http://10.0.0.1:30000/sessions/session-1",
        prompt="fix the task",
        request_kwargs={"temperature": 0.7},
        metadata={"agent_name": agent_name, "instance_id": "task-1"},
    )

    post_agent_server.assert_awaited_once_with(
        "http://agent-server:11000/run",
        {
            "agent_name": agent_name,
            "instance_id": "task-1",
            "base_url": expected_base_url,
            "model": expected_model,
            "sampling_params": {"temperature": 0.7},
        },
    )
    assert result == {
        "reward": 1.0,
        "exit_status": "",
        "eval_report": {},
        "agent_metrics": {},
    }


@pytest.mark.asyncio
async def test_run_rewrites_claude_code_endpoint_for_external_sandbox(
    agent_function: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_MODEL_NAME", "GLM-4.7-Flash")
    monkeypatch.setenv("MILES_ROUTER_EXTERNAL_HOST", "trainer.example.test")
    post_agent_server = AsyncMock(return_value={})
    monkeypatch.setattr(agent_function, "_post_agent_server", post_agent_server)

    await agent_function.run(
        base_url="http://10.0.0.1:30123/sessions/session-1",
        prompt="fix the task",
        metadata={
            "agent_name": "claude-code",
            "instance_id": "task-1",
            "max_seq_len": "65536",
            "session_server_id": "10.0.0.1:30123",
            "session_server_instance_id": "session-server-1",
        },
    )

    _, request = post_agent_server.await_args.args
    assert request["base_url"] == ("http://trainer.example.test:30123/sessions/session-1")
    assert request["model"] == "GLM-4.7-Flash"
    assert request["max_seq_len"] == 65536
    assert request["session_server_id"] == "trainer.example.test:30123"
    assert request["session_server_instance_id"] == "session-server-1"
