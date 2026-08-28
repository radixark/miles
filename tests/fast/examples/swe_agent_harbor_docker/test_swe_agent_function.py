"""Failure semantics of the Harbor agent-server client.

The example directory name is not a Python identifier, so the module is loaded
by path, as test_agent_metrics does.
"""

import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from miles.rollout.agent_function import InfraAbort

REPO_ROOT = Path(__file__).resolve().parents[4]
AGENT_FUNCTION = REPO_ROOT / "examples" / "swe-agent-harbor-docker" / "swe_agent_function.py"


@pytest.fixture(scope="module")
def agent_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("swe_agent_harbor_docker_agent_function", AGENT_FUNCTION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(agent_module, post):
    agent_module._post_agent_server = post
    return asyncio.run(
        agent_module.run("http://t:1/sessions/s", [{"role": "user", "content": "q"}], {}, {"instance_id": "task-1"})
    )


def test_server_verdict_passes_through(agent_module, monkeypatch):
    async def post(url, payload):
        assert payload["instance_id"] == "task-1"
        return {
            "reward": 1.0,
            "exit_status": "Submitted",
            "eval_report": {"reward": 1.0},
            "agent_metrics": {"turns": 4},
        }

    result = _run(agent_module, post)
    assert result == {
        "reward": 1.0,
        "exit_status": "Submitted",
        "eval_report": {"reward": 1.0},
        "agent_metrics": {"turns": 4},
    }


def test_client_timeout_scores_zero(agent_module, monkeypatch):
    """The trial is still running on the server and the policy may be what stalls it: a negative sample."""

    async def slow(url, payload):
        await asyncio.sleep(10)

    monkeypatch.setenv("AGENT_TRIAL_TIMEOUT", "0")
    monkeypatch.setattr(agent_module, "_agent_trial_timeout_s", lambda: 0.01)
    result = _run(agent_module, slow)
    assert result["reward"] == 0.0
    assert result["exit_status"] == "TimeLimitExceeded"


def test_unreachable_agent_server_discards_the_sample(agent_module):
    """No trial ran: nothing to attribute to the policy."""

    async def unreachable(url, payload):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(InfraAbort) as excinfo:
        _run(agent_module, unreachable)
    assert excinfo.value.exit_status == "ServerUnreachable"


def test_server_error_scores_zero(agent_module):
    async def erroring(url, payload):
        raise httpx.HTTPStatusError("500", request=httpx.Request("POST", url), response=httpx.Response(500))

    result = _run(agent_module, erroring)
    assert result["reward"] == 0.0
    assert result["exit_status"] == "AgentError"
