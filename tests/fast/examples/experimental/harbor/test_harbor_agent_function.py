"""Offline tests for the in-process Harbor agent function (no Harbor, no sandbox)."""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import harbor_agent_function as haf
import pytest


def run_async(coro):
    return asyncio.run(coro)


@pytest.fixture
def tasks_dir(tmp_path, monkeypatch):
    (tmp_path / "task-1").mkdir()
    monkeypatch.setenv("HARBOR_TASKS_DIR", str(tmp_path))
    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2b")
    monkeypatch.delenv("MILES_ROUTER_EXTERNAL_HOST", raising=False)
    monkeypatch.delenv("HARBOR_ENV_KWARGS", raising=False)
    return tmp_path


def _verdict(reward=1.0, **agent_fields):
    t0 = datetime(2026, 1, 1, 0, 0, 0)
    return SimpleNamespace(
        exception_info=None,
        verifier_result=SimpleNamespace(rewards={"reward": reward}),
        agent_result=SimpleNamespace(
            n_input_tokens=1000,
            n_output_tokens=200,
            cost_usd=None,
            n_steps=7,
            metadata={"tool_calls": 5},
            **agent_fields
        ),
        started_at=t0,
        finished_at=t0 + timedelta(seconds=90),
        environment_setup=SimpleNamespace(started_at=t0, finished_at=t0 + timedelta(seconds=10)),
        agent_setup=None,
        agent_execution=SimpleNamespace(started_at=t0 + timedelta(seconds=10), finished_at=t0 + timedelta(seconds=80)),
        verifier=SimpleNamespace(started_at=t0 + timedelta(seconds=80), finished_at=t0 + timedelta(seconds=90)),
    )


# --- trial config ----------------------------------------------------------


def test_environment_type_is_passed_straight_through(tasks_dir, monkeypatch):
    monkeypatch.setenv("HARBOR_ENV_TYPE", "daytona")
    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"auto_snapshot": true}')
    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})
    assert cfg.environment.type.value == "daytona"
    assert cfg.environment.kwargs == {"auto_snapshot": True}
    assert cfg.environment.delete is True


def test_unknown_environment_type_is_an_error_not_docker(tasks_dir, monkeypatch):
    """The agent server silently fell back to docker on an unknown HARBOR_ENV_TYPE; this path refuses."""
    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2bb")
    with pytest.raises(ValueError):
        haf.build_trial_config({"instance_id": "task-1"}, "http://s/v1", {})


def test_environment_type_is_required(tasks_dir, monkeypatch):
    monkeypatch.delenv("HARBOR_ENV_TYPE")
    with pytest.raises(ValueError, match="HARBOR_ENV_TYPE"):
        haf.build_trial_config({"instance_id": "task-1"}, "http://s/v1", {})


def test_mini_swe_agent_binding_hands_the_session_url_through_openai_env(tasks_dir, monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_NAME", "glm")
    cfg = haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": "mini-swe-agent", "max_seq_len": 4096},
        "http://s/v1",
        {"temperature": 0.8},
    )
    assert cfg.agent.name == "mini-swe-agent"
    assert cfg.agent.model_name == "openai/glm"
    assert cfg.agent.env["OPENAI_API_BASE"] == "http://s/v1"
    assert cfg.agent.env["MSWEA_COST_TRACKING"] == "ignore_errors"
    assert cfg.agent.kwargs["max_seq_len"] == 4096
    assert cfg.agent.kwargs["model_info"]["max_output_tokens"] == 8192
    assert cfg.task.path.name == "task-1"


def test_terminus_2_binding_aborts_on_truncation_and_carries_sampling_params(tasks_dir, monkeypatch):
    monkeypatch.delenv("HARBOR_RESPONSE_LENGTH_POLICY", raising=False)
    cfg = haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": "terminus-2"}, "http://s/v1", {"max_tokens": 512}
    )
    assert cfg.agent.kwargs["response_length_exceeded_policy"] == "abort"
    assert cfg.agent.kwargs["llm_call_kwargs"] == {"max_tokens": 512}
    assert cfg.agent.kwargs["api_base"] == "http://s/v1"
    assert cfg.agent.env == {"OPENAI_API_KEY": "dummy", "OPENAI_API_BASE": "http://s/v1"}


def test_claude_code_binding_uses_anthropic_env(tasks_dir, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_OUTPUT_TOKENS", "4096")
    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "claude-code"}, "http://s/v1", {})
    assert cfg.agent.env["ANTHROPIC_BASE_URL"] == "http://s/v1"
    assert cfg.agent.env["ENABLE_TOOL_SEARCH"] == "false"
    assert cfg.agent.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "4096"
    assert cfg.agent.kwargs["disallowed_tools"] == "WebSearch,WebFetch"


@pytest.mark.parametrize("bad", ["", "../etc", "no-such-task"])
def test_instance_id_must_name_a_task_dir(tasks_dir, bad):
    with pytest.raises((ValueError, FileNotFoundError)):
        haf.build_trial_config({"instance_id": bad}, "http://s/v1", {})


# --- result mapping --------------------------------------------------------


def test_verdict_maps_reward_metrics_and_timings():
    out = haf.trial_result_to_metadata(_verdict(reward=1.0))
    assert out["reward"] == 1.0
    assert out["exit_status"] == "Submitted"
    assert out["eval_report"] == {"reward": 1.0}
    m = out["agent_metrics"]
    assert m["turns"] == 7 and m["tool_calls"] == 5 and m["n_input_tokens"] == 1000
    assert m["total_time"] == 90.0 and m["env_setup_time"] == 10.0 and m["eval_time"] == 10.0
    assert "agent_setup_time" not in m


@pytest.mark.parametrize(
    ("exc_type", "exit_status"),
    [
        ("AgentTimeoutError", "TimeLimitExceeded"),
        ("EnvironmentStartTimeoutError", "TimeLimitExceeded"),
        ("SingleTurnMaxSeqLenExceededError", "SequenceLengthLimitExceeded"),
        ("RuntimeError", "AgentError"),
    ],
)
def test_harbor_exceptions_map_to_the_exit_status_vocabulary(exc_type, exit_status):
    result = SimpleNamespace(
        exception_info=SimpleNamespace(exception_type=exc_type), verifier_result=None, agent_result=None
    )
    out = haf.trial_result_to_metadata(result)
    assert out["reward"] == 0.0
    assert out["exit_status"] == exit_status


# --- entry -----------------------------------------------------------------


def test_run_returns_the_verdict_and_trial_dir(tasks_dir, fake_harbor, monkeypatch):
    fake_harbor.result = _verdict(reward=1.0)
    monkeypatch.setenv("MILES_ROUTER_EXTERNAL_HOST", "trainer.tailnet")

    out = run_async(
        haf.run(
            "http://10.0.0.1:30000/sessions/s1",
            [],
            {"temperature": 0.8},
            {"instance_id": "task-1", "agent_name": "mini-swe-agent"},
        )
    )

    assert out["reward"] == 1.0 and out["exit_status"] == "Submitted"
    assert out["trial_dir"].endswith("task-1")
    (trial,) = fake_harbor.created
    # in-sandbox agents call the model from inside the sandbox: the external host must be in the URL they get
    assert trial.config.agent.env["OPENAI_API_BASE"] == "http://trainer.tailnet:30000/sessions/s1/v1"


def test_run_scores_a_timeout_zero(tasks_dir, fake_harbor, monkeypatch):
    """The trial may still be running (the policy may be what stalls it): a negative sample, not a discard."""
    fake_harbor.result = _verdict()
    fake_harbor.run_delay_s = 10
    monkeypatch.setenv("AGENT_TRIAL_TIMEOUT", "0")

    out = run_async(haf.run("http://s/sessions/s1", [], {}, {"instance_id": "task-1"}))
    assert out == {"reward": 0.0, "exit_status": "TimeLimitExceeded", "eval_report": {}, "agent_metrics": {}}


def test_run_scores_a_trial_exception_zero(tasks_dir, fake_harbor):
    fake_harbor.result = RuntimeError("sandbox exploded")
    out = run_async(haf.run("http://s/sessions/s1", [], {}, {"instance_id": "task-1"}))
    assert out["reward"] == 0.0 and out["exit_status"] == "AgentError"


def test_run_scores_a_missing_task_zero(tasks_dir, fake_harbor):
    out = run_async(haf.run("http://s/sessions/s1", [], {}, {"instance_id": "missing"}))
    assert out["reward"] == 0.0 and out["exit_status"] == "AgentError"
    assert fake_harbor.created == []
