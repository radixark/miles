"""Offline tests for the in-process Harbor agent function (no Harbor, no sandbox).

The ``harbor`` and ``e2b`` packages are faked in sys.modules (module-local
autouse fixtures), so these tests run where Harbor is not installed. The
contract against the REAL package lives in test_harbor_contract.py.
"""

import asyncio
import enum
import os
import sys
import types
from datetime import datetime, timedelta
from types import SimpleNamespace

import harbor_agent_function as haf
import pytest


class _EnvironmentType(str, enum.Enum):
    DOCKER = "docker"
    DAYTONA = "daytona"
    E2B = "e2b"
    AGENTENV = "agentenv"
    MODAL = "modal"
    NOVITA = "novita"


def _record(name):
    def ctor(**kwargs):
        return SimpleNamespace(_kind=name, **kwargs)

    return ctor


class FakeTrial:
    """Records the config it was created with; ``run`` returns a scripted result."""

    created: list = []
    result = None
    run_delay_s = 0.0

    def __init__(self, config):
        self.config = config
        self.paths = SimpleNamespace(trial_dir=f"/tmp/harbor_trials/{config.task.path.name}")

    @classmethod
    async def create(cls, config):
        trial = cls(config)
        cls.created.append(trial)
        return trial

    async def run(self):
        import asyncio

        if self.run_delay_s:
            await asyncio.sleep(self.run_delay_s)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture(autouse=True)
def fake_harbor(monkeypatch):
    harbor = types.ModuleType("harbor")
    models = types.ModuleType("harbor.models")
    env_type = types.ModuleType("harbor.models.environment_type")
    env_type.EnvironmentType = _EnvironmentType
    trial_models = types.ModuleType("harbor.models.trial")
    config = types.ModuleType("harbor.models.trial.config")
    for name in ("AgentConfig", "TaskConfig", "TrialConfig", "VerifierConfig", "EnvironmentConfig"):
        setattr(config, name, _record(name))
    trial_pkg = types.ModuleType("harbor.trial")
    trial_mod = types.ModuleType("harbor.trial.trial")
    trial_mod.Trial = FakeTrial
    for mod in (harbor, models, env_type, trial_models, config, trial_pkg, trial_mod):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    FakeTrial.created = []
    FakeTrial.result = None
    FakeTrial.run_delay_s = 0.0
    yield FakeTrial


class FakeTemplate:
    """Answers the E2B template lookups the preflight sends; ``error`` makes them fail."""

    looked_up: list = []
    error = None

    @classmethod
    def exists(cls, name):
        cls.looked_up.append(name)
        if cls.error is not None:
            raise cls.error
        return False


class _AuthenticationException(Exception):
    pass


@pytest.fixture(autouse=True)
def fake_e2b(monkeypatch):
    e2b = types.ModuleType("e2b")
    e2b.Template = FakeTemplate
    e2b.ConnectionConfig = lambda: SimpleNamespace(api_url=os.getenv("E2B_API_URL", "https://api.e2b.app"))
    monkeypatch.setitem(sys.modules, "e2b", e2b)
    FakeTemplate.looked_up = []
    FakeTemplate.error = None
    yield FakeTemplate


def run_async(coro):
    return asyncio.run(coro)


@pytest.fixture
def tasks_dir(tmp_path, monkeypatch):
    (tmp_path / "task-1").mkdir()
    monkeypatch.setenv("HARBOR_TASKS_DIR", str(tmp_path))
    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2b")
    # so tests never read a real key file from the developer's machine
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    monkeypatch.delenv("HARBOR_ENV_KWARGS", raising=False)
    monkeypatch.delenv("HARBOR_CPU_ENFORCEMENT_POLICY", raising=False)
    monkeypatch.delenv("HARBOR_MEMORY_ENFORCEMENT_POLICY", raising=False)
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
            **agent_fields,
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
    monkeypatch.setenv("HARBOR_ENV_TYPE", "modal")
    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"passthrough_probe": 1}')
    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})
    assert cfg.environment.type.value == "modal"
    assert cfg.environment.kwargs == {"passthrough_probe": 1}
    assert cfg.environment.delete is True


def test_daytona_reclaim_timer_outlasts_the_trial_cap(tasks_dir, monkeypatch):
    """Harbor's Daytona defaults never reclaim a sandbox a killed worker left
    behind; ours must, without ever stopping a live trial."""
    monkeypatch.setenv("HARBOR_ENV_TYPE", "daytona")
    monkeypatch.setenv("AGENT_TRIAL_TIMEOUT", "1200")  # 20 minutes
    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})
    assert cfg.environment.kwargs == {"auto_stop_interval_mins": 50, "auto_delete_interval_mins": 1440}

    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"auto_stop_interval_mins": 45, "auto_snapshot": true}')
    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})
    assert cfg.environment.kwargs == {
        "auto_stop_interval_mins": 45,  # the caller's value wins when it is safe
        "auto_delete_interval_mins": 1440,
        "auto_snapshot": True,
    }

    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"auto_stop_interval_mins": 20}')
    with pytest.raises(ValueError, match="mid-trial"):
        haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})


@pytest.mark.parametrize("env_type", ["e2b", "agentenv"])
def test_e2b_max_lifetime_outlasts_the_trial_cap(tasks_dir, monkeypatch, env_type):
    """A sandbox Harbor keeps renewing must still end soon after the longest
    trial could, and never while a trial runs."""
    monkeypatch.setenv("HARBOR_ENV_TYPE", env_type)
    monkeypatch.setenv("AGENT_TRIAL_TIMEOUT", "1200")
    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})
    assert cfg.environment.kwargs == {"max_lifetime_secs": 3000}  # and none of Daytona's reclaim timers

    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"max_lifetime_secs": 2400}')
    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})
    assert cfg.environment.kwargs == {"max_lifetime_secs": 2400}  # the caller's value wins when it is safe

    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"max_lifetime_secs": 1200}')
    with pytest.raises(ValueError, match="mid-trial"):
        haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})


@pytest.mark.parametrize(
    "var, field, value",
    [
        ("HARBOR_OVERRIDE_CPUS", "override_cpus", 2),
        ("HARBOR_OVERRIDE_MEMORY_MB", "override_memory_mb", 20480),
        ("HARBOR_OVERRIDE_STORAGE_MB", "override_storage_mb", 10240),
    ],
)
def test_resource_overrides_reach_harbor_and_reject_nonpositive(tasks_dir, monkeypatch, var, field, value):
    monkeypatch.setenv(var, str(value))
    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})
    assert getattr(cfg.environment, field) == value

    monkeypatch.setenv(var, "0")
    with pytest.raises(ValueError, match=var):
        haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})


@pytest.mark.parametrize(
    "var, field",
    [
        ("HARBOR_CPU_ENFORCEMENT_POLICY", "cpu_enforcement_policy"),
        ("HARBOR_MEMORY_ENFORCEMENT_POLICY", "memory_enforcement_policy"),
    ],
)
def test_resource_enforcement_policies_reach_harbor(tasks_dir, monkeypatch, var, field):
    monkeypatch.setenv(var, "request")

    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})

    assert getattr(cfg.environment, field) == "request"


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
    # no top-level api_key parameter on Terminus2; litellm gets it per call via llm_kwargs
    assert cfg.agent.kwargs["llm_kwargs"] == {"api_key": "dummy"}
    assert "api_key" not in cfg.agent.kwargs
    assert cfg.agent.env == {"OPENAI_API_KEY": "dummy", "OPENAI_API_BASE": "http://s/v1"}


def test_claude_code_binding_uses_anthropic_env(tasks_dir, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_OUTPUT_TOKENS", "4096")
    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "claude-code"}, "http://s/sess/v1", {})
    # the session root: the SDK appends /v1/messages, the server route is
    # /sessions/{id}/v1/messages -- a /v1-suffixed base would 404 on /v1/v1/messages
    assert cfg.agent.env["ANTHROPIC_BASE_URL"] == "http://s/sess"
    assert cfg.agent.env["ENABLE_TOOL_SEARCH"] == "false"
    assert cfg.agent.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "4096"
    assert cfg.agent.kwargs["disallowed_tools"] == "WebSearch,WebFetch"


def test_opencode_binding_remaps_the_openai_provider(tasks_dir, monkeypatch):
    """OpenCode resolves provider id "openai" to the Responses API, which the session server rejects."""
    monkeypatch.setenv("AGENT_MODEL_NAME", "glm")
    monkeypatch.delenv("AGENT_MAX_OUTPUT_TOKENS", raising=False)
    cfg = haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": "opencode", "max_seq_len": 65536}, "http://s/v1", {"max_tokens": 512}
    )
    assert cfg.agent.model_name == "openai-compatible/glm"
    provider = cfg.agent.kwargs["opencode_config"]["provider"]
    entry = provider["openai-compatible"]
    assert entry["npm"] == "@ai-sdk/openai-compatible"
    assert entry["options"]["baseURL"] == "http://s/v1"
    # limits: without them OpenCode never auto-compacts a custom-provider model
    assert entry["models"]["glm"] == {"limit": {"context": 65536, "output": 512}}
    # both URL vars: OpenCode consults OPENAI_BASE_URL, litellm-style agents OPENAI_API_BASE
    assert cfg.agent.env["OPENAI_BASE_URL"] == "http://s/v1"
    assert cfg.agent.env["OPENAI_API_BASE"] == "http://s/v1"
    # no turn cap: OpenCode has no flag for it, a kwarg would be silently discarded
    assert "max_turns" not in cfg.agent.kwargs


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/glm", ("openai-compatible", "glm")),
        ("bare-model", ("openai-compatible", "bare-model")),
        ("anthropic/claude", ("anthropic", "claude")),
    ],
)
def test_opencode_provider_split(model, expected):
    assert haf._opencode_provider_model(model) == expected


@pytest.mark.parametrize("bad", ["", "../etc", "no-such-task"])
def test_instance_id_must_name_a_task_dir(tasks_dir, bad):
    with pytest.raises((ValueError, FileNotFoundError)):
        haf.build_trial_config({"instance_id": bad}, "http://s/v1", {})


def test_task_dir_errors_name_every_instance_id_a_trial_would_reject(tasks_dir):
    """The launcher reports these before the run, in the words a trial would raise them."""
    errors = haf.task_dir_errors(tasks_dir, ["task-1", "", "../etc", "no-such-task"])
    assert errors == [
        "invalid instance_id ''",
        "invalid instance_id '../etc'",
        f"no Harbor task dir for 'no-such-task' under {tasks_dir}",
    ]


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


def test_run_returns_the_verdict_and_trial_dir(tasks_dir, fake_harbor):
    fake_harbor.result = _verdict(reward=1.0)

    out = run_async(
        haf.run(
            "http://trainer.tailnet:30000/sessions/s1",
            [],
            {"temperature": 0.8},
            {"instance_id": "task-1", "agent_name": "mini-swe-agent"},
        )
    )

    assert out["reward"] == 1.0 and out["exit_status"] == "Submitted"
    assert out["trial_dir"].endswith("task-1")
    (trial,) = fake_harbor.created
    # in-sandbox agents call the model from inside the sandbox: base_url already names the
    # instance the way the sandbox reaches it, so it goes through with only /v1 appended
    assert trial.config.agent.env["OPENAI_API_BASE"] == "http://trainer.tailnet:30000/sessions/s1/v1"


def test_run_scores_a_timeout_zero(tasks_dir, fake_harbor, monkeypatch):
    """The trial may still be running (the policy may be what stalls it): a negative sample, not a discard."""
    fake_harbor.result = _verdict()
    fake_harbor.run_delay_s = 10
    monkeypatch.setenv("AGENT_TRIAL_TIMEOUT", "0")

    out = run_async(haf.run("http://s/sessions/s1", [], {}, {"instance_id": "task-1"}))
    assert out == {"reward": 0.0, "exit_status": "TimeLimitExceeded", "eval_report": {}, "agent_metrics": {}}


@pytest.mark.parametrize("env_type", ["e2b", "agentenv"])
def test_run_resolves_the_provider_key_file_into_the_sdk_env_var(
    tasks_dir, fake_harbor, tmp_path, monkeypatch, env_type
):
    """The launcher forwards the key by PATH; the SDK reads its own env var, so
    the worker must resolve the file -- the gap the first live e2e run hit."""
    monkeypatch.setenv("HARBOR_ENV_TYPE", env_type)
    key_file = tmp_path / "api_key"
    key_file.write_text("e2b_test_key\n")
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setenv("E2B_API_KEY_FILE", str(key_file))
    fake_harbor.result = _verdict()

    run_async(haf.run("http://s/sessions/s1", [], {}, {"instance_id": "task-1"}))
    assert os.environ["E2B_API_KEY"] == "e2b_test_key"


def test_run_raises_when_no_provider_key_is_resolvable(tasks_dir, fake_harbor, monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setenv("E2B_API_KEY_FILE", "/nonexistent/api_key")
    with pytest.raises(RuntimeError, match="no API key"):
        run_async(haf.run("http://s/sessions/s1", [], {}, {"instance_id": "task-1"}))
    assert fake_harbor.created == []


def test_run_scores_a_trial_exception_zero(tasks_dir, fake_harbor):
    fake_harbor.result = RuntimeError("sandbox exploded")
    out = run_async(haf.run("http://s/sessions/s1", [], {}, {"instance_id": "task-1"}))
    assert out["reward"] == 0.0 and out["exit_status"] == "AgentError"


def test_run_raises_on_a_missing_task_instead_of_scoring_zero(tasks_dir, fake_harbor):
    """A missing task dir is not the policy's doing: raising drops the sample instead of training on a zero reward."""
    with pytest.raises(FileNotFoundError):
        run_async(haf.run("http://s/sessions/s1", [], {}, {"instance_id": "missing"}))
    assert fake_harbor.created == []


@pytest.mark.parametrize("env_type", ["e2b", "agentenv"])
def test_preflight_passes_a_working_setup_without_running_a_trial(
    tasks_dir, fake_harbor, fake_aenv, fake_e2b, monkeypatch, env_type
):
    fake_aenv("aenv 0.2.5")
    monkeypatch.setenv("HARBOR_ENV_TYPE", env_type)
    run_async(haf.preflight(None))
    assert fake_harbor.created == []
    assert len(fake_e2b.looked_up) == 1


@pytest.mark.parametrize("env_type", ["e2b", "agentenv"])
def test_preflight_fails_when_the_e2b_api_refuses_the_request(tasks_dir, fake_aenv, fake_e2b, monkeypatch, env_type):
    """A wrong endpoint answers 401 just like a bad key, so the error must name the endpoint."""
    fake_aenv("aenv 0.2.5")
    monkeypatch.setenv("HARBOR_ENV_TYPE", env_type)
    monkeypatch.setenv("E2B_API_URL", "https://wrong.example")
    fake_e2b.error = _AuthenticationException("401: Unauthorized, please check your credentials.")
    with pytest.raises(RuntimeError, match="https://wrong.example.*401: Unauthorized"):
        run_async(haf.preflight(None))


def test_preflight_sends_no_e2b_request_for_other_backends(tasks_dir, fake_e2b, monkeypatch):
    """Other backends have no E2B key, so asking the E2B API would fail a working setup."""
    monkeypatch.setenv("HARBOR_ENV_TYPE", "docker")
    run_async(haf.preflight(None))
    assert fake_e2b.looked_up == []


@pytest.mark.parametrize(
    "env, error",
    [
        ({"HARBOR_TASKS_DIR": None}, "set HARBOR_TASKS_DIR"),
        ({"HARBOR_TASKS_DIR": "/nonexistent/tasks"}, "not a directory on this host"),
        ({"E2B_API_KEY": None, "E2B_API_KEY_FILE": "/nonexistent/api_key"}, "no API key"),
        ({"HARBOR_ENV_TYPE": "no-such-backend"}, "no-such-backend"),
        ({"HARBOR_ENV_KWARGS": "{not json"}, "Expecting property name"),
        # read only by the harnesses with a turn cap, so only a check under each harness catches it
        ({"HARBOR_AGENT_MAX_ITERATIONS": "ten"}, "invalid literal"),
    ],
)
def test_preflight_fails_a_setup_every_trial_would_fail_on(tasks_dir, monkeypatch, env, error):
    for var, value in env.items():
        if value is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, value)
    with pytest.raises(Exception, match=error):
        run_async(haf.preflight(None))


def test_preflight_fails_a_tasks_dir_with_no_task_in_it(tasks_dir, tmp_path, monkeypatch):
    """A wrong path or an unfinished clone leaves every row without its task."""
    clone = tmp_path / "unfinished-clone"
    (clone / ".git").mkdir(parents=True)
    monkeypatch.setenv("HARBOR_TASKS_DIR", str(clone))
    with pytest.raises(FileNotFoundError, match="no Harbor task dir"):
        run_async(haf.preflight(None))


def test_preflight_fails_agentenv_without_aenv(tasks_dir, fake_aenv, monkeypatch):
    monkeypatch.setenv("HARBOR_ENV_TYPE", "agentenv")
    with pytest.raises(RuntimeError, match="aenv is not on PATH"):
        run_async(haf.preflight(None))


def test_preflight_fails_when_harbor_cannot_be_imported(tasks_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "harbor.trial.trial", None)
    with pytest.raises(ImportError):
        run_async(haf.preflight(None))
