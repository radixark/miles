"""The agent function's contract with the Harbor agent server (no network).

This is the ``--custom-agent-function-path`` half of an agentic rollout: it runs
once per trial, and everything it gets wrong is expensive and quiet. A client
rebuilt per call leaks a connection pool per trial (#2545 is what that costs on
a 16-node run). A raised exception where the caller expects ``None`` fails a
whole batch instead of one sample. A request body missing ``session_server_id``
sends the agent at the wrong session server.

The example directory name is not a Python identifier, so the module is loaded
by path the same way the sibling metrics test does it.
"""

import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
AGENT_FUNCTION_SCRIPT = REPO_ROOT / "examples" / "swe-agent-harbor-docker" / "swe_agent_function.py"

_ENV_VARS = (
    "AGENT_SERVER_URL",
    "SWE_AGENT_URL",
    "AGENT_MODEL_NAME",
    "SWE_AGENT_MODEL_NAME",
    "AGENT_TRIAL_TIMEOUT",
    "MILES_ROUTER_EXTERNAL_HOST",
    "HARBOR_ADMIN_SECRET",
)


@pytest.fixture(scope="module")
def agent_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("swe_agent_harbor_docker_agent_function", AGENT_FUNCTION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every knob this module reads comes from the environment, so a developer's
    exported override must not decide what the assertions see."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def captured_request(agent_module, monkeypatch) -> dict:
    """Capture the (url, payload) that would go to the agent server."""
    captured: dict = {}

    async def fake_post(url, payload):
        captured["url"] = url
        captured["payload"] = payload
        return {}

    monkeypatch.setattr(agent_module, "_post_agent_server", fake_post)
    return captured


def run(agent_module, **overrides):
    call = {"base_url": "http://pod:30000/sessions/s1", "prompt": "ignored", "metadata": {}}
    call.update(overrides)
    return asyncio.run(agent_module.run(**call))


# --- the process's one client ------------------------------------------------


def test_the_agent_server_client_is_built_once_per_process(agent_module, monkeypatch):
    """A client per call leaks its connection pool: httpx owns one, nothing closes
    it, and this runs once per trial. Compare #2545 on the Daytona side."""
    built = []

    class _FakeClient:
        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setattr(agent_module.httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(agent_module, "_agent_server_client", None)

    clients = [agent_module._get_agent_server_client() for _ in range(5)]

    assert len(built) == 1, "the agent-server client was rebuilt per call"
    assert all(client is clients[0] for client in clients)


def test_the_client_imposes_no_deadline_of_its_own(agent_module, monkeypatch):
    """A trial runs for hours with nothing on the wire. The per-trial ceiling is
    asyncio.wait_for below; an HTTP read timeout here would cut the trial short
    long before it, and report the loss as a transport failure."""
    monkeypatch.setattr(agent_module, "_agent_server_client", None)

    client = agent_module._get_agent_server_client()
    try:
        assert client.timeout.read is None
        assert client.timeout.pool is None
    finally:
        asyncio.run(client.aclose())


# --- the trial timeout -------------------------------------------------------


def test_trial_timeout_falls_back_to_the_backstop(agent_module):
    assert agent_module._agent_trial_timeout_s() == agent_module._DEFAULT_AGENT_TRIAL_TIMEOUT_S


def test_trial_timeout_reads_its_env_knob(agent_module, monkeypatch):
    monkeypatch.setenv("AGENT_TRIAL_TIMEOUT", "900")
    assert agent_module._agent_trial_timeout_s() == 900


# --- what the agent server is asked to run -----------------------------------


def test_request_carries_the_session_url_the_agent_must_call(agent_module, captured_request):
    run(
        agent_module,
        request_kwargs={"temperature": 0.8},
        metadata={"instance_id": "django__django-10973", "session_server_id": "pod:30000"},
    )

    assert captured_request["url"] == "http://localhost:11000/run"
    payload = captured_request["payload"]
    assert payload["base_url"] == "http://pod:30000/sessions/s1/v1"  # OpenAI-shaped, per session
    assert payload["model"] == "openai/model"
    assert payload["sampling_params"] == {"temperature": 0.8}
    assert payload["instance_id"] == "django__django-10973"  # metadata rides at the top level
    assert payload["session_server_id"] == "pod:30000"


def test_agent_server_url_and_model_take_their_env_overrides(agent_module, captured_request, monkeypatch):
    monkeypatch.setenv("AGENT_SERVER_URL", "http://agent-server:8080")
    monkeypatch.setenv("AGENT_MODEL_NAME", "glm-5.2")

    run(agent_module)

    assert captured_request["url"] == "http://agent-server:8080/run"
    assert captured_request["payload"]["model"] == "openai/glm-5.2"


def test_the_legacy_swe_agent_env_names_still_work(agent_module, captured_request, monkeypatch):
    monkeypatch.setenv("SWE_AGENT_URL", "http://legacy:9000")
    monkeypatch.setenv("SWE_AGENT_MODEL_NAME", "legacy-model")

    run(agent_module)

    assert captured_request["url"] == "http://legacy:9000/run"
    assert captured_request["payload"]["model"] == "openai/legacy-model"


def test_external_host_rewrites_both_ways_back_to_the_trainer(agent_module, captured_request, monkeypatch):
    """The agent server reaches the trainer over a different network than the pod
    hostname resolves on, so both routes home are rewritten while keeping ports."""
    monkeypatch.setenv("MILES_ROUTER_EXTERNAL_HOST", "100.64.0.7")

    run(agent_module, metadata={"session_server_id": "pod-hostname:30000"})

    payload = captured_request["payload"]
    assert payload["base_url"] == "http://100.64.0.7:30000/sessions/s1/v1"
    assert payload["session_server_id"] == "100.64.0.7:30000"


def test_optional_fields_are_omitted_rather_than_sent_empty(agent_module, captured_request):
    run(agent_module, metadata={})

    payload = captured_request["payload"]
    assert "max_seq_len" not in payload
    assert "session_server_id" not in payload
    assert "session_server_instance_id" not in payload


def test_max_seq_len_reaches_the_agent_as_an_int(agent_module, captured_request):
    """It arrives from sample metadata, where it may be a string."""
    run(agent_module, metadata={"max_seq_len": "8192"})

    assert captured_request["payload"]["max_seq_len"] == 8192


# --- what comes back ---------------------------------------------------------


def test_the_reward_models_keys_are_always_present(agent_module, monkeypatch):
    """The returned dict is merged into sample.metadata for --custom-rm-path, so a
    server that answers with less must not make those keys disappear."""

    async def sparse_response(url, payload):
        return {"reward": 1.0}

    monkeypatch.setattr(agent_module, "_post_agent_server", sparse_response)

    assert run(agent_module) == {"reward": 1.0, "exit_status": "", "eval_report": {}, "agent_metrics": {}}


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("connection refused"), asyncio.TimeoutError(), asyncio.CancelledError()],
    ids=["transport", "timeout", "cancelled"],
)
def test_a_failed_trial_returns_none_instead_of_raising(agent_module, monkeypatch, failure):
    """One unreachable agent server must cost one sample, not the batch: the
    generate layer drops a None, but an exception propagates into the rollout."""

    async def failing_post(url, payload):
        raise failure

    monkeypatch.setattr(agent_module, "_post_agent_server", failing_post)

    assert run(agent_module) is None


# --- releasing the containers of an aborted batch ----------------------------


def test_abort_flushes_the_agent_servers_in_flight_trials(agent_module, monkeypatch):
    """Oversampling aborts SGLang mid-trial; without this flush the trials keep
    looping until their own timeout, holding their containers the whole time."""
    flushed = {}

    async def fake_post(url, payload, **kwargs):
        flushed["url"] = url
        flushed["payload"] = payload
        flushed["headers"] = kwargs.get("headers")
        return {}

    monkeypatch.setattr(agent_module, "post", fake_post)
    monkeypatch.setenv("AGENT_SERVER_URL", "http://agent-server:8080/")
    monkeypatch.setenv("HARBOR_ADMIN_SECRET", "s3cret")

    asyncio.run(agent_module.abort(SimpleNamespace(session_server_instance_id="pod:30000")))

    assert flushed["url"] == "http://agent-server:8080/flush"  # trailing slash not doubled
    assert flushed["payload"] == {"session_server_instance_id": "pod:30000"}
    assert flushed["headers"] == {"Authorization": "Bearer s3cret"}


@pytest.mark.parametrize(
    ("env", "args"),
    [
        ({}, SimpleNamespace(session_server_instance_id="pod:30000")),
        ({"AGENT_SERVER_URL": "http://agent-server:8080"}, SimpleNamespace()),
    ],
    ids=["no-server-url", "no-instance-id"],
)
def test_abort_is_a_no_op_without_something_to_flush(agent_module, monkeypatch, env, args):
    def unreachable_post(*call_args, **call_kwargs):
        raise AssertionError("abort posted without knowing where or what to flush")

    monkeypatch.setattr(agent_module, "post", unreachable_post)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    asyncio.run(agent_module.abort(args))


def test_abort_swallows_a_flush_failure(agent_module, monkeypatch):
    """Abort runs while the batch is already being torn down; a dead agent server
    must not turn cleanup into the error the caller sees."""

    async def failing_post(url, payload, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(agent_module, "post", failing_post)
    monkeypatch.setenv("AGENT_SERVER_URL", "http://agent-server:8080")

    asyncio.run(agent_module.abort(SimpleNamespace(session_server_instance_id="pod:30000")))
