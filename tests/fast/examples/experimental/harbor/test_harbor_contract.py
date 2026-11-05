"""Contract tests against the REAL harbor package: what the agent function
builds must be something Harbor's own models accept.

Skipped wherever harbor is not installed (CPU CI today); they run in any
environment provisioned per the example README. They catch the drift the
faked-package unit tests cannot: an EnvironmentType value or AgentName that
stopped existing, a TrialConfig/AgentConfig field that moved. (Pydantic's
default extra="ignore" means a RENAMED optional field can still slip through;
required-field and enum drift is caught.)
"""

import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

harbor = pytest.importorskip("harbor")

import harbor_agent_function as haf  # noqa: E402
from harbor.agents.computer_1.computer_1 import Computer1, Computer1Chat, EpisodeLoggingPaths  # noqa: E402
from harbor.agents.computer_1.providers.base import image_url_part  # noqa: E402
from harbor.agents.computer_1.providers.generic import GenericJsonProvider  # noqa: E402
from harbor.agents.computer_1.runtime import DisplayGeometry, normalize_completion_action  # noqa: E402
from harbor.agents.factory import AgentFactory  # noqa: E402
from harbor.llms.base import LLMResponse  # noqa: E402
from harbor.models.agent.name import AgentName  # noqa: E402
from harbor.models.environment_type import EnvironmentType  # noqa: E402
from harbor.models.trial.config import TrialConfig  # noqa: E402
from harbor.trial.trial import Trial  # noqa: E402

from miles.utils.media_refs import collect_media_refs  # noqa: E402

_SESSION_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": 512}


@pytest.fixture
def tasks_dir(tmp_path, monkeypatch):
    (tmp_path / "task-1").mkdir()
    monkeypatch.setenv("HARBOR_TASKS_DIR", str(tmp_path))
    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2b")
    return tmp_path


@pytest.fixture
def chat_server():
    requests: list[tuple[str, dict, dict]] = []
    reply = json.dumps(
        {
            "id": "r",
            "object": "chat.completion",
            "created": 0,
            "model": "model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"analysis": "", "plan": "", "action": {"type": "wait"}}',
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", requests
    server.shutdown()


def _computer_1(session_url: str, logs_dir, monkeypatch):
    monkeypatch.setenv("HARBOR_AGENT_IMPORT_PATH", f"{Computer1.__module__}:{Computer1.__name__}")
    cfg = haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": "computer-1", "max_seq_len": 4096}, session_url, _SESSION_SAMPLING
    )
    return AgentFactory.create_agent_from_config(cfg.agent, logs_dir=logs_dir)


@pytest.fixture
def full_hd_computer_1(tasks_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(GenericJsonProvider, "from_agent", GenericJsonProvider.__dict__["from_agent"])
    monkeypatch.setattr(GenericJsonProvider, "parse", GenericJsonProvider.parse)
    monkeypatch.setattr(haf, "_coordinate_space_applied", False, raising=False)
    monkeypatch.setenv("HARBOR_AGENT_KWARGS", json.dumps({"desktop_width": 1920, "desktop_height": 1080}))

    def build():
        haf.apply_coordinate_space()
        return GenericJsonProvider.from_agent(_computer_1("http://s/sessions/s1/v1", tmp_path, monkeypatch))

    return build


def _click(provider, x: int, y: int):
    reply = json.dumps({"analysis": "", "plan": "", "action": {"type": "click", "x": x, "y": y}})
    action = provider.parse(LLMResponse(content=reply)).action
    return normalize_completion_action(action, DisplayGeometry(desktop_width=1920, desktop_height=1080))


def test_the_coordinate_space_maps_grid_clicks_onto_the_desktop(full_hd_computer_1, monkeypatch):
    monkeypatch.setenv("HARBOR_CUA_COORDINATE_SPACE", "normalized_0_999")
    provider = full_hd_computer_1()

    assert "1000x1000 pixels" in provider.record_text("task")
    action = _click(provider, 999, 500)
    assert (action.x, action.y) == (1919, 540)


def test_without_a_coordinate_space_clicks_stay_desktop_pixels(full_hd_computer_1, monkeypatch):
    monkeypatch.delenv("HARBOR_CUA_COORDINATE_SPACE", raising=False)
    provider = full_hd_computer_1()

    assert "1920x1080 pixels" in provider.record_text("task")
    action = _click(provider, 999, 500)
    assert (action.x, action.y) == (999, 500)


def test_an_unknown_coordinate_space_stops_the_run(monkeypatch):
    monkeypatch.setattr(haf, "_coordinate_space_applied", False, raising=False)
    monkeypatch.setenv("HARBOR_CUA_COORDINATE_SPACE", "pixels")

    with pytest.raises(ValueError, match="normalized_0_999"):
        haf.apply_coordinate_space()


def test_harbor_builds_computer_1_from_the_import_path_with_max_seq_len_set(tasks_dir, tmp_path, monkeypatch):
    agent = _computer_1("http://trainer:30000/sessions/s1/v1", tmp_path, monkeypatch)

    assert type(agent) is Computer1
    assert agent._enable_images is True


def test_every_call_computer_1_makes_carries_miles_sampling(tasks_dir, tmp_path, monkeypatch, chat_server):
    base_url, requests = chat_server
    agent = _computer_1(f"{base_url}/sessions/s1/v1", tmp_path, monkeypatch)
    agent._chat = chat = Computer1Chat(agent._llm)
    screenshot = f"data:image/png;base64,{base64.b64encode(b'screen').decode()}"

    async def every_kind_of_call():
        prompt = [{"role": "user", "content": [{"type": "text", "text": "Begin."}, image_url_part(screenshot)]}]
        await agent._query_litellm(chat, prompt, EpisodeLoggingPaths(None, None, None))
        assert await agent._compactor._build_summary_from_history(chat, "the task") is not None
        assert await agent._compactor._build_short_summary("the task", "the screen") is not None
        await agent._litellm_extract_text_fallback("the task")

    asyncio.run(every_kind_of_call())

    assert len(requests) == 4
    for path, headers, body in requests:
        assert path == "/sessions/s1/v1/chat/completions"
        assert headers["authorization"] == "Bearer dummy"
        assert {key: body.get(key) for key in _SESSION_SAMPLING} == _SESSION_SAMPLING
    assert len(collect_media_refs(requests[0][2]["messages"], str(tmp_path / "media"))["images"]) == 1


@pytest.mark.parametrize("env_type", ["docker", "daytona", "e2b", "modal"])
def test_the_pass_through_environment_types_exist(env_type):
    assert EnvironmentType(env_type).value == env_type


@pytest.mark.parametrize("agent_name", sorted(haf.HARNESS_BINDINGS))
def test_every_binding_names_a_real_harbor_agent(agent_name):
    assert AgentName(agent_name).value == agent_name


@pytest.mark.parametrize("agent_name", [*sorted(haf.HARNESS_BINDINGS), "some-other-openai-chat-agent"])
def test_the_built_config_is_a_valid_trial_config(tasks_dir, agent_name):
    """Real pydantic validation of everything build_trial_config assembles."""
    cfg = haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": agent_name, "max_seq_len": 4096},
        "http://trainer:30000/sessions/s1/v1",
        {"temperature": 0.8, "max_tokens": 512},
    )
    assert isinstance(cfg, TrialConfig)
    assert cfg.environment.type == EnvironmentType.E2B


def test_resource_enforcement_policy_uses_harbor_model(tasks_dir, monkeypatch):
    monkeypatch.setenv("HARBOR_CPU_ENFORCEMENT_POLICY", "request")
    monkeypatch.setenv("HARBOR_MEMORY_ENFORCEMENT_POLICY", "limit")

    cfg = haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": "mini-swe-agent"},
        "http://trainer:30000/sessions/s1/v1",
        {},
    )

    assert cfg.environment.cpu_enforcement_policy.value == "request"
    assert cfg.environment.memory_enforcement_policy.value == "limit"


def test_the_trial_entrypoints_exist():
    assert callable(Trial.create) and callable(Trial.run)


def test_the_result_fields_the_mapping_reads_still_exist():
    """trial_result_to_metadata reads these via getattr(default None): a rename
    would silently drop metrics/timings, so lock the names here."""
    from harbor.models.trial.result import TrialResult

    fields = set(TrialResult.model_fields)
    assert {
        "started_at",
        "finished_at",
        "environment_setup",
        "agent_setup",
        "agent_execution",
        "verifier",
        "exception_info",
        "verifier_result",
        "agent_result",
    } <= fields
