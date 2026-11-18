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
import os
import shlex
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

harbor = pytest.importorskip("harbor")

import harbor_agent_function as haf  # noqa: E402
from harbor.agents.computer_1 import runtime  # noqa: E402
from harbor.agents.computer_1.computer_1 import Computer1, Computer1Chat, EpisodeLoggingPaths  # noqa: E402
from harbor.agents.computer_1.providers import generic  # noqa: E402
from harbor.agents.computer_1.providers.base import image_url_part  # noqa: E402
from harbor.agents.computer_1.providers.generic import GenericJsonProvider  # noqa: E402
from harbor.agents.computer_1.runtime import (  # noqa: E402
    Computer1Session,
    ComputerAction,
    DisplayGeometry,
    RuntimeRequestError,
    normalize_completion_action,
)
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


def _reply(action: dict) -> LLMResponse:
    return LLMResponse(content=json.dumps({"analysis": "", "plan": "", "action": action}))


@pytest.mark.parametrize(
    "action",
    [
        {"type": "click", "x": [499, 284]},
        {"type": "click", "x": [499], "y": [284]},
        {"type": "click", "x": [499, 284], "y": [499, 284]},
        {"type": "click", "coordinate": [499, 284]},
        {"type": "click", "x": ["499", 284.0]},
    ],
)
def test_a_point_written_as_a_list_lands_where_it_pointed(full_hd_computer_1, monkeypatch, action):
    monkeypatch.setenv("HARBOR_CUA_COORDINATE_SPACE", "normalized_0_999")
    provider = full_hd_computer_1()

    step = provider.parse(_reply(action))
    click = normalize_completion_action(step.action, DisplayGeometry(desktop_width=1920, desktop_height=1080))
    assert (click.x, click.y) == (959, 307)


@pytest.mark.parametrize(
    "action",
    [
        {"type": "drag", "x": [133, 29], "end_x": [700, 400]},
        {"type": "drag", "x": [133], "y": [29], "end_x": [700], "end_y": [400]},
    ],
)
def test_a_drag_written_as_lists_moves_between_both_points(full_hd_computer_1, monkeypatch, action):
    monkeypatch.setenv("HARBOR_CUA_COORDINATE_SPACE", "normalized_0_999")
    provider = full_hd_computer_1()

    step = provider.parse(_reply(action))
    drag = normalize_completion_action(step.action, DisplayGeometry(desktop_width=1920, desktop_height=1080))
    assert (drag.x, drag.y, drag.end_x, drag.end_y) == (255, 31, 1345, 432)


@pytest.mark.parametrize(
    ("action", "missing"),
    [
        ({"type": "click", "x": 992}, '"x" and "y"'),
        ({"type": "click"}, '"x" and "y"'),
        ({"type": "right_click", "x": [1, 2, 3]}, '"x" and "y"'),
        ({"type": "click", "x": [499, 284], "y": [500, 300]}, '"x" and "y"'),
        ({"type": "drag", "x": 133, "y": 29}, '"end_x" and "end_y"'),
        ({"type": "drag", "x": 133, "y": 29, "end_x": [1, 2, 3]}, '"end_x" and "end_y"'),
    ],
)
def test_a_pointer_action_with_no_readable_point_is_retried_not_clicked_at_the_corner(
    full_hd_computer_1, monkeypatch, action, missing
):
    monkeypatch.setenv("HARBOR_CUA_COORDINATE_SPACE", "normalized_0_999")
    provider = full_hd_computer_1()

    reply = _reply(action)
    step = provider.parse(reply)
    assert step.needs_retry and step.action is None
    assert missing in step.feedback
    assert step.llm_response is reply


def test_actions_without_a_point_and_scalar_points_parse_as_before(full_hd_computer_1, monkeypatch):
    monkeypatch.setenv("HARBOR_CUA_COORDINATE_SPACE", "normalized_0_999")
    provider = full_hd_computer_1()

    key = provider.parse(_reply({"type": "key", "keys": ["ctrl", "s"]}))
    assert not key.needs_retry and key.action.keys == ["ctrl", "s"]
    scroll = provider.parse(_reply({"type": "scroll", "scroll_y": 300}))
    assert not scroll.needs_retry and (scroll.action.x, scroll.action.y) == (None, None)
    reply = _reply({"type": "click", "x": 999, "y": 500})
    click = provider.parse(reply)
    assert click.message == reply.content and click.llm_response is reply


def test_an_unknown_coordinate_space_stops_the_run(monkeypatch):
    monkeypatch.setattr(haf, "_coordinate_space_applied", False, raising=False)
    monkeypatch.setenv("HARBOR_CUA_COORDINATE_SPACE", "pixels")

    with pytest.raises(ValueError, match="normalized_0_999"):
        haf.apply_coordinate_space()


_SESSION_PATCHED = ("start", "is_session_alive", "reset", "_recover_chromium_if_needed", "execute", "fetch_screenshot")
_DESKTOP_PATCHED = [
    *[(Computer1Session, name) for name in _SESSION_PATCHED],
    (GenericJsonProvider, "from_agent"),
    (Computer1, "_query_litellm"),
    (Computer1, "_fallback_message_history"),
]
_X_SOCKET_ALIVE = "test -S /tmp/.X11-unix/X1 && echo ok || echo down"


@pytest.fixture
def harbor_restored(monkeypatch):
    for owner, name in _DESKTOP_PATCHED:
        monkeypatch.setattr(owner, name, owner.__dict__[name])
    monkeypatch.setattr(haf, "_desktop_applied", False, raising=False)


@pytest.fixture
def desktop(harbor_restored, tasks_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("HARBOR_CUA_DESKTOP", "existing")

    def build(session_url="http://s/sessions/s1/v1", **env):
        for var, value in env.items():
            monkeypatch.setenv(var, value)
        haf.apply_existing_desktop()
        return _computer_1(session_url, tmp_path, monkeypatch)

    return build


def _recording_environment():
    commands: list[str] = []

    async def exec(command, timeout_sec=None, user=None, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="ok", stderr="", return_code=0)

    return SimpleNamespace(exec=exec, default_user=None), commands


def _started(agent):
    environment, commands = _recording_environment()
    asyncio.run(agent.setup(environment))
    started = list(commands)
    commands.clear()
    return agent._session, started, commands


def _image_turn(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}, image_url_part("data:image/png;base64,AA==")]}


def _images(messages) -> int:
    return sum(
        part.get("type") == "image_url" for m in messages if isinstance(m["content"], list) for part in m["content"]
    )


def test_on_an_existing_desktop_the_session_starts_no_desktop_or_browser(desktop):
    _, started, _ = _started(desktop())

    assert not [c for c in started for word in ("Xvfb", "startxfce4", "x11vnc", "chromium") if word in c]
    assert len(started) == 2
    assert "mkdir -p" in started[0]
    assert started[1] == "test -S /tmp/.X11-unix/X1 && echo ok || echo wait"


def test_on_an_existing_desktop_liveness_reads_only_the_x_socket(desktop):
    session, _, commands = _started(desktop())

    assert asyncio.run(session.is_session_alive()) is True
    assert commands == [_X_SOCKET_ALIVE]
    assert asyncio.run(session._recover_chromium_if_needed("click", RuntimeError())) is None
    with pytest.raises(RuntimeError, match="no browser"):
        asyncio.run(session.reset())


def test_on_an_existing_desktop_browser_actions_are_refused_and_screen_changes_settle(desktop, monkeypatch):
    session, _, _ = _started(desktop(HARBOR_CUA_SETTLE_SEC="0.25"))
    sleeps = []

    async def recorded_sleep(delay, *args, **kwargs):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", recorded_sleep)

    for action in (ComputerAction(type="navigate", url="https://example.com"), ComputerAction(type="zoom")):
        with pytest.raises(RuntimeRequestError, match="not available"):
            asyncio.run(session.execute(action))
    asyncio.run(session.execute(ComputerAction(type="click", x=5, y=5)))
    asyncio.run(session.execute(ComputerAction(type="wait")))
    assert sleeps == [0.25, 1.0]


def test_on_an_existing_desktop_the_screenshot_is_a_resized_png(desktop):
    agent = desktop(HARBOR_CUA_SCREENSHOT_SIZE="1280x720")
    session, _, commands = _started(agent)
    provider = GenericJsonProvider.from_agent(agent)

    asyncio.run(session.fetch_screenshot("/logs/agent/screenshot_1.png"))
    png = "/tmp/computer_1-screenshots/latest.png"
    assert (
        f"import -window root {png}; convert {png} -resize 1280x720! {png}; cp {png} " in shlex.split(commands[0])[2]
    )
    with pytest.raises(ValueError, match="PNG"):
        asyncio.run(session.fetch_screenshot("/logs/agent/screenshot_1.webp"))
    assert provider.screenshot_format == "png"
    assert '"navigate"' not in provider.record_text("task") and "zoom_region" not in provider.record_text("task")


@pytest.mark.parametrize("size, args", [("1280x720", " 1280 720"), ("", "")])
def test_on_an_existing_desktop_a_capture_command_takes_the_screenshot(desktop, size, args):
    capture = "/opt/env/venv/bin/python /opt/capture.py"
    session, _, commands = _started(desktop(HARBOR_CUA_SCREENSHOT_SIZE=size, HARBOR_CUA_CAPTURE_COMMAND=capture))

    asyncio.run(session.fetch_screenshot("/logs/agent/screenshot_1.png"))
    script = shlex.split(commands[0])[2]
    png = "/tmp/computer_1-screenshots/latest.png"
    assert f"; {capture} {png}{args}; cp {png} /logs/agent/screenshot_1.png" in script
    assert "import -window" not in script and "convert" not in script


@pytest.mark.parametrize(
    "env",
    [
        {"HARBOR_CUA_SCREENSHOT_SIZE": "1280x720"},
        {"HARBOR_CUA_CAPTURE_COMMAND": "grab", "HARBOR_CUA_SCREENSHOT_SIZE": ""},
    ],
)
def test_on_an_existing_desktop_a_failed_capture_never_hands_back_the_previous_screen(
    desktop, tmp_path, monkeypatch, env
):
    shots = tmp_path / "shots"
    shots.mkdir()
    (shots / "latest.png").write_bytes(b"previous screen")
    monkeypatch.setattr(runtime, "_SCREENSHOT_DIR", str(shots))
    failing = {f"BASH_FUNC_{name}%%": "() { return 1; }" for name in ("import", "grab")}

    async def bash(command, timeout_sec=None, user=None, **kwargs):
        done = subprocess.run(["bash", "-c", command], capture_output=True, text=True, env={**os.environ, **failing})
        return SimpleNamespace(stdout=done.stdout, stderr=done.stderr, return_code=done.returncode)

    agent = desktop(**env)
    agent._session = session = Computer1Session(SimpleNamespace(exec=bash, default_user=None), PurePosixPath(tmp_path))
    target = tmp_path / "agent" / "screenshot_2.png"

    with pytest.raises(RuntimeRequestError):
        asyncio.run(session.fetch_screenshot(str(target)))
    assert not target.exists()


def test_on_an_existing_desktop_a_request_carries_at_most_the_kept_screenshots(desktop, chat_server):
    base_url, requests = chat_server
    agent = desktop(f"{base_url}/sessions/s1/v1", HARBOR_CUA_KEEP_SCREENSHOTS="2")
    agent._chat = chat = Computer1Chat(agent._llm)
    for turn in range(3):
        chat.messages.extend([_image_turn(f"screen {turn}"), {"role": "assistant", "content": "click"}])

    asyncio.run(agent._query_litellm(chat, [_image_turn("now")], EpisodeLoggingPaths(None, None, None)))
    agent._latest_screenshot_path = "/logs/agent/screenshot_4.png"

    assert _images(requests[0][2]["messages"]) == 2
    assert _images(agent._fallback_message_history()) == 1


def test_without_a_desktop_harbor_is_untouched(harbor_restored, monkeypatch):
    monkeypatch.delenv("HARBOR_CUA_DESKTOP", raising=False)
    before = [owner.__dict__[name] for owner, name in _DESKTOP_PATCHED]

    haf.apply_existing_desktop()

    assert all(owner.__dict__[name] is kept for (owner, name), kept in zip(_DESKTOP_PATCHED, before, strict=True))


@pytest.mark.parametrize(
    "env, error",
    [
        ({"HARBOR_CUA_DESKTOP": "xfce"}, "existing"),
        ({"HARBOR_CUA_DESKTOP": "existing", "HARBOR_CUA_SCREENSHOT_SIZE": "1280"}, "WIDTHxHEIGHT"),
        ({"HARBOR_CUA_DESKTOP": "existing", "HARBOR_CUA_KEEP_SCREENSHOTS": "0"}, "at least 1"),
    ],
)
def test_a_bad_desktop_setting_stops_the_run(harbor_restored, monkeypatch, env, error):
    for var, value in env.items():
        monkeypatch.setenv(var, value)

    with pytest.raises(ValueError, match=error):
        haf.apply_existing_desktop()


def test_the_private_harbor_names_the_desktop_patch_touches_exist(tasks_dir, tmp_path, monkeypatch):
    agent = _computer_1("http://s/sessions/s1/v1", tmp_path, monkeypatch)
    session = Computer1Session(_recording_environment()[0], PurePosixPath("/logs/agent"))

    assert all(
        hasattr(runtime, name) for name in ("_RUNTIME_DIR", "_SCREENSHOT_DIR", "_DEFAULT_DISPLAY", "_bash_inline")
    )
    assert (generic._TEMPLATES_DIR / "computer-1-json.txt").is_file()
    assert hasattr(GenericJsonProvider.from_agent(agent), "_prompt_template")
    assert all(
        callable(getattr(session, name))
        for name in ("_recover_chromium_if_needed", "_wait_for_x11", "_exec", "_raw_exec")
    )
    assert all(hasattr(session, name) for name in ("_started", "_agent_dir", "_request_timeout_sec"))
    assert all(callable(getattr(agent, name)) for name in ("_query_litellm", "_fallback_message_history"))
    assert callable(agent._compactor._trim_old_screenshots)
    assert all(hasattr(agent, name) for name in ("_enable_images", "_latest_screenshot_path"))


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
