import pytest
import requests
from fastapi.responses import JSONResponse
from tests.fast.router.test_sessions import _create_session, _post_chat
from tests.fast.router.test_sessions import _serve_router as _serve_v1
from tests.fast.router.test_sessions_v2 import _serve_router as _serve_v2

from miles.rollout.session.v2 import tree_trajectory
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType, resolve_fixed_chat_template

USER = {"role": "user", "content": "hi"}
QWEN35 = {
    "hf_checkpoint": "Qwen/Qwen3.5-0.8B",
    "tito_model": TITOTokenizerType.QWEN35.value,
    "chat_template_path": resolve_fixed_chat_template(TITOTokenizerType.QWEN35)[0],
}


def _serve(version: str, extra_args: dict | None = None):
    return (_serve_v1 if version == "v1" else _serve_v2)(extra_args)


@pytest.fixture(params=["v1", "v2"])
def env(request):
    with _serve(request.param) as served:
        yield served


def _records(url: str, session_id: str) -> list:
    return requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()["records"]


def test_generating_route_outside_the_recorded_ones_is_refused(env):
    calls = []

    @env.backend.app.post("/v1/responses")
    async def responses():
        calls.append(1)
        return JSONResponse(content={"id": "resp_1", "output": [{"type": "message"}]})

    session_id = _create_session(env.url)
    response = requests.post(
        f"{env.url}/sessions/{session_id}/v1/responses", json={"model": "m", "input": "hi"}, timeout=5.0
    )

    assert response.status_code == 404
    error = response.json()["error"]
    assert error.startswith("POST /v1/responses is not served by the session server")
    assert "POST /v1/chat/completions, POST /v1/messages" in error
    assert "GET /v1/models" in error
    assert calls == []
    assert _records(env.url, session_id) == []


def test_read_only_routes_are_still_forwarded(env):
    @env.backend.app.get("/v1/models")
    async def models():
        return JSONResponse(content={"object": "list", "data": [{"id": "m"}]})

    session_id = _create_session(env.url)
    response = requests.get(f"{env.url}/sessions/{session_id}/v1/models", timeout=5.0)

    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": [{"id": "m"}]}


def test_more_than_one_choice_is_refused(env):
    session_id = _create_session(env.url)
    response = _post_chat(env.url, session_id, {"messages": [USER], "n": 2})

    assert response.status_code == 400
    assert "n=2" in response.json()["error"]
    assert env.backend.request_log == []
    assert _records(env.url, session_id) == []


def test_request_the_template_cannot_render_is_a_client_error(env):
    session_id = _create_session(env.url)
    system = {"role": "system", "content": [{"type": "text", "text": "Be brief."}]}
    response = _post_chat(env.url, session_id, {"messages": [system, USER]})

    assert response.status_code == 400
    assert response.json()["error"].startswith("the chat template cannot render this request")
    assert env.backend.request_log == []


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_request_the_template_refuses_is_a_client_error(version):
    with _serve(version, QWEN35) as env:
        session_id = _create_session(env.url)
        late_system = {"role": "system", "content": "Be brief."}
        response = _post_chat(env.url, session_id, {"messages": [USER, late_system]})

        assert response.status_code == 400
        assert "System message must be at the beginning." in response.json()["error"]
        assert env.backend.request_log == []


def test_node_cap_is_refused_before_generating(monkeypatch):
    monkeypatch.setattr(tree_trajectory, "MAX_NODES", 2)
    with _serve_v2() as env:
        session_id = _create_session(env.url)
        for text in ("one", "two"):
            accepted = _post_chat(env.url, session_id, {"messages": [{"role": "user", "content": text}]})
            assert accepted.status_code == 200

        response = _post_chat(env.url, session_id, {"messages": [{"role": "user", "content": "three"}]})

        assert response.status_code == 400
        assert response.json()["error"].startswith("node cap reached (2)")
        assert len(env.backend.request_log) == 2
