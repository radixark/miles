from types import SimpleNamespace

import pytest
import requests

from miles.session_server.server import create_session_server_app
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


@pytest.fixture(scope="class")
def session_server_env():
    """Start a session server in front of a mock sglang backend."""

    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

    with with_mock_server(process_fn=process_fn) as backend:
        args = SimpleNamespace(
            hf_checkpoint="Qwen/Qwen3-0.6B",
            chat_template_path=None,
            upstream_url=backend.url,
        )
        app = create_session_server_app(args)

        port = find_available_port(32000)
        server = UvicornThreadServer(app, host="127.0.0.1", port=port)
        server.start()

        url = f"http://127.0.0.1:{port}"
        try:
            yield SimpleNamespace(url=url, backend=backend)
        finally:
            server.stop()


class TestSessionServerCRUD:
    def test_create_session(self, session_server_env):
        response = requests.post(f"{session_server_env.url}/sessions", timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 32

    def test_get_session_initial_state(self, session_server_env):
        session_id = requests.post(f"{session_server_env.url}/sessions", timeout=5.0).json()["session_id"]
        get_resp = requests.get(f"{session_server_env.url}/sessions/{session_id}", timeout=5.0)
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["session_id"] == session_id
        assert data["records"] == []

    def test_get_session_not_found(self, session_server_env):
        response = requests.get(f"{session_server_env.url}/sessions/nonexistent", timeout=5.0)
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"

    def test_delete_session(self, session_server_env):
        session_id = requests.post(f"{session_server_env.url}/sessions", timeout=5.0).json()["session_id"]
        delete_resp = requests.delete(f"{session_server_env.url}/sessions/{session_id}", timeout=5.0)
        assert delete_resp.status_code == 204
        assert delete_resp.text == ""
        assert requests.delete(f"{session_server_env.url}/sessions/{session_id}", timeout=5.0).status_code == 404

    def test_delete_session_not_found(self, session_server_env):
        response = requests.delete(f"{session_server_env.url}/sessions/nonexistent", timeout=5.0)
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"


class TestSessionServerChatCompletions:
    def test_chat_completions_appends_record(self, session_server_env):
        session_id = requests.post(f"{session_server_env.url}/sessions", timeout=5.0).json()["session_id"]

        payload = {
            "messages": [{"role": "user", "content": "What is 1+2?"}],
        }
        resp = requests.post(
            f"{session_server_env.url}/sessions/{session_id}/v1/chat/completions",
            json=payload,
            timeout=10.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "choices" in body
        assert body["choices"]

        # Verify record was stored
        get_resp = requests.get(f"{session_server_env.url}/sessions/{session_id}", timeout=5.0)
        records = get_resp.json()["records"]
        assert len(records) == 1
        record = records[0]
        assert record["path"] == "/v1/chat/completions"
        assert record["status_code"] == 200

    def test_chat_completions_injects_logprobs_defaults(self, session_server_env):
        """Session server should set logprobs=True and return_prompt_token_ids=True."""
        session_id = requests.post(f"{session_server_env.url}/sessions", timeout=5.0).json()["session_id"]

        payload = {"messages": [{"role": "user", "content": "hello"}]}
        resp = requests.post(
            f"{session_server_env.url}/sessions/{session_id}/v1/chat/completions",
            json=payload,
            timeout=10.0,
        )
        assert resp.status_code == 200

        # Check the backend received the injected fields
        last_request = session_server_env.backend.request_log[-1]
        assert last_request["logprobs"] is True
        assert last_request["return_prompt_token_ids"] is True


class TestSessionServerProxy:
    def test_transparent_proxy_health(self, session_server_env):
        """Non-session requests should be proxied transparently to the backend."""
        resp = requests.get(f"{session_server_env.url}/health", timeout=5.0)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_transparent_proxy_generate(self, session_server_env):
        """POST to /generate should be proxied to backend."""
        payload = {"input_ids": [1, 2, 3], "return_logprob": True}
        resp = requests.post(f"{session_server_env.url}/generate", json=payload, timeout=10.0)
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body
