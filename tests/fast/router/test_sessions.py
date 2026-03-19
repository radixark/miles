"""Integration tests for session HTTP routes (create / get / delete / proxy)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from miles.router.router import MilesRouter
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


@pytest.fixture(scope="class")
def router_env():
    """Create a MilesRouter with session routes and a mock backend."""

    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

    original_chat_response = MockSGLangServer._compute_chat_completions_response

    def patched_chat_response(self, payload: dict) -> dict:
        response = original_chat_response(self, payload)
        logprobs_content = response["choices"][0]["logprobs"]["content"]
        for item in logprobs_content:
            item["token_id"] = self.tokenizer.convert_tokens_to_ids(item["token"])
        return response

    with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched_chat_response):
        with with_mock_server(process_fn=process_fn) as backend:
            args = SimpleNamespace(
                miles_router_max_connections=10,
                miles_router_timeout=30,
                miles_router_middleware_paths=[],
                rollout_health_check_interval=60,
                miles_router_health_check_failure_threshold=3,
                hf_checkpoint="Qwen/Qwen3-0.6B",
                chat_template_path=None,
                trajectory_manager="single_user_turn_trajectory",
            )
            router = MilesRouter(args)

            port = find_available_port(31000)
            server = UvicornThreadServer(router.app, host="127.0.0.1", port=port)
            server.start()

            url = f"http://127.0.0.1:{port}"
            requests.post(f"{url}/add_worker", json={"url": backend.url}, timeout=5.0)

            try:
                yield SimpleNamespace(url=url)
            finally:
                server.stop()


class TestSessionRoutes:
    def test_create_session(self, router_env):
        response = requests.post(f"{router_env.url}/sessions", timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 32

    def test_get_session_initial_state(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        get_resp = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["session_id"] == session_id
        assert data["records"] == []

    def test_get_session_not_found(self, router_env):
        response = requests.get(f"{router_env.url}/sessions/nonexistent", timeout=5.0)
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"

    def test_delete_session(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        delete_resp = requests.delete(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        assert delete_resp.status_code == 204
        assert delete_resp.text == ""

        assert requests.delete(f"{router_env.url}/sessions/{session_id}", timeout=5.0).status_code == 404

    def test_delete_session_not_found(self, router_env):
        response = requests.delete(f"{router_env.url}/sessions/nonexistent", timeout=5.0)
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"


class TestSessionProxy:
    def test_proxy_chat_appends_record(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        payload = {
            "messages": [{"role": "user", "content": "What is 1+2?"}],
            "return_logprob": True,
        }
        resp = requests.post(
            f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
            json=payload,
            timeout=10.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "choices" in body
        assert body["choices"]

        get_resp = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        records = get_resp.json()["records"]

        assert isinstance(records, list)
        assert len(records) == 1
        record = records[0]
        assert record["path"] == "/v1/chat/completions"
        assert record["status_code"] == 200
