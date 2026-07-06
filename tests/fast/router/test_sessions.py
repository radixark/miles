"""Integration tests for session HTTP routes (create / get / delete / proxy)."""

import re
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.inloop_session_server import InloopSessionServer
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server


@pytest.fixture(scope="class")
def router_env():
    """Create an in-loop session server (router app + one worker shard) with a mock backend."""

    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

    original_chat_response = MockSGLangServer._compute_chat_completions_response

    def patched_chat_response(self, payload: dict) -> dict:
        response = original_chat_response(self, payload)
        choice = response["choices"][0]
        logprobs_content = choice["logprobs"]["content"]
        output_token_logprobs = [
            (item["logprob"], self.tokenizer.convert_tokens_to_ids(item["token"])) for item in logprobs_content
        ]
        choice["meta_info"] = {
            "output_token_logprobs": output_token_logprobs,
            "completion_tokens": len(output_token_logprobs),
            # R3 replay payloads: must reach the session record but never the
            # client-facing chat response (see _strip_replay_payloads).
            "routed_experts": [[0, 1], [2, 3]],
            "indexer_topk": [[4], [5]],
        }
        return response

    with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched_chat_response):
        with with_mock_server(process_fn=process_fn) as backend:
            args = SimpleNamespace(
                miles_router_timeout=30,
                hf_checkpoint="Qwen/Qwen3-0.6B",
                chat_template_path=None,
                apply_chat_template_kwargs={"enable_thinking": False},
                tito_model="default",
                tito_allowed_append_roles=["tool"],
                trajectory_manager="linear_trajectory",
                session_server_instance_id=uuid.uuid4().hex,
            )
            port = find_available_port(31000)
            server = InloopSessionServer(args, backend.url, host="127.0.0.1", port=port)
            try:
                server.start()
                yield SimpleNamespace(url=server.url, backend=backend)
            finally:
                server.stop()


class TestSessionRoutes:
    def test_health_reports_stable_instance_id(self, router_env):
        first = requests.get(f"{router_env.url}/health", timeout=5.0)
        second = requests.get(f"{router_env.url}/health", timeout=5.0)

        assert first.status_code == 200
        assert second.status_code == 200
        first_body = first.json()
        second_body = second.json()
        assert first_body["status"] == "ok"
        assert second_body["status"] == "ok"
        assert re.fullmatch(r"[0-9a-f]{32}", first_body["session_server_instance_id"])
        assert second_body["session_server_instance_id"] == first_body["session_server_instance_id"]

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
        assert response.json()["error"] == "session not found: session_id=nonexistent"

    def test_delete_session(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        delete_resp = requests.delete(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        assert delete_resp.status_code == 204
        assert delete_resp.text == ""

        assert requests.delete(f"{router_env.url}/sessions/{session_id}", timeout=5.0).status_code == 404

    def test_delete_session_not_found(self, router_env):
        response = requests.delete(f"{router_env.url}/sessions/nonexistent", timeout=5.0)
        assert response.status_code == 404
        assert response.json()["error"] == "session not found: session_id=nonexistent"


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

    def test_chat_malformed_json_body_returns_400(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        resp = requests.post(
            f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
            timeout=10.0,
        )
        assert resp.status_code == 400
        assert resp.json()["error"].startswith("invalid JSON body:")

    def test_chat_upstream_null_message_returns_502(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        fixture_response = MockSGLangServer._compute_chat_completions_response

        def null_message_response(self, payload: dict) -> dict:
            response = fixture_response(self, payload)
            response["choices"][0]["message"] = None
            return response

        with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=null_message_response):
            resp = requests.post(
                f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                timeout=10.0,
            )
        assert resp.status_code == 502
        assert "assistant message content is None" in resp.json()["error"]

    def test_chat_response_strips_replay_payloads_but_record_keeps_them(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        payload = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "return_logprob": True,
        }
        resp = requests.post(
            f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
            json=payload,
            timeout=10.0,
        )
        assert resp.status_code == 200
        client_meta = resp.json()["choices"][0]["meta_info"]
        assert "routed_experts" not in client_meta
        assert "indexer_topk" not in client_meta
        # Stripping must not swallow the rest of meta_info.
        assert "output_token_logprobs" in client_meta

        record = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"][0]
        record_meta = record["response"]["choices"][0]["meta_info"]
        assert record_meta["routed_experts"] == [[0, 1], [2, 3]]
        assert record_meta["indexer_topk"] == [[4], [5]]
