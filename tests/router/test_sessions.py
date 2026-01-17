from types import SimpleNamespace

import pytest
import requests

from miles.router.router import MilesRouter
from miles.router.sessions import SessionManager, SessionRecord
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


class TestSessionManager:
    def test_create_session(self):
        manager = SessionManager()
        session_id = manager.create_session()
        assert session_id is not None
        assert len(session_id) == 32
        assert session_id in manager.sessions
        assert manager.sessions[session_id] == []

    def test_get_session_exists(self):
        manager = SessionManager()
        session_id = manager.create_session()
        records = manager.get_session(session_id)
        assert records == []

    def test_get_session_not_exists(self):
        manager = SessionManager()
        records = manager.get_session("nonexistent")
        assert records is None

    def test_delete_session_exists(self):
        manager = SessionManager()
        session_id = manager.create_session()
        records = manager.delete_session(session_id)
        assert records == []
        assert session_id not in manager.sessions

    def test_delete_session_not_exists(self):
        manager = SessionManager()
        with pytest.raises(AssertionError):
            manager.delete_session("nonexistent")

    def test_add_record(self):
        manager = SessionManager()
        session_id = manager.create_session()
        record = SessionRecord(
            timestamp=1234567890.0,
            method="POST",
            path="generate",
            request={"prompt": "hello"},
            response={"text": "world"},
            status_code=200,
        )
        manager.add_record(session_id, record)
        assert len(manager.sessions[session_id]) == 1
        assert manager.sessions[session_id][0] == record

    def test_add_record_nonexistent_session(self):
        manager = SessionManager()
        record = SessionRecord(
            timestamp=1234567890.0,
            method="POST",
            path="generate",
            request={},
            response={},
            status_code=200,
        )
        with pytest.raises(AssertionError):
            manager.add_record("nonexistent", record)


@pytest.fixture(scope="class")
def router_url():
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

    with with_mock_server(process_fn=process_fn) as backend:
        args = SimpleNamespace(
            miles_router_max_connections=10,
            miles_router_timeout=30,
            miles_router_middleware_paths=[],
            rollout_health_check_interval=60,
            miles_router_health_check_failure_threshold=3,
            hf_checkpoint="Qwen/Qwen3-0.6B",
        )
        router = MilesRouter(args)

        port = find_available_port(31000)
        server = UvicornThreadServer(router.app, host="127.0.0.1", port=port)
        server.start()

        url = f"http://127.0.0.1:{port}"
        requests.post(f"{url}/add_worker", json={"url": backend.url})

        try:
            yield url
        finally:
            server.stop()


class TestSessionRoutes:
    def test_create_session(self, router_url):
        response = requests.post(f"{router_url}/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 32

    def test_get_session(self, router_url):
        session_id = requests.post(f"{router_url}/sessions").json()["session_id"]

        get_resp = requests.get(f"{router_url}/sessions/{session_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["session_id"] == session_id
        assert data["records"] == []

    def test_get_session_not_found(self, router_url):
        response = requests.get(f"{router_url}/sessions/nonexistent")
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"

    def test_get_with_records(self, router_url):
        session_id = requests.post(f"{router_url}/sessions").json()["session_id"]

        requests.post(
            f"{router_url}/sessions/{session_id}/generate",
            json={"input_ids": [1, 2, 3], "sampling_params": {}, "return_logprob": True},
        )

        get_resp = requests.get(f"{router_url}/sessions/{session_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["session_id"] == session_id
        assert len(data["records"]) == 1

    def test_delete_session(self, router_url):
        session_id = requests.post(f"{router_url}/sessions").json()["session_id"]

        delete_resp = requests.delete(f"{router_url}/sessions/{session_id}")
        assert delete_resp.status_code == 204
        assert delete_resp.text == ""

        assert requests.delete(f"{router_url}/sessions/{session_id}").status_code == 404

    def test_delete_session_not_found(self, router_url):
        response = requests.delete(f"{router_url}/sessions/nonexistent")
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"


class TestSessionProxy:
    def test_proxy_session_not_found(self, router_url):
        response = requests.post(f"{router_url}/sessions/nonexistent/generate", json={})
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"

    def test_proxy_records_request_response(self, router_url):
        session_id = requests.post(f"{router_url}/sessions").json()["session_id"]

        resp = requests.post(
            f"{router_url}/sessions/{session_id}/generate",
            json={"input_ids": [1, 2, 3], "sampling_params": {}, "return_logprob": True},
        )
        assert resp.status_code == 200
        assert "text" in resp.json()

        get_resp = requests.get(f"{router_url}/sessions/{session_id}")
        records = get_resp.json()["records"]
        assert len(records) == 1
        assert records[0]["method"] == "POST"
        assert records[0]["path"] == "generate"
        assert records[0]["request"]["input_ids"] == [1, 2, 3]
        assert "text" in records[0]["response"]

        delete_resp = requests.delete(f"{router_url}/sessions/{session_id}")
        assert delete_resp.status_code == 204

    def test_proxy_accumulates_records(self, router_url):
        session_id = requests.post(f"{router_url}/sessions").json()["session_id"]

        for _ in range(3):
            requests.post(
                f"{router_url}/sessions/{session_id}/generate",
                json={"input_ids": [1], "sampling_params": {}, "return_logprob": True},
            )

        get_resp = requests.get(f"{router_url}/sessions/{session_id}")
        records = get_resp.json()["records"]
        assert len(records) == 3

        delete_resp = requests.delete(f"{router_url}/sessions/{session_id}")
        assert delete_resp.status_code == 204
