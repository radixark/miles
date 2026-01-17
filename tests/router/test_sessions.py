import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from miles.router.router import MilesRouter
from miles.router.sessions import SessionManager, SessionRecord
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server


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
            request_json={"prompt": "hello"},
            response_json={"text": "world"},
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
            request_json={},
            response_json={},
            status_code=200,
        )
        with pytest.raises(AssertionError):
            manager.add_record("nonexistent", record)


@pytest.fixture(scope="class")
def integration_client():
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

    with with_mock_server(process_fn=process_fn) as server:
        args = SimpleNamespace(
            miles_router_max_connections=10,
            miles_router_timeout=30,
            miles_router_middleware_paths=[],
            rollout_health_check_interval=60,
            miles_router_health_check_failure_threshold=3,
        )
        router = MilesRouter(args)
        router.worker_request_counts[server.url] = 0
        router.worker_failure_counts[server.url] = 0
        yield TestClient(router.app)


class TestSessionRoutes:
    def test_create_session(self, integration_client):
        response = integration_client.post("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 32

    def test_delete_session(self, integration_client):
        session_id = integration_client.post("/sessions").json()["session_id"]

        delete_resp = integration_client.delete(f"/sessions/{session_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["session_id"] == session_id
        assert delete_resp.json()["records"] == []

        assert integration_client.delete(f"/sessions/{session_id}").status_code == 404

    def test_delete_session_not_found(self, integration_client):
        response = integration_client.delete("/sessions/nonexistent")
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"


class TestSessionProxy:
    def test_proxy_session_not_found(self, integration_client):
        response = integration_client.post("/sessions/nonexistent/generate", json={})
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"

    def test_proxy_records_request_response(self, integration_client):
        session_id = integration_client.post("/sessions").json()["session_id"]

        resp = integration_client.post(
            f"/sessions/{session_id}/generate",
            json={"input_ids": [1, 2, 3], "sampling_params": {}, "return_logprob": True},
        )
        assert resp.status_code == 200
        assert "text" in resp.json()

        records = integration_client.delete(f"/sessions/{session_id}").json()["records"]
        assert len(records) == 1
        assert records[0]["method"] == "POST"
        assert records[0]["path"] == "generate"
        assert records[0]["request_json"]["input_ids"] == [1, 2, 3]
        assert "text" in records[0]["response_json"]

    def test_proxy_accumulates_records(self, integration_client):
        session_id = integration_client.post("/sessions").json()["session_id"]

        for _ in range(3):
            integration_client.post(
                f"/sessions/{session_id}/generate",
                json={"input_ids": [1], "sampling_params": {}, "return_logprob": True},
            )

        records = integration_client.delete(f"/sessions/{session_id}").json()["records"]
        assert len(records) == 3
