import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from miles.router.sessions import SessionManager, SessionRecord, setup_session_routes


@pytest.fixture
def mock_router():
    router = MagicMock()
    router._use_url = MagicMock(return_value="http://mock-worker:8000")
    router._finish_url = MagicMock()
    router.client = AsyncMock(spec=httpx.AsyncClient)
    return router


@pytest.fixture
def app_with_sessions(mock_router):
    app = FastAPI()
    setup_session_routes(app, mock_router)
    return app, mock_router


@pytest.fixture
def client(app_with_sessions):
    app, _ = app_with_sessions
    return TestClient(app)


class TestSessionManager:
    def test_create_session(self, mock_router):
        manager = SessionManager(mock_router)
        session_id = manager.create_session()
        assert session_id is not None
        assert len(session_id) == 32
        assert session_id in manager.sessions
        assert manager.sessions[session_id] == []

    def test_get_session_exists(self, mock_router):
        manager = SessionManager(mock_router)
        session_id = manager.create_session()
        records = manager.get_session(session_id)
        assert records == []

    def test_get_session_not_exists(self, mock_router):
        manager = SessionManager(mock_router)
        records = manager.get_session("nonexistent")
        assert records is None

    def test_delete_session_exists(self, mock_router):
        manager = SessionManager(mock_router)
        session_id = manager.create_session()
        records = manager.delete_session(session_id)
        assert records == []
        assert session_id not in manager.sessions

    def test_delete_session_not_exists(self, mock_router):
        manager = SessionManager(mock_router)
        records = manager.delete_session("nonexistent")
        assert records is None

    def test_add_record(self, mock_router):
        manager = SessionManager(mock_router)
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

    def test_add_record_nonexistent_session(self, mock_router):
        manager = SessionManager(mock_router)
        record = SessionRecord(
            timestamp=1234567890.0,
            method="POST",
            path="generate",
            request_json={},
            response_json={},
            status_code=200,
        )
        manager.add_record("nonexistent", record)


class TestSessionRoutes:
    def test_create_session(self, client):
        response = client.post("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 32

    def test_get_session(self, client):
        create_resp = client.post("/sessions")
        session_id = create_resp.json()["session_id"]

        get_resp = client.get(f"/sessions/{session_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["session_id"] == session_id
        assert data["records"] == []

    def test_get_session_not_found(self, client):
        response = client.get("/sessions/nonexistent")
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"

    def test_delete_session(self, client):
        create_resp = client.post("/sessions")
        session_id = create_resp.json()["session_id"]

        delete_resp = client.delete(f"/sessions/{session_id}")
        assert delete_resp.status_code == 200
        data = delete_resp.json()
        assert data["session_id"] == session_id
        assert data["records"] == []

        get_resp = client.get(f"/sessions/{session_id}")
        assert get_resp.status_code == 404

    def test_delete_session_not_found(self, client):
        response = client.delete("/sessions/nonexistent")
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"


class TestSessionProxy:
    def test_proxy_json_request_response(self, app_with_sessions):
        app, mock_router = app_with_sessions
        client = TestClient(app)

        create_resp = client.post("/sessions")
        session_id = create_resp.json()["session_id"]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=json.dumps({"result": "ok"}).encode())
        mock_router.client.request = AsyncMock(return_value=mock_response)

        proxy_resp = client.post(
            f"/sessions/{session_id}/generate",
            json={"prompt": "hello"},
        )

        assert proxy_resp.status_code == 200
        assert proxy_resp.json() == {"result": "ok"}

        mock_router._use_url.assert_called()
        mock_router._finish_url.assert_called_with("http://mock-worker:8000")

        get_resp = client.get(f"/sessions/{session_id}")
        records = get_resp.json()["records"]
        assert len(records) == 1
        assert records[0]["method"] == "POST"
        assert records[0]["path"] == "generate"
        assert records[0]["request_json"] == {"prompt": "hello"}
        assert records[0]["response_json"] == {"result": "ok"}
        assert records[0]["status_code"] == 200

    def test_proxy_non_json_response(self, app_with_sessions):
        app, mock_router = app_with_sessions
        client = TestClient(app)

        create_resp = client.post("/sessions")
        session_id = create_resp.json()["session_id"]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.aread = AsyncMock(return_value=b"plain text response")
        mock_router.client.request = AsyncMock(return_value=mock_response)

        proxy_resp = client.post(f"/sessions/{session_id}/health")

        assert proxy_resp.status_code == 200
        assert proxy_resp.text == "plain text response"

        get_resp = client.get(f"/sessions/{session_id}")
        records = get_resp.json()["records"]
        assert records[0]["response_json"] is None

    def test_proxy_session_not_found(self, client):
        response = client.post("/sessions/nonexistent/generate", json={})
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"

    def test_proxy_multiple_requests(self, app_with_sessions):
        app, mock_router = app_with_sessions
        client = TestClient(app)

        create_resp = client.post("/sessions")
        session_id = create_resp.json()["session_id"]

        for i in range(3):
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.aread = AsyncMock(return_value=json.dumps({"i": i}).encode())
            mock_router.client.request = AsyncMock(return_value=mock_response)

            client.post(f"/sessions/{session_id}/test", json={"req": i})

        get_resp = client.get(f"/sessions/{session_id}")
        records = get_resp.json()["records"]
        assert len(records) == 3
        for i, record in enumerate(records):
            assert record["request_json"] == {"req": i}
            assert record["response_json"] == {"i": i}

    def test_proxy_different_http_methods(self, app_with_sessions):
        app, mock_router = app_with_sessions
        client = TestClient(app)

        create_resp = client.post("/sessions")
        session_id = create_resp.json()["session_id"]

        methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        for method in methods:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.aread = AsyncMock(return_value=json.dumps({"method": method}).encode())
            mock_router.client.request = AsyncMock(return_value=mock_response)

            resp = client.request(method, f"/sessions/{session_id}/test")
            assert resp.status_code == 200

        get_resp = client.get(f"/sessions/{session_id}")
        records = get_resp.json()["records"]
        assert len(records) == len(methods)
        for i, record in enumerate(records):
            assert record["method"] == methods[i]
