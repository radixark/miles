from types import SimpleNamespace

from fastapi.testclient import TestClient

from miles.ray.tinker.http_server import TinkerHTTPServer


def _client(api_key: str | None, *, ready: bool = True) -> TestClient:
    backend = SimpleNamespace(
        args=SimpleNamespace(tinker_api_key=api_key, seq_length=4096),
        model_name="test/model",
        ready=ready,
    )
    server = TinkerHTTPServer(backend)
    app = server.create_app()
    server.add_routes(app)
    return TestClient(app)


def test_healthz_does_not_require_authentication():
    with _client("tml-secret") as client:
        response = client.get("/api/v1/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_is_not_ready_until_the_trainer_is_initialized():
    with _client(None, ready=False) as client:
        response = client.get("/api/v1/healthz")

    assert response.status_code == 503
    assert response.json() == {"detail": "trainer is initializing"}


def test_official_sdk_api_key_header_is_authenticated():
    with _client("tml-secret") as client:
        unauthorized = client.get("/api/v1/get_server_capabilities")
        authorized = client.get(
            "/api/v1/get_server_capabilities",
            headers={"X-API-Key": "tml-secret"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_bearer_header_is_accepted_for_non_sdk_clients():
    with _client("tml-secret") as client:
        response = client.get(
            "/api/v1/get_server_capabilities",
            headers={"Authorization": "Bearer tml-secret"},
        )

    assert response.status_code == 200
