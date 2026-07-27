from __future__ import annotations

import http.client
import json

import pytest
from tests.fast.utils.test_utils.conftest import connect_via_url

from miles.utils.test_utils.mock_sglang_http_server import MockSGLangHttpServer, RecordedRequest


class TestClose:
    def test_close_severs_an_already_established_connection(self):
        """A crashed engine must stop serving a client that already holds a keep-alive connection."""
        server = MockSGLangHttpServer()
        connection = http.client.HTTPConnection(server.host, server.port, timeout=5)
        connection.request("GET", "/before")
        first = connection.getresponse()
        assert first.status == 200
        first.read()

        server.close()

        with pytest.raises(OSError):
            connection.request("GET", "/after")
            connection.getresponse().read()
        assert server.paths == ["/before"]

    def test_close_refuses_a_fresh_connection(self):
        """A crashed engine must not accept new connections either."""
        server = MockSGLangHttpServer()
        host, port = server.host, server.port
        server.close()

        connection = http.client.HTTPConnection(host, port, timeout=5)
        with pytest.raises(OSError):
            connection.request("GET", "/after")


class TestRecording:
    def test_get_and_post_record_methods_normalized_paths_and_payloads(self, make_server):
        """Every request is recorded in arrival order with its method, query-stripped path and decoded JSON body."""
        server = make_server()
        connection = connect_via_url(server)

        connection.request("GET", "/get_model_info?verbose=1")
        connection.getresponse().read()
        connection.request(
            "POST",
            "/release_memory_occupation",
            body=json.dumps({"tags": ["weights"]}),
            headers={"Content-Type": "application/json"},
        )
        connection.getresponse().read()
        connection.request("POST", "/resume_memory_occupation?tags=kv_cache")
        connection.getresponse().read()
        connection.close()

        assert server.requests == [
            RecordedRequest(method="GET", path="/get_model_info", payload=None),
            RecordedRequest(method="POST", path="/release_memory_occupation", payload={"tags": ["weights"]}),
            RecordedRequest(method="POST", path="/resume_memory_occupation", payload=None),
        ]
        assert server.payloads_of("/release_memory_occupation") == [{"tags": ["weights"]}]
        assert server.payloads_of("/never_called") == []

    def test_requests_returns_an_isolated_snapshot(self, make_server):
        """Mutating the list handed back by ``requests`` must not corrupt the server's own history."""
        server = make_server()
        connection = connect_via_url(server)
        connection.request("GET", "/first")
        connection.getresponse().read()
        connection.close()

        snapshot = server.requests
        snapshot.append(RecordedRequest(method="GET", path="/forged", payload=None))

        assert server.paths == ["/first"]


class TestResponse:
    def test_default_and_configured_payloads_are_returned_as_json(self, make_server):
        """The server answers on its own ``url`` with a JSON body: a default marker, or the configured payload."""
        default_server = make_server()
        connection = connect_via_url(default_server)
        connection.request("GET", "/get_model_info")
        response = connection.getresponse()
        assert response.status == 200
        assert response.headers["Content-Type"] == "application/json"
        assert json.loads(response.read()) == {"mock": True}
        connection.close()

        configured_server = make_server(response_payload={"model_path": "/fake/model"})
        connection = connect_via_url(configured_server)
        connection.request("GET", "/get_model_info")
        assert json.loads(connection.getresponse().read()) == {"model_path": "/fake/model"}
        connection.close()


class TestPortBinding:
    def test_requesting_an_occupied_port_raises(self, make_server):
        """An explicitly requested port that is already taken must fail loudly, not silently serve elsewhere."""
        occupant = make_server()

        with pytest.raises(OSError):
            make_server(port=occupant.port)


class TestMalformedRequest:
    def test_malformed_post_is_not_recorded_and_does_not_stop_the_server(self, make_server):
        """A POST body that is not JSON fails only its own request: nothing is recorded, later requests still work."""
        server = make_server()

        bad_connection = connect_via_url(server)
        with pytest.raises((OSError, http.client.HTTPException)):
            bad_connection.request(
                "POST",
                "/update_weights",
                body="not-json",
                headers={"Content-Type": "application/json"},
            )
            bad_connection.getresponse().read()
        bad_connection.close()

        good_connection = connect_via_url(server)
        good_connection.request("GET", "/health")
        assert good_connection.getresponse().status == 200
        good_connection.close()

        assert server.paths == ["/health"]
