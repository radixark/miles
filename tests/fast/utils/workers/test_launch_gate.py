from __future__ import annotations

import httpx
import pytest

from miles.utils.workers import launch_gate
from miles.utils.workers.launch_gate import activate_launch_gate


class _StubResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPError(f"status {self.status_code}")


class _StubHttpClient:
    def __init__(self, outcomes: list):
        self._outcomes = outcomes
        self.posted_urls: list[str] = []
        self.posted_timeouts: list[float] = []

    async def post(self, url: str, json: dict, timeout: float):
        self.posted_urls.append(url)
        self.posted_timeouts.append(timeout)
        outcome = self._outcomes.pop(0) if self._outcomes else _StubResponse()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def stub_http(monkeypatch):
    def _install(outcomes: list) -> _StubHttpClient:
        client = _StubHttpClient(outcomes)
        monkeypatch.setattr(launch_gate.GeneralHttpClientProvider, "client", staticmethod(lambda: client))
        monkeypatch.setattr(launch_gate, "_INITIAL_DELAY_SECONDS", 0.001)
        monkeypatch.setattr(launch_gate, "_MAX_DELAY_SECONDS", 0.001)
        monkeypatch.setattr(launch_gate, "_ATTEMPT_TIMEOUT_SECONDS", 30.0)
        return client

    return _install


class TestActivateLaunchGate:
    async def test_it_posts_to_the_activate_endpoint_of_the_gate_port(self, stub_http):
        """The gate protocol lives on its own port and path, not on the serving api."""
        client = stub_http([_StubResponse()])

        await activate_launch_gate(gate_url="http://10.0.0.1:13000")

        assert client.posted_urls == ["http://10.0.0.1:13000/gate/activate"]

    async def test_it_retries_while_the_gate_port_refuses_connections(self, stub_http):
        """The port only starts listening once the engine reaches its gate, minutes after launch."""
        client = stub_http([httpx.ConnectError("refused"), httpx.ConnectError("refused"), _StubResponse()])

        await activate_launch_gate(gate_url="http://10.0.0.1:13000")

        assert len(client.posted_urls) == 3

    async def test_it_retries_an_os_error_from_the_gate_transport(self, stub_http):
        """A raw socket error while the engine host is still coming up must be retried, not fatal."""
        client = stub_http([OSError("no route to host"), _StubResponse()])

        await activate_launch_gate(gate_url="http://10.0.0.1:13000")

        assert len(client.posted_urls) == 2

    async def test_it_does_not_retry_an_unlisted_client_error(self, stub_http):
        """A bug in our own request building must surface at once instead of burning the whole deadline."""
        client = stub_http([RuntimeError("boom")])

        with pytest.raises(RuntimeError):
            await activate_launch_gate(gate_url="http://10.0.0.1:13000")

        assert len(client.posted_urls) == 1

    async def test_it_retries_a_rejected_activation(self, stub_http):
        """A gate answering with an error is not yet ready to be told to proceed."""
        client = stub_http([_StubResponse(status_code=503), _StubResponse()])

        await activate_launch_gate(gate_url="http://10.0.0.1:13000")

        assert len(client.posted_urls) == 2

    async def test_it_gives_up_after_the_deadline_instead_of_hanging_forever(self, stub_http):
        """An engine that never reaches its gate must surface as a failure, not a silent hang."""
        stub_http([httpx.ConnectError("refused")] * 1000)

        with pytest.raises(httpx.ConnectError):
            await activate_launch_gate(gate_url="http://10.0.0.1:13000", timeout=0.0)

    async def test_it_stops_posting_once_the_gate_accepts(self, stub_http):
        """Activation is a one-shot release; repeating it would be pointless traffic."""
        client = stub_http([_StubResponse(), _StubResponse()])

        await activate_launch_gate(gate_url="http://10.0.0.1:13000")

        assert len(client.posted_urls) == 1

    async def test_every_attempt_carries_its_own_timeout(self, stub_http):
        """A gate that accepts the connection but never answers must not outlive the deadline."""
        client = stub_http([httpx.ConnectError("refused"), _StubResponse()])

        await activate_launch_gate(gate_url="http://10.0.0.1:13000")

        assert client.posted_timeouts == [30.0, 30.0]

    async def test_the_attempt_timeout_shrinks_to_what_is_left_of_the_deadline(self, stub_http):
        """Late attempts must not be allowed to hang past the overall budget."""
        client = stub_http([_StubResponse()])

        await activate_launch_gate(gate_url="http://10.0.0.1:13000", timeout=2.0)

        assert client.posted_timeouts[0] == pytest.approx(2.0, abs=0.5)
