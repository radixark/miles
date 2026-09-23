import pytest
from fastapi import FastAPI
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session import sessions
from miles.rollout.session.sessions import setup_session_routes


class _UnusedBackend:
    async def do_proxy(self, *args, **kwargs):
        raise AssertionError("setup_session_routes must not touch the proxy backend")


class TestSetupSessionRoutes:
    def test_without_hf_checkpoint_registers_no_session_routes(self, monkeypatch: pytest.MonkeyPatch):
        """A config without a checkpoint loads no tokenizer and leaves the app's routes untouched."""
        tokenizer_calls: list[tuple] = []

        def record_tokenizer_load(*args, **kwargs):
            tokenizer_calls.append((args, kwargs))
            raise AssertionError("setup_session_routes must not load a tokenizer without a checkpoint")

        monkeypatch.setattr(sessions, "load_tokenizer", record_tokenizer_load)
        app = FastAPI()
        before = [route.path for route in app.routes]

        setup_session_routes(app, _UnusedBackend(), make_session_server_config(hf_checkpoint=None))

        assert tokenizer_calls == []
        assert [route.path for route in app.routes] == before
        assert "/health" not in before
