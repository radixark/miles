import time
from types import SimpleNamespace

import pytest
import requests


class _Resp:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}", response=self)


def _engine(api_key=None):
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000
    engine.node_rank = 0
    engine.server_api_key = api_key
    return engine


def test_flush_cache_sleeps_between_pending_request_retries(monkeypatch):
    """Regression test for the fully_async weight-update crash: sglang
    returns 400 (not an exception) while requests are still pending, so the
    retry loop must back off on THAT path too, or all 60 "attempts" burn
    through in a fraction of a second — nowhere near enough time for
    in-flight generation to drain — and flush_cache raises TimeoutError
    almost immediately after pause_generation instead of after ~60s."""
    engine = _engine()

    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: _Resp(status_code=400, text="pending"))

    with pytest.raises(TimeoutError, match="Timeout while flushing cache"):
        engine.flush_cache()

    assert len(sleep_calls) == 60, (
        f"expected the loop to back off on every one of its 60 attempts, got {len(sleep_calls)} sleeps "
        "-- a 400 response (pending requests) must not skip the retry delay"
    )


def test_wait_server_healthy_omits_authorization_when_unset(monkeypatch):
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import _wait_server_healthy

    seen = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            seen.append(headers)
            return _Resp()

    monkeypatch.setattr(requests, "Session", lambda: _Session())
    _wait_server_healthy("http://127.0.0.1:1", None, lambda: True)
    assert seen
    assert all("Authorization" not in headers for headers in seen)
    assert all("Bearer None" not in str(headers) for headers in seen)


def test_wait_server_healthy_sends_bearer_when_key_set(monkeypatch):
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import _wait_server_healthy

    seen = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            seen.append(headers)
            return _Resp()

    monkeypatch.setattr(requests, "Session", lambda: _Session())
    _wait_server_healthy("http://127.0.0.1:1", "secret", lambda: True)
    assert all(headers.get("Authorization") == "Bearer secret" for headers in seen)


def test_make_request_uses_server_headers(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs.get("headers") or {}, kwargs.get("json")))
        return _Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    _engine("secret")._make_request("update_weights_from_tensor", {"x": 1})
    url, headers, payload = calls[0]
    assert url.endswith("/update_weights_from_tensor")
    assert headers == {"Authorization": "Bearer secret"}
    assert payload == {"x": 1}


def test_control_plane_omits_authorization_when_no_key(monkeypatch):
    calls = []

    def fake_request(method):
        def _call(url, **kwargs):
            calls.append(kwargs.get("headers") or {})
            return _Resp(json_data={"weight_version": "v1"})

        return _call

    monkeypatch.setattr(requests, "get", fake_request("GET"))
    monkeypatch.setattr(requests, "post", fake_request("POST"))
    engine = _engine(None)
    engine.get_server_info()
    engine.flush_cache()
    engine._make_request("release_memory_occupation", {"tags": ["kv_cache"]})
    assert calls
    assert all("Authorization" not in headers for headers in calls)
    assert all("Bearer None" not in str(headers) for headers in calls)


@pytest.mark.parametrize("status_code", [401, 403])
def test_flush_cache_fails_fast_on_auth_error(monkeypatch, status_code):
    engine = _engine("wrong-key")
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: _Resp(status_code=status_code, text="nope"))
    with pytest.raises(requests.exceptions.HTTPError):
        engine.flush_cache()
    assert sleep_calls == []


def test_router_registration_sends_bearer_and_forwards_worker_key(monkeypatch):
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils import sglang_engine as mod
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs.get("headers") or {}, kwargs.get("json")))
        return _Resp()

    monkeypatch.setattr(mod, "sglang_router", SimpleNamespace(__version__="0.3.0"))
    monkeypatch.setattr(mod, "ServerArgs", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(mod, "launch_server_process", lambda server_args: SimpleNamespace(pid=0))
    monkeypatch.setattr(requests, "post", fake_post)

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.args = SimpleNamespace(use_miles_router=False, router_api_key="router-key", rollout_external=False)
    engine.node_rank = 0
    engine.worker_type = "regular"
    engine.server_host = "127.0.0.1"
    engine.server_port = 12345
    engine.router_ip = "router"
    engine.router_port = 8000
    engine._init_normal({"api_key": "worker-key"})

    url, headers, payload = calls[0]
    assert url.endswith("/workers")
    assert headers == {"Authorization": "Bearer router-key"}
    assert payload["api_key"] == "worker-key"
