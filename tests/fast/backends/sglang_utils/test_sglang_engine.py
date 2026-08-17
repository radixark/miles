import time
from types import SimpleNamespace

import pytest
import requests


def test_flush_cache_sleeps_between_pending_request_retries(monkeypatch):
    """Regression test for the fully_async weight-update crash: sglang
    returns 400 (not an exception) while requests are still pending, so the
    retry loop must back off on THAT path too, or all 60 "attempts" burn
    through in a fraction of a second — nowhere near enough time for
    in-flight generation to drain — and flush_cache raises TimeoutError
    almost immediately after pause_generation instead of after ~60s."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "fake-host"
    engine.server_port = 1234

    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(requests, "get", lambda url: type("Resp", (), {"status_code": 400})())

    with pytest.raises(TimeoutError, match="Timeout while flushing cache"):
        engine.flush_cache()

    assert len(sleep_calls) == 60, (
        f"expected the loop to back off on every one of its 60 attempts, got {len(sleep_calls)} sleeps "
        "-- a 400 response (pending requests) must not skip the retry delay"
    )


@pytest.mark.parametrize(
    "multi_lora, expected_payload",
    [
        # Multi-LoRA: one tenant's publish must not abort another tenant's
        # in-flight sampling, so the bump explicitly opts out of the abort.
        (True, {"new_version": "3", "abort_all_requests": False}),
        # Single-model: keep the endpoint's default (abort on weight update is
        # the intended staleness control), i.e. don't send the knob at all.
        (False, {"new_version": "3"}),
    ],
)
def test_update_weight_version_abort_policy(monkeypatch, multi_lora, expected_payload):
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "fake-host"
    engine.server_port = 1234
    engine.args = SimpleNamespace(multi_lora=multi_lora)

    posts = []

    def fake_post(url, json=None):
        posts.append((url, json))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})

    monkeypatch.setattr(requests, "post", fake_post)

    engine.update_weight_version("3")

    assert posts == [("http://fake-host:1234/update_weight_version", expected_payload)]
