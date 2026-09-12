"""Optional contract checks against the installed Dynamo frontend parser.

Miles' regular unit tests do not require Dynamo. When a compatible Dynamo is
present in an integration-test image, these checks turn our reviewed v1.4 pins
into an actual upstream drift detector.
"""

import argparse

import pytest

from miles.backends.dynamo_utils.arguments import DYNAMO_UPSTREAM_DEFAULTS


def test_frontend_defaults_match_installed_dynamo(monkeypatch):
    frontend_args = pytest.importorskip("dynamo.frontend.frontend_args")

    env_vars = {
        "DYN_DISCOVERY_BACKEND",
        "DYN_REQUEST_PLANE",
        "DYN_EVENT_PLANE",
        "DYN_ROUTER_MODE",
        "DYN_ROUTER_KV_EVENTS",
        "DYN_ROUTER_TTL_SECS",
        "DYN_ROUTER_PREDICTED_TTL_SECS",
        "DYN_ROUTER_MIN_INITIAL_WORKERS",
        "DYN_ROUTER_QUEUE_THRESHOLD",
    }
    for name in env_vars:
        monkeypatch.delenv(name, raising=False)

    parser = argparse.ArgumentParser()
    frontend_args.FrontendArgGroup().add_arguments(parser)
    parsed = parser.parse_args([])

    assert parsed.discovery_backend == DYNAMO_UPSTREAM_DEFAULTS["discovery-backend"]
    assert parsed.request_plane == DYNAMO_UPSTREAM_DEFAULTS["request-plane"]
    assert parsed.router_mode == DYNAMO_UPSTREAM_DEFAULTS["router-mode"]
    assert parsed.use_kv_events == DYNAMO_UPSTREAM_DEFAULTS["router-kv-events"]
    assert parsed.router_ttl_secs == DYNAMO_UPSTREAM_DEFAULTS["router-ttl-secs"]
    assert parsed.router_predicted_ttl_secs == DYNAMO_UPSTREAM_DEFAULTS["router-predicted-ttl-secs"]
    assert parsed.min_initial_workers == DYNAMO_UPSTREAM_DEFAULTS["router-min-initial-workers"]
    assert parsed.router_queue_threshold == DYNAMO_UPSTREAM_DEFAULTS["router-queue-threshold"]

    # The parser leaves this unset; Dynamo's runtime resolves the effective
    # event plane to ZMQ. Keep the distinction explicit in the contract test.
    assert parsed.event_plane is None
    assert DYNAMO_UPSTREAM_DEFAULTS["event-plane"] == "zmq"
