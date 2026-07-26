import pytest


def test_engine_exposes_only_launcher_duties():
    """The actor keeps launcher duties only; http calls live on SGLangApiClient."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    client_methods = {name for name in vars(SGLangApiClient) if not name.startswith("_")}
    engine_methods = {name for name in vars(SGLangEngine) if not name.startswith("_")}

    assert not (engine_methods & client_methods), "these belong on the api client, not on the actor"
    assert {"init", "shutdown", "simulate_crash", "get_topology_info"} <= engine_methods
