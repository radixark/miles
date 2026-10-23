from argparse import Namespace

import pytest
from tests.fast.fixtures.args_fixtures import sglang_router_args

from miles.router.backend import _BACKEND_CLASSES, RouterKind, make_router_backend


def test_every_router_kind_has_its_own_backend():
    """A kind without a backend would only fail once a run selects it."""
    assert set(_BACKEND_CLASSES) == set(RouterKind)
    assert all(backend.kind is kind for kind, backend in _BACKEND_CLASSES.items())


@pytest.mark.parametrize(
    "router, policy, expected",
    [
        ("sglang", "consistent_hashing", {"X-SMG-Routing-Key": "key-1"}),
        ("sglang", "manual", {"X-SMG-Routing-Key": "key-1"}),
        ("sglang", "cache_aware", {}),
        ("sglang", None, {}),
        ("miles", "manual", {}),
    ],
    ids=[
        "sgl-router-consistent-hashing",
        "sgl-router-manual",
        "sgl-router-cache-aware",
        "sgl-router-default",
        "miles-router",
    ],
)
def test_a_routing_key_header_is_sent_only_to_a_router_that_reads_it(router, policy, expected):
    backend = make_router_backend(Namespace(**sglang_router_args(policy) | {"router": router}))

    assert backend.routing_headers("key-1") == expected


def test_a_key_routed_request_without_a_key_is_rejected():
    """Without the header a key policy falls back to an implicit key and can pin every request to one engine."""
    backend = make_router_backend(Namespace(**sglang_router_args("manual")))

    with pytest.raises(ValueError, match="routes by routing key"):
        backend.routing_headers(None)


@pytest.mark.parametrize(
    "policies, expected",
    [
        ({"router_policy": "manual"}, True),
        ({"router_policy": "manual", "sglang_router_policy": "cache_aware"}, False),
        ({"router_prefill_policy": "consistent_hashing"}, True),
        ({"router_decode_policy": "manual"}, True),
        ({"router_prefill_policy": "cache_aware", "router_decode_policy": "round_robin"}, False),
    ],
    ids=[
        "pass-through-policy",
        "sglang-router-policy-overrides-pass-through",
        "pd-prefill-policy",
        "pd-decode-policy",
        "pd-load-balancing-policies",
    ],
)
def test_routing_by_key_follows_the_policy_the_router_runs(policies, expected):
    """The header must reach a router launched with a key policy from any of its policy flags."""
    backend = make_router_backend(Namespace(**sglang_router_args() | policies))

    assert backend.routes_by_key() is expected
