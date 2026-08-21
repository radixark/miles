from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils.generate_endpoint_utils import compute_routing_headers
from miles.utils.types import Sample


def test_compute_routing_headers_none_without_key_or_routing():
    args = SimpleNamespace(sglang_router_policy="round_robin", router_api_key=None)
    assert compute_routing_headers(args, Sample(index=0)) is None


def test_compute_routing_headers_omits_authorization_when_key_unset():
    args = SimpleNamespace(sglang_router_policy="round_robin")
    headers = compute_routing_headers(args, Sample(index=0, routing_key="rk"))
    assert headers == {"X-SMG-Routing-Key": "rk"}
    assert "Authorization" not in headers
    assert "Bearer None" not in str(headers)


def test_compute_routing_headers_merges_router_bearer():
    args = SimpleNamespace(sglang_router_policy="round_robin", router_api_key="router-secret")
    assert compute_routing_headers(args, Sample(index=0)) == {"Authorization": "Bearer router-secret"}
    assert compute_routing_headers(args, Sample(index=0, routing_key="rk")) == {
        "X-SMG-Routing-Key": "rk",
        "Authorization": "Bearer router-secret",
    }


def test_compute_routing_headers_requires_routing_key_for_hash_policy():
    args = SimpleNamespace(sglang_router_policy="consistent_hashing", router_api_key=None)
    with pytest.raises(ValueError, match="X-SMG-Routing-Key"):
        compute_routing_headers(args, Sample(index=7))
