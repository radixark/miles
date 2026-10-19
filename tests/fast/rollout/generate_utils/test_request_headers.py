import asyncio

import pytest

from miles.rollout.generate_utils.request_headers import compute_request_headers


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_provider_adds_headers_without_mutating_inputs(asynchronous):
    original = {"X-SMG-Routing-Key": "session"}
    additions = {"X-Workload": "evaluation"}

    def get_headers(*, evaluation):
        assert evaluation is True
        return asyncio.sleep(0, result=additions) if asynchronous else additions

    result = await compute_request_headers(original, get_headers, evaluation=True)

    assert result == {"X-SMG-Routing-Key": "session", "X-Workload": "evaluation"}
    assert original == {"X-SMG-Routing-Key": "session"}
    assert additions == {"X-Workload": "evaluation"}


@pytest.mark.parametrize(
    "headers,additions",
    [
        ({"X-SMG-Routing-Key": "session"}, {"x-smg-routing-key": "other"}),
        (None, {"Content-Length": "0"}),
        (None, {"cOnTenT-tYpE": "text/plain"}),
        (None, {"Host": "other"}),
        (None, {"X-Policy": "one", "x-policy": "two"}),
    ],
)
async def test_provider_cannot_override_owned_or_duplicate_headers(headers, additions):
    with pytest.raises(ValueError, match="cannot override header"):
        await compute_request_headers(headers, lambda **_: additions, evaluation=False)


@pytest.mark.parametrize("additions", [None, {"X-Policy": 1}])
async def test_provider_must_return_string_headers(additions):
    with pytest.raises(TypeError, match=r"dict\[str, str\]"):
        await compute_request_headers(None, lambda **_: additions, evaluation=False)
