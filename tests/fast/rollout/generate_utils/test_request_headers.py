import asyncio

import pytest

from miles.rollout.generate_utils.request_headers import compute_request_headers


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_function_customizes_defaults_without_mutating_inputs(asynchronous):
    original = {"X-SMG-Routing-Key": "session", "X-Remove": "old", "Content-Type": "application/json"}

    def customize_headers(headers, *, evaluation):
        assert evaluation is True
        assert headers == {"x-smg-routing-key": "session", "x-remove": "old"}
        headers["x-smg-routing-key"] = "custom-session"
        headers.pop("x-remove")
        headers["X-Workload"] = "evaluation"
        return asyncio.sleep(0, result=headers) if asynchronous else headers

    result = await compute_request_headers(original, customize_headers, evaluation=True)

    assert result == {
        "x-smg-routing-key": "custom-session",
        "x-workload": "evaluation",
        "content-type": "application/json",
    }
    assert original == {"X-SMG-Routing-Key": "session", "X-Remove": "old", "Content-Type": "application/json"}


async def test_empty_result_does_not_restore_application_defaults():
    result = await compute_request_headers({"X-SMG-Routing-Key": "session"}, lambda headers, **_: {}, evaluation=False)
    assert result == {}


@pytest.mark.parametrize("name", ["Content-Length", "cOnTenT-tYpE", "Host"])
async def test_function_cannot_set_transport_headers(name):
    with pytest.raises(ValueError, match="cannot set transport header"):
        await compute_request_headers(None, lambda headers, **_: {name: "value"}, evaluation=False)


async def test_function_cannot_return_duplicate_header_names():
    with pytest.raises(ValueError, match="Duplicate rollout request header"):
        await compute_request_headers(
            None, lambda headers, **_: {"X-Policy": "one", "x-policy": "two"}, evaluation=False
        )


@pytest.mark.parametrize("resolved", [None, {"X-Policy": 1}])
async def test_function_must_return_string_headers(resolved):
    with pytest.raises(TypeError, match=r"dict\[str, str\]"):
        await compute_request_headers(None, lambda headers, **_: resolved, evaluation=False)
