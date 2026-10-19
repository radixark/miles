"""Additional service metadata for rollout inference requests."""

from miles.utils.async_utils import maybe_await


async def compute_request_headers(
    headers: dict[str, str] | None, headers_function, *, evaluation: bool
) -> dict[str, str]:
    """Add provider headers without changing existing headers or HTTP framing."""
    extra_headers = await maybe_await(headers_function(evaluation=evaluation))
    if not isinstance(extra_headers, dict) or any(
        not isinstance(name, str) or not isinstance(value, str) for name, value in extra_headers.items()
    ):
        raise TypeError("The rollout request headers function must return dict[str, str]")

    result = dict(headers or {})
    # These fields describe the HTTP exchange, not external-service metadata.
    owned = {name.lower() for name in result} | {
        "accept",
        "accept-encoding",
        "connection",
        "content-encoding",
        "content-length",
        "content-type",
        "expect",
        "host",
        "keep-alive",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
    for name, value in extra_headers.items():
        if name.lower() in owned:
            raise ValueError(f"The rollout request headers function cannot override header {name!r}")
        result[name] = value
        owned.add(name.lower())
    return result
