"""Customize application headers without changing the inference payload."""

from miles.utils.async_utils import maybe_await


async def compute_request_headers(
    headers: dict[str, str] | None, headers_function, *, evaluation: bool
) -> dict[str, str]:
    """Resolve application headers from defaults; retain HTTP framing outside the hook."""
    transport_fields = {
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
    defaults = {name.lower(): value for name, value in (headers or {}).items()}
    application_headers = {name: value for name, value in defaults.items() if name not in transport_fields}
    resolved = await maybe_await(headers_function(application_headers, evaluation=evaluation))
    if not isinstance(resolved, dict) or any(
        not isinstance(name, str) or not isinstance(value, str) for name, value in resolved.items()
    ):
        raise TypeError("The rollout request headers function must return dict[str, str]")

    # The hook's result replaces application defaults; only transport fields are retained.
    result = {name: value for name, value in defaults.items() if name in transport_fields}
    for name, value in resolved.items():
        key = name.lower()
        if key in transport_fields:
            raise ValueError(f"The rollout request headers function cannot set transport header {name!r}")
        if key in result:
            raise ValueError(f"Duplicate rollout request header {name!r}")
        result[key] = value
    return result
