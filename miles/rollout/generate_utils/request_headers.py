from collections.abc import Callable

from miles.utils.async_utils import maybe_await


async def customize_request_headers(
    headers: dict[str, str] | None, function: Callable | None, *, evaluation: bool
) -> dict[str, str] | None:
    if function is None:
        return headers
    return await maybe_await(function(headers if headers is not None else {}, evaluation=evaluation))
