from __future__ import annotations

import httpx


def make_ok_response() -> httpx.Response:
    return httpx.Response(status_code=200, request=httpx.Request("POST", "https://example.com"))


def make_error_response(status_code: int = 500) -> httpx.Response:
    return httpx.Response(status_code=status_code, request=httpx.Request("POST", "https://example.com"))
