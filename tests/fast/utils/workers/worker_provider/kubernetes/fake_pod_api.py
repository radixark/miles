from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

_INSTALLED: list[Any] = []
CLOSE_CALLS: list[Any] = []


def install(api: object) -> object:
    _INSTALLED.append(api)
    return api


def current() -> Any:
    assert _INSTALLED, "no fake pod api was installed"
    return _INSTALLED[-1]


@asynccontextmanager
async def installed() -> AsyncIterator[Any]:
    assert _INSTALLED, "no fake pod api was installed, so this test would talk to a real cluster"

    api = _INSTALLED[-1]
    try:
        yield api
    finally:
        CLOSE_CALLS.append(api)


def reset() -> None:
    _INSTALLED.clear()
    CLOSE_CALLS.clear()
