from __future__ import annotations

_INSTALLED: list = []


def install(api: object) -> object:
    _INSTALLED.append(api)
    return api


def installed() -> object:
    assert _INSTALLED, "no fake pod api was installed, so this test would talk to a real cluster"
    return _INSTALLED[-1]


def reset() -> None:
    _INSTALLED.clear()
