"""Stub sgl_kernel.kvcacheio — see sgl_kernel/__init__.py for rationale."""

from unittest.mock import MagicMock


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return MagicMock(name=f"sgl_kernel.kvcacheio.{name}")
