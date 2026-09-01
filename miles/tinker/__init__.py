"""Tinker 0.26.1 compatibility boundary; the only package (plus the tinker driver) allowed to import tinker."""

import importlib.metadata
import sys

TINKER_PINNED_VERSION = "0.26.1"


# This module must stay SDK-import-free so the gate can run (and fail cleanly) on any Python.
def ensure_tinker_runtime() -> None:
    """Fail fast when the tinker frontend cannot run here, without importing tinker."""
    if sys.version_info < (3, 11):
        raise RuntimeError(f"frontend=tinker requires Python >= 3.11, got {sys.version.split()[0]}")
    try:
        installed = importlib.metadata.version("tinker")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError(
            f"frontend=tinker requires tinker=={TINKER_PINNED_VERSION}; the SDK is not installed"
        ) from None
    if installed != TINKER_PINNED_VERSION:
        raise RuntimeError(f"frontend=tinker requires tinker=={TINKER_PINNED_VERSION}, got {installed}")
