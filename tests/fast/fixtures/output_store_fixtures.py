"""An in-memory output store for replay tests: the package's real ``OutputStore`` over a backend that
keeps objects in a dict and records every read."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence

import numpy as np
import pybase64
import pytest

radix_transfer_plugins = pytest.importorskip("radix_transfer_plugins")

from radix_transfer_plugins import OutputStore, StorageBackend  # noqa: E402
from radix_transfer_plugins.stashes.token_output import TokenOutputStash  # noqa: E402


class MemoryBackend(StorageBackend):
    # Miles reads only the run's Mooncake; the fake stands in for it.
    name = "mooncake"
    config_type = None

    def __init__(self) -> None:
        self.parts: dict[str, bytes] = {}
        # One entry per get_many call: the (part key, bytes) it read.
        self.reads: list[list[tuple[str, int]]] = []

    def put_many(self, objects: Sequence[Mapping[str, memoryview]]) -> list[str]:
        keys = [f"miles-output-store/test/{uuid.uuid4().hex}" for _ in objects]
        for key, parts in zip(keys, objects, strict=True):
            for name, data in parts.items():
                self.parts[f"{key}/{name}"] = bytes(data)
        return keys

    def get_many(self, keys, *, parts):
        self.reads.append(
            [(f"{key}/{name}", nbytes) for key, named in zip(keys, parts, strict=True) for name, nbytes in named]
        )
        objects = []
        for key, named in zip(keys, parts, strict=True):
            got = {}
            for name, _ in named:
                part_key = f"{key}/{name}"
                if part_key not in self.parts:
                    raise KeyError(f"output store part not found: {part_key}")
                got[name] = memoryview(bytearray(self.parts[part_key]))
            objects.append(got)
        return objects

    def delete_many(self, keys, *, parts) -> None:
        for key, names in zip(keys, parts, strict=True):
            for name in names:
                self.parts.pop(f"{key}/{name}", None)

    def close(self) -> None:
        pass


def make_store() -> tuple[OutputStore, MemoryBackend]:
    backend = MemoryBackend()
    return OutputStore(backend=backend), backend


def put_response(store: OutputStore, **arrays: np.ndarray) -> dict:
    """Store one response's fields; returns the ref as it appears in ``meta_info``."""
    (ref,) = store.put_many([TokenOutputStash(**arrays)])
    return ref.to_builtins()


def b64(array: np.ndarray) -> str:
    return pybase64.b64encode(np.ascontiguousarray(array, dtype=np.int32).tobytes()).decode("ascii")


def mask_arrays(supports: list[list[int]], support_logprobs: list[list[float]] | None = None) -> dict[str, np.ndarray]:
    """The stored sampling-mask fields of an inline ``output_token_sampling_mask``; support mode adds the
    per-entry ``output_token_sampling_logprobs``."""
    arrays = {
        "output_token_sampling_mask_lengths": np.asarray([len(s) for s in supports], dtype=np.int32),
        "output_token_sampling_mask_token_ids": np.asarray([t for s in supports for t in s], dtype=np.int32),
    }
    if support_logprobs is not None:
        flat = [value for row in support_logprobs for value in row]
        arrays["output_token_sampling_logprobs"] = np.asarray(flat, dtype=np.float32)
    return arrays
