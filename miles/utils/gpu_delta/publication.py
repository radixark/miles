"""Immutable, independently framed canonical publications for direct GPU apply.

Matrix payloads contain GPU Snappy/LZ4 frames with optional GPU Zstd wrapping.
These publications are not checkpoint patches for the disk-delta loader.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path

import numpy as np
import orjson

from miles.utils.disk_delta import _tensor_locations

FRAME_BYTES = 1 << 20
CODECS = {"snappy-zstd": ("snappy", "zstd"), "lz4-zstd": ("lz4", "zstd"), "lz4": ("lz4", None)}
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2FNUZ": 1,
    "U16": 2,
    "I16": 2,
    "BF16": 2,
    "F16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


# This feature fixes its checkpoint for the stream lifetime. Ordinary disk-delta
# readers keep their existing uncached indexing when checkpoints change in place.
_immutable_tensor_locations = cache(_tensor_locations)


def checkpoint_tensor_layout(ckpt_dir: str, name: str) -> tuple[str, tuple[int, ...]]:
    """Return the immutable startup checkpoint's declared dtype and shape."""
    _, _, _, dtype, shape = _immutable_tensor_locations(ckpt_dir)[name]
    return dtype, shape


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(data) -> str:
    return hashlib.sha256(data).hexdigest()


def configured_codec(initial_sync: bool = False) -> str:
    variable = "GPU_DELTA_INITIAL_SYNC_CODEC" if initial_sync else "GPU_DELTA_CODEC"
    codec = os.environ.get(variable, "lz4-zstd" if initial_sync else "snappy-zstd")
    if codec not in CODECS:
        raise ValueError(f"Expected {variable}=snappy-zstd, lz4-zstd or lz4")
    return codec


def _bytes_view(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if not value.flags.c_contiguous:
            raise ValueError("Canonical buffers must be contiguous")
        return value.reshape(-1).view(np.uint8)
    return np.frombuffer(value, dtype=np.uint8)


def tensor_metadata(name: str, dtype: str, shape: list[int], views=None, encoding="xor_bytes") -> dict:
    """Canonical tensor schema for compressed matrices and raw scalar/vector targets."""
    if not name or dtype not in DTYPE_BYTES or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Invalid canonical tensor schema")
    if encoding not in ("xor_bytes", "raw_bytes"):
        raise ValueError("Unsupported gpu-delta encoding")
    if encoding == "raw_bytes" and len(shape) > 1:
        raise ValueError("Raw replacements require scalar or vector tensors")
    nbytes = math.prod(shape) * DTYPE_BYTES[dtype]
    definitions = views if views is not None else [{"id": "full", "slices": [[0, n] for n in shape]}]
    if len({v["id"] for v in definitions}) != len(definitions):
        raise ValueError("Duplicate canonical view id")
    entry = {
        "name": name,
        "dtype": dtype,
        "shape": list(shape),
        "nbytes": nbytes,
        "byte_order": "little",
        "encoding": encoding,
        "views": [],
        "frames": [],
    }
    for view in definitions:
        slices = view["slices"]
        if len(slices) != len(shape):
            raise ValueError("Canonical view dimensions differ")
        for size, bounds in zip(shape, slices, strict=True):
            if len(bounds) != 2 or any(type(x) is not int for x in bounds) or not 0 <= bounds[0] <= bounds[1] <= size:
                raise ValueError("Invalid canonical half-open view bounds")
        entry["views"].append({"id": view["id"], "slices": slices})
    return entry


def _write_exclusive(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


class PublicationWriter:
    """One owner's append-only payload; manifest is sealed after all owners finish.

    A CPU worker writes raw scalar/vector targets while matrix compression runs
    on the GPU. The lock serializes final byte appends and optional CPU hashes.
    Partial files are preserved on failure and never overwritten by a retry.
    """

    def __init__(
        self,
        directory,
        stream_id: str,
        base_version: int,
        target_version: int,
        plan_digest: str,
        codec: str,
        owner: int = 0,
        publication_id: str | None = None,
        frame_bytes: int = FRAME_BYTES,
    ):
        if type(frame_bytes) is not int or not 0 < frame_bytes <= 4 << 20:
            raise ValueError("GPU-delta frame_bytes must be a positive integer at most 4 MiB")
        self.frame_bytes = frame_bytes
        if codec not in CODECS:
            raise ValueError("GPU-delta codec must be snappy-zstd, lz4-zstd or lz4")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.payload_metrics = dict(matrix_hash_write_s=0.0, matrix_inner_arena_bytes=0, matrix_payload_bytes=0)
        self._hash = None if os.environ.get("GPU_DELTA_SKIP_PAYLOAD_HASH") == "1" else hashlib.sha256()
        self.metadata = {
            "codec": codec,
            "frame_bytes": frame_bytes,
            "stream_id": stream_id,
            "publication_id": publication_id or uuid.uuid4().hex,
            "base_version": base_version,
            "target_version": target_version,
            "plan_digest": plan_digest,
            "payload_checksum_format": "none" if self._hash is None else "sha256",
        }
        self._filename = f"owner-{owner:05d}.bin"
        self._file = (self.directory / self._filename).open("xb")
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}

    def add_raw_tensor(self, name: str, old, new, dtype: str, shape: list[int], views=None):
        """Write complete scalar/vector targets without XOR, frames or codecs."""
        entry = tensor_metadata(name, dtype=dtype, shape=shape, views=views, encoding="raw_bytes")
        previous, current = _bytes_view(old), _bytes_view(new)
        if previous.size != entry["nbytes"] or current.size != entry["nbytes"]:
            raise ValueError(f"Canonical tensor byte count differs for {name}")
        # CPU-only comparison preserves update-density metrics and omits
        # unchanged scalars. Changed tensors replace all bytes without a mask.
        entry["changed_bytes"] = int(np.count_nonzero(previous != current))
        with self._lock:
            if entry["changed_bytes"]:
                payload = memoryview(current)
                padding = bytes((-self._file.tell()) % 16)
                self._file.write(padding)
                if self._hash is not None:
                    self._hash.update(padding)
                entry["raw"] = dict(file=self._filename, encoded_offset=self._file.tell(), encoded_bytes=len(payload))
                self._file.write(payload)
                if self._hash is not None:
                    self._hash.update(payload)
            self._entries[name] = entry
        return entry

    def add_encoded_tensor(self, name, frames, payload, outer, changed_bytes, dtype, shape, views=None):
        """Publish finalized matrix bytes; optional CPU hashing covers only final wire bytes."""
        entry = tensor_metadata(name, dtype=dtype, shape=shape, views=views)
        entry["changed_bytes"] = changed_bytes
        # The encoder owns frame construction and checks native sizes/statuses.
        entry["frames"] = frames
        with self._lock:
            if outer is not None:
                self._write_encoded_bytes(bytes((-self._file.tell()) % 16))
                entry["outer"] = dict(outer, file=self._filename, encoded_offset=self._file.tell())
                self._write_encoded_bytes(payload)
                self.payload_metrics["matrix_inner_arena_bytes"] += outer["decoded_bytes"]
                self.payload_metrics["matrix_payload_bytes"] += len(payload)
            self._entries[name] = entry
        return entry

    def _write_encoded_bytes(self, data):
        if not data:
            return
        started = time.monotonic()
        self._file.write(data)
        if self._hash is not None:
            self._hash.update(data)
        self.payload_metrics["matrix_hash_write_s"] += time.monotonic() - started

    def finish_shard(self) -> dict:
        with self._lock:
            self._file.flush()
            os.fsync(self._file.fileno())
            size = self._file.tell()
            self._file.close()
            return {
                "metadata": self.metadata,
                "files": [
                    {
                        "name": self._filename,
                        "nbytes": size,
                        "sha256": self._hash.hexdigest() if self._hash is not None else None,
                    }
                ],
                "tensors": list(self._entries.values()),
            }

    def finish(self) -> dict:
        return seal_publication(self.directory, [self.finish_shard()])

    def close(self) -> None:
        with self._lock:
            self._file.close()


def seal_publication(directory, shards: Iterable[Mapping]) -> dict:
    """Publish manifest last; reject conflicting owner metadata or duplicate names."""
    shards = list(shards)
    if not shards:
        raise ValueError("A publication requires at least one owner shard")
    metadata = dict(shards[0]["metadata"])
    if any(s["metadata"] != metadata for s in shards):
        raise ValueError("Owner publication identities differ")
    tensors = [t for shard in shards for t in shard["tensors"]]
    if len({t["name"] for t in tensors}) != len(tensors):
        raise ValueError("Duplicate canonical tensor ownership")
    files = [f for shard in shards for f in shard["files"]]
    if len({f["name"] for f in files}) != len(files):
        raise ValueError("Duplicate owner payload path")
    manifest = metadata | {
        "tensors": sorted(tensors, key=lambda t: t["name"]),
        "files": sorted(files, key=lambda f: f["name"]),
    }
    # The descriptor authenticates these exact bytes. Keep canonical_json
    # unchanged for the public plan-digest contract.
    content = orjson.dumps(manifest, option=orjson.OPT_SORT_KEYS)
    directory = Path(directory)
    temporary = directory / "manifest.json.pending"
    _write_exclusive(temporary, content)
    # link is exclusive: even another completed writer cannot replace a version.
    os.link(temporary, directory / "manifest.json")
    temporary.unlink()
    descriptor = metadata | {
        "manifest_path": str((directory / "manifest.json").resolve()),
        "manifest_sha256": sha256(content),
    }
    return descriptor


def write_checkpoint_ready(directory, descriptor):
    """Publish paired completion only after the Megatron checkpoint writer drains."""
    path = Path(directory) / "READY.json"
    _write_exclusive(path, canonical_json(descriptor))
