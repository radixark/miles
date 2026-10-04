"""Immutable, independently framed canonical publications for direct GPU apply.

This is a new wire format. The disk-delta checkpoint patcher must never consume
it. GPU Snappy followed by GPU Zstd is the sole matrix payload contract.
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
# 2 MiB is a producer benchmark profile, not a streaming-receiver capability.
_FRAME_SIZES = (1 << 16, FRAME_BYTES, 1 << 21)
CODEC = "snappy-zstd"
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


def configured_codec() -> str:
    codec = os.environ.get("GPU_DELTA_CODEC", CODEC)
    if codec != CODEC:
        raise ValueError("Expected GPU_DELTA_CODEC=snappy-zstd")
    return codec


def _bytes_view(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if not value.flags.c_contiguous:
            raise ValueError("Canonical buffers must be contiguous")
        return value.reshape(-1).view(np.uint8)
    return np.frombuffer(value, dtype=np.uint8)


def _check_frame_bytes(frame_bytes):
    if type(frame_bytes) is not int or frame_bytes not in _FRAME_SIZES:
        raise ValueError("GPU-delta frame_bytes must be 64 KiB, 1 MiB or 2 MiB")


def selected_bytes(data, shape: list[int], dtype: str, slices: list[list[int]]) -> np.ndarray:
    """C-order canonical selected-view bytes; the final axis holds storage bytes."""
    raw = _bytes_view(data)
    itemsize = DTYPE_BYTES[dtype]
    if len(slices) != len(shape) or raw.size != math.prod(shape) * itemsize:
        raise ValueError("Canonical view shape/byte count mismatch")
    for size, bounds in zip(shape, slices, strict=True):
        if len(bounds) != 2 or any(type(x) is not int for x in bounds) or not 0 <= bounds[0] <= bounds[1] <= size:
            raise ValueError("Invalid canonical half-open view bounds")
    selection = tuple(slice(start, end) for start, end in slices) + (slice(None),)
    return np.ascontiguousarray(raw.reshape(*shape, itemsize)[selection]).reshape(-1)


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


def _validate_gpu_outer(entry, outer, payload, frame_bytes):
    """Validate metadata without reading unwrapped Snappy buffers back to the CPU."""
    frames = entry["frames"]
    if not frames:
        if outer is not None or len(payload) or entry["changed_bytes"]:
            raise ValueError("Unchanged GPU outer tensor must have no payload")
        return
    if not isinstance(outer, dict) or not entry["changed_bytes"]:
        raise ValueError("Changed GPU outer tensor requires an outer description")
    end, inner_end = 0, 0
    for frame in frames:
        offset, size = frame["decoded_offset"], frame["decoded_bytes"]
        encoded_offset, encoded_size = frame["encoded_offset"], frame["encoded_bytes"]
        if (
            type(offset) is not int
            or type(size) is not int
            or offset < end
            or offset % frame_bytes
            or size != min(frame_bytes, entry["nbytes"] - offset)
            or size <= 0
            or type(encoded_offset) is not int
            or encoded_offset != (inner_end + 15) // 16 * 16
            or type(encoded_size) is not int
            or encoded_size <= 0
            or encoded_size > 32 + size + size // 6
        ):
            raise ValueError("Invalid GPU outer inner frame")
        end, inner_end = offset + size, encoded_offset + encoded_size
    if outer.get("decoded_bytes") != inner_end or outer.get("encoded_bytes") != len(payload):
        raise ValueError("Invalid GPU outer arena size")
    encoded_end, decoded_end = 0, 0
    for frame in outer.get("frames", []):
        offset, size = frame["decoded_offset"], frame["decoded_bytes"]
        encoded_offset, encoded_size = frame["encoded_offset"], frame["encoded_bytes"]
        if (
            type(offset) is not int
            or offset != decoded_end
            or type(size) is not int
            or size != min(FRAME_BYTES, inner_end - offset)
            or size <= 0
            or type(encoded_offset) is not int
            or encoded_offset != (encoded_end + 15) // 16 * 16
            or type(encoded_size) is not int
            or encoded_size <= 0
        ):
            raise ValueError("Invalid independently framed GPU outer range")
        encoded_end, decoded_end = encoded_offset + encoded_size, offset + size
    if decoded_end != inner_end or encoded_end != len(payload):
        raise ValueError("Incomplete GPU outer frame coverage")


class PublicationWriter:
    """One owner's append-only payload; manifest is sealed after all owners finish.

    A CPU worker writes raw scalar/vector targets while matrix compression runs
    on the GPU. The lock serializes only final byte appends and their CPU hashes.
    Partial files are preserved on failure and never overwritten by a retry.
    """

    def __init__(
        self,
        directory,
        stream_id: str,
        base_version: int,
        target_version: int,
        plan_digest: str,
        owner: int = 0,
        publication_id: str | None = None,
        frame_bytes: int = FRAME_BYTES,
    ):
        if (
            not stream_id
            or type(base_version) is not int
            or base_version < 0
            or type(target_version) is not int
            or target_version != base_version + 1
        ):
            raise ValueError("GPU-delta versions must be consecutive")
        _check_frame_bytes(frame_bytes)
        self.frame_bytes = frame_bytes
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.outer_metrics = dict(outer_hash_write_s=0.0, outer_input_bytes=0, outer_output_bytes=0)
        self.metadata = {
            "protocol_version": 4,
            "codec": CODEC,
            "frame_bytes": frame_bytes,
            "stream_id": stream_id,
            "publication_id": publication_id or uuid.uuid4().hex,
            "base_version": base_version,
            "target_version": target_version,
            "plan_digest": plan_digest,
            "payload_checksum_format": "sha256",
        }
        self._filename = f"owner-{owner:05d}.bin"
        self._file = (self.directory / self._filename).open("xb")
        self._hash = hashlib.sha256()
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._closed = False

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
            if self._closed or name in self._entries:
                raise ValueError("Publication is sealed or tensor was already published")
            if entry["changed_bytes"]:
                payload = memoryview(current)
                padding = bytes((-self._file.tell()) % 16)
                self._file.write(padding)
                self._hash.update(padding)
                entry["raw"] = dict(file=self._filename, encoded_offset=self._file.tell(), encoded_bytes=len(payload))
                self._file.write(payload)
                self._hash.update(payload)
            self._entries[name] = entry
        return entry

    def add_gpu_outer_tensor(self, name, frames, payload, outer, changed_bytes, dtype, shape, views=None):
        """Publish already wrapped GPU bytes; only the final wire bytes are CPU hashed."""
        entry = tensor_metadata(name, dtype=dtype, shape=shape, views=views)
        if type(changed_bytes) is not int or not 0 <= changed_bytes <= entry["nbytes"]:
            raise ValueError("Invalid changed-byte count")
        entry["changed_bytes"] = changed_bytes
        entry["frames"] = [dict(frame) for frame in frames]
        _validate_gpu_outer(entry, outer, payload, self.frame_bytes)
        with self._lock:
            if self._closed or name in self._entries:
                raise ValueError("Publication is sealed or tensor was already published")
            if outer is not None:
                self._write_outer_bytes(bytes((-self._file.tell()) % 16))
                entry["outer"] = dict(outer, file=self._filename, encoded_offset=self._file.tell())
                self._write_outer_bytes(payload)
                self.outer_metrics["outer_input_bytes"] += outer["decoded_bytes"]
                self.outer_metrics["outer_output_bytes"] += len(payload)
            self._entries[name] = entry
        return entry

    def _write_outer_bytes(self, data):
        if not data:
            return
        started = time.monotonic()
        self._file.write(data)
        self._hash.update(data)
        self.outer_metrics["outer_hash_write_s"] += time.monotonic() - started

    def finish_shard(self) -> dict:
        with self._lock:
            if self._closed:
                raise RuntimeError("Publication shard was already sealed")
            self._file.flush()
            os.fsync(self._file.fileno())
            size = self._file.tell()
            self._file.close()
            self._closed = True
            return {
                "metadata": self.metadata,
                "files": [{"name": self._filename, "nbytes": size, "sha256": self._hash.hexdigest()}],
                "tensors": list(self._entries.values()),
            }

    def finish(self) -> dict:
        return seal_publication(self.directory, [self.finish_shard()])

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._file.close()
                self._closed = True


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
