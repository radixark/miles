"""Immutable, independently framed canonical publications for direct GPU apply.

This is a new wire format. The disk-delta checkpoint patcher must never consume
it. Codec choice changes encoded bytes, not the canonical update or its hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import zstandard

FRAME_BYTES = 1 << 20
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


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(data) -> str:
    return hashlib.sha256(data).hexdigest()


def settings_from_env() -> tuple[str, str]:
    codec = os.environ.get("WEIGHT_DELTA_CODEC", "zstd")
    staging = os.environ.get("WEIGHT_DELTA_STAGING", "full")
    if codec not in ("zstd", "snappy", "none") or staging not in ("full", "tensor"):
        raise ValueError("Expected WEIGHT_DELTA_CODEC=zstd|snappy|none and WEIGHT_DELTA_STAGING=full|tensor")
    return codec, staging


def _bytes_view(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if not value.flags.c_contiguous:
            raise ValueError("Canonical buffers must be contiguous")
        return value.reshape(-1).view(np.uint8)
    return np.frombuffer(value, dtype=np.uint8)


def selected_bytes(data, *, shape: list[int], dtype: str, slices: list[list[int]]) -> np.ndarray:
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


def encode_tensor(
    name: str,
    old,
    new,
    *,
    dtype: str,
    shape: list[int],
    codec: str,
    views: list[dict] | None = None,
    encoding: str = "xor_bytes",
) -> tuple[dict, list[bytes]]:
    """Encode one known W0/W1 tensor without changing either caller's buffer."""
    previous, current = _bytes_view(old), _bytes_view(new)
    if not name or dtype not in DTYPE_BYTES or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Invalid canonical tensor schema")
    nbytes = math.prod(shape) * DTYPE_BYTES[dtype]
    if previous.size != nbytes or current.size != nbytes:
        raise ValueError(f"Canonical tensor byte count differs for {name}")
    if encoding not in ("xor_bytes", "replace_bytes") or codec not in ("zstd", "snappy", "none"):
        raise ValueError("Unsupported gpu-delta encoding/codec")
    if codec == "snappy":
        # Optional at import time; the explicitly selected profile requires it.
        import snappy

        compress = snappy.compress
    elif codec == "zstd":
        compress = zstandard.ZstdCompressor(level=1, write_content_size=True).compress
    else:
        compress = bytes
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
    payloads, changed = [], 0
    for offset in range(0, nbytes, FRAME_BYTES):
        end = min(nbytes, offset + FRAME_BYTES)
        delta = np.bitwise_xor(previous[offset:end], current[offset:end])
        nonzero = int(np.count_nonzero(delta))
        changed += nonzero
        if encoding == "xor_bytes" and not nonzero:
            continue
        # Both CPU codecs consume contiguous buffers synchronously. Only raw
        # retained frames need an owned copy of the uncompressed input bytes.
        raw = memoryview(delta if encoding == "xor_bytes" else current[offset:end])
        payload = compress(raw)
        frame_codec = codec
        if len(payload) >= len(raw):
            if codec != "none":
                payload = bytes(raw)
            frame_codec = "none"
        entry["frames"].append(
            {
                "decoded_offset": offset,
                "decoded_bytes": len(raw),
                "encoded_bytes": len(payload),
                "codec": frame_codec,
                "encoded_sha256": sha256(payload),
            }
        )
        payloads.append(payload)
    entry["changed_bytes"] = changed
    return entry, payloads


def _write_exclusive(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


class PublicationWriter:
    """One owner's append-only payload; manifest is sealed after all owners finish.

    add_tensor may be called by bounded CPU encoding workers. The lock protects
    append offsets only; compression/hash work happens outside it. Partial files
    are preserved on failure and never overwritten by a retry.
    """

    def __init__(
        self,
        directory,
        *,
        stream_id: str,
        base_version: int,
        target_version: int,
        plan_digest: str,
        codec: str | None = None,
        owner: int = 0,
        publication_id: str | None = None,
    ):
        if (
            not stream_id
            or type(base_version) is not int
            or base_version < 0
            or type(target_version) is not int
            or target_version != base_version + 1
        ):
            raise ValueError("GPU-delta versions must be consecutive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.codec = codec or settings_from_env()[0]
        if self.codec not in ("zstd", "snappy", "none"):
            raise ValueError("Unknown gpu-delta codec")
        self.metadata = {
            "protocol_version": 2,
            "stream_id": stream_id,
            "publication_id": publication_id or uuid.uuid4().hex,
            "base_version": base_version,
            "target_version": target_version,
            "plan_digest": plan_digest,
            "payload_checksum_format": "sha256",
            "codec_profile": self.codec + "-independent-1mib-v1",
        }
        self._filename = f"owner-{owner:05d}.bin"
        self._file = (self.directory / self._filename).open("xb")
        self._hash = hashlib.sha256()
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._closed = False

    def add_tensor(self, name: str, old, new, *, dtype: str, shape: list[int], views=None, encoding="xor_bytes"):
        entry, payloads = encode_tensor(
            name, old, new, dtype=dtype, shape=shape, codec=self.codec, views=views, encoding=encoding
        )
        with self._lock:
            if self._closed or name in self._entries:
                raise ValueError("Publication is sealed or tensor was already published")
            for frame, payload in zip(entry["frames"], payloads, strict=True):
                padding = bytes((-self._file.tell()) % 16)
                self._file.write(padding)
                self._hash.update(padding)
                frame.update(file=self._filename, encoded_offset=self._file.tell())
                self._file.write(payload)
                self._hash.update(payload)
            self._entries[name] = entry
        return entry

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
    content = canonical_json(manifest)
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
