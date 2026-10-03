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
import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import zstandard

FRAME_BYTES = 1 << 20
# 2 MiB is a producer benchmark profile, not a streaming-receiver capability.
_FRAME_PROFILES = {1 << 16: "64kib", FRAME_BYTES: "1mib", 1 << 21: "2mib"}
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
    codec = os.environ.get("WEIGHT_DELTA_CODEC", "snappy")
    encoder = os.environ.get("WEIGHT_DELTA_ENCODER", "gpu")
    if codec not in ("zstd", "snappy") or encoder not in ("gpu", "cpu"):
        raise ValueError("Expected WEIGHT_DELTA_CODEC=zstd|snappy and WEIGHT_DELTA_ENCODER=gpu|cpu")
    if "WEIGHT_DELTA_STAGING" in os.environ:
        raise ValueError("WEIGHT_DELTA_STAGING was removed; GPU-delta receivers always stream tensors")
    return codec, encoder


def _inner_payload_layout(payloads):
    """Preserve payload leases; expose an existing aligned arena when possible."""
    views = [memoryview(payload).cast("B") for payload in payloads]
    offsets, end = [], 0
    for view in views:
        offset = (end + 15) // 16 * 16
        offsets.append(offset)
        end = offset + len(view)
    if len(views) == 1:
        return views, offsets, end, views[0]
    if views and all(view.obj is views[0].obj for view in views):
        parent = memoryview(views[0].obj).cast("B")
        parent_address = np.frombuffer(parent, dtype=np.uint8).ctypes.data
        start = np.frombuffer(views[0], dtype=np.uint8).ctypes.data - parent_address
        if (
            0 <= start
            and start + end <= len(parent)
            and all(
                np.frombuffer(view, dtype=np.uint8).ctypes.data == parent_address + start + offset
                for view, offset in zip(views, offsets, strict=True)
            )
        ):
            arena = parent[start : start + end]
            ends = [0] + [offset + len(view) for offset, view in zip(offsets[:-1], views[:-1], strict=True)]
            if all(not any(arena[end:offset]) for end, offset in zip(ends, offsets, strict=True)):
                return views, offsets, end, arena
    # Packed GPU batches normally lack inter-frame alignment padding. Stream
    # these immutable views plus tiny zero gaps instead of copying the tensor.
    return views, offsets, end, None


def _bytes_view(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if not value.flags.c_contiguous:
            raise ValueError("Canonical buffers must be contiguous")
        return value.reshape(-1).view(np.uint8)
    return np.frombuffer(value, dtype=np.uint8)


def _check_frame_bytes(frame_bytes):
    if type(frame_bytes) is not int or frame_bytes not in _FRAME_PROFILES:
        raise ValueError("GPU-delta frame_bytes must be 64 KiB, 1 MiB or 2 MiB")


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


def tensor_metadata(name: str, *, dtype: str, shape: list[int], views=None, encoding="xor_bytes") -> dict:
    """Canonical tensor schema shared by CPU and GPU encoders."""
    if not name or dtype not in DTYPE_BYTES or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Invalid canonical tensor schema")
    if encoding not in ("xor_bytes", "replace_bytes"):
        raise ValueError("Unsupported gpu-delta encoding")
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
    frame_bytes: int = FRAME_BYTES,
) -> tuple[dict, list[bytes]]:
    """Encode one known W0/W1 tensor without changing either caller's buffer."""
    previous, current = _bytes_view(old), _bytes_view(new)
    entry = tensor_metadata(name, dtype=dtype, shape=shape, views=views, encoding=encoding)
    nbytes = entry["nbytes"]
    if previous.size != nbytes or current.size != nbytes:
        raise ValueError(f"Canonical tensor byte count differs for {name}")
    if codec not in ("zstd", "snappy"):
        raise ValueError("Unsupported gpu-delta codec")
    _check_frame_bytes(frame_bytes)
    if codec == "snappy":
        # Optional at import time; the explicitly selected profile requires it.
        import snappy

        compress = snappy.compress
    else:
        compress = zstandard.ZstdCompressor(level=1, write_content_size=True).compress
    payloads, changed = [], 0
    for offset in range(0, nbytes, frame_bytes):
        end = min(nbytes, offset + frame_bytes)
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
    append offsets and the reused Snappy outer-Zstd context. Inner compression
    and hashing happen outside it. Partial files are preserved on failure and
    never overwritten by a retry.
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
        self.codec = codec or settings_from_env()[0]
        if self.codec not in ("zstd", "snappy"):
            raise ValueError("Unknown gpu-delta codec")
        self._outer_compressor = (
            zstandard.ZstdCompressor(level=1, threads=0, write_content_size=True) if self.codec == "snappy" else None
        )
        self.outer_metrics = dict(
            outer_compress_s=0.0,
            outer_hash_write_s=0.0,
            inner_hash_s=0.0,
            outer_input_bytes=0,
            outer_output_bytes=0,
            outer_contiguous_tensors=0,
            outer_streamed_tensors=0,
        )
        self.metadata = {
            "protocol_version": 3 if self.codec == "snappy" else 2,
            "stream_id": stream_id,
            "publication_id": publication_id or uuid.uuid4().hex,
            "base_version": base_version,
            "target_version": target_version,
            "plan_digest": plan_digest,
            "payload_checksum_format": "sha256",
            "codec_profile": f"{self.codec}-independent-{_FRAME_PROFILES[frame_bytes]}{'-zstd' if self.codec == 'snappy' else ''}-v1",
        }
        self._filename = f"owner-{owner:05d}.bin"
        self._file = (self.directory / self._filename).open("xb")
        self._hash = hashlib.sha256()
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._closed = False

    def add_tensor(self, name: str, old, new, *, dtype: str, shape: list[int], views=None, encoding="xor_bytes"):
        entry, payloads = encode_tensor(
            name,
            old,
            new,
            dtype=dtype,
            shape=shape,
            codec=self.codec,
            views=views,
            encoding=encoding,
            frame_bytes=self.frame_bytes,
        )
        return self._append_tensor(entry, payloads)

    def add_encoded_tensor(
        self, name, frames, payloads, *, changed_bytes, dtype, shape, views=None, encoding="xor_bytes"
    ):
        """Append GPU-produced frames; hash only encoded CPU buffers for transport."""
        entry = tensor_metadata(name, dtype=dtype, shape=shape, views=views, encoding=encoding)
        if type(changed_bytes) is not int or not 0 <= changed_bytes <= entry["nbytes"]:
            raise ValueError("Invalid changed-byte count")
        entry["changed_bytes"] = changed_bytes
        end, hash_s = 0, 0.0
        for frame, payload in zip(frames, payloads, strict=True):
            frame = dict(frame)
            offset, size = frame["decoded_offset"], frame["decoded_bytes"]
            if (
                type(offset) is not int
                or type(size) is not int
                or offset < end
                or offset % self.frame_bytes != 0
                or size != min(self.frame_bytes, entry["nbytes"] - offset)
                or size <= 0
                or (encoding == "replace_bytes" and offset != end)
            ):
                raise ValueError("Invalid independently framed canonical range")
            if frame["codec"] not in (self.codec, "none") or frame["encoded_bytes"] != len(payload):
                raise ValueError("Encoded frame profile or size mismatch")
            if not payload or (frame["codec"] == "none" and len(payload) != size):
                raise ValueError("Invalid raw frame byte count")
            started = time.monotonic() if self.codec == "snappy" else 0.0
            frame["encoded_sha256"] = sha256(payload)
            if self.codec == "snappy":
                hash_s += time.monotonic() - started
            entry["frames"].append(frame)
            end = offset + size
        if encoding == "replace_bytes" and end != entry["nbytes"]:
            raise ValueError("Replacement frames must cover the complete canonical tensor")
        return self._append_tensor(entry, payloads, inner_hash_s=hash_s)

    def _append_tensor(self, entry, payloads, *, inner_hash_s=0.0):
        name = entry["name"]
        with self._lock:
            if self._closed or name in self._entries:
                raise ValueError("Publication is sealed or tensor was already published")
            if self.codec == "snappy":
                self.outer_metrics["inner_hash_s"] += inner_hash_s
                self._append_outer(entry, payloads)
            else:
                for frame, payload in zip(entry["frames"], payloads, strict=True):
                    padding = bytes((-self._file.tell()) % 16)
                    self._file.write(padding)
                    self._hash.update(padding)
                    frame.update(file=self._filename, encoded_offset=self._file.tell())
                    self._file.write(payload)
                    self._hash.update(payload)
            self._entries[name] = entry
        return entry

    def _write_outer_bytes(self, data):
        if not data:
            return
        started = time.monotonic()
        self._file.write(data)
        self._hash.update(data)
        self.outer_metrics["outer_hash_write_s"] += time.monotonic() - started

    def _append_outer(self, entry, payloads):
        if not entry["frames"]:
            return
        views, offsets, size, arena = _inner_payload_layout(payloads)
        for frame, offset in zip(entry["frames"], offsets, strict=True):
            frame.update(file=self._filename, encoded_offset=offset)
        self._write_outer_bytes(bytes((-self._file.tell()) % 16))
        start = self._file.tell()
        if arena is not None:
            started = time.monotonic()
            encoded = self._outer_compressor.compress(arena)
            self.outer_metrics["outer_compress_s"] += time.monotonic() - started
            self._write_outer_bytes(encoded)
            self.outer_metrics["outer_contiguous_tensors"] += 1
        else:
            compressor = self._outer_compressor.compressobj(size=size)
            end = 0
            for view, offset in zip(views, offsets, strict=True):
                parts = (bytes(offset - end), view) if offset > end else (view,)
                for part in parts:
                    started = time.monotonic()
                    encoded = compressor.compress(part)
                    self.outer_metrics["outer_compress_s"] += time.monotonic() - started
                    self._write_outer_bytes(encoded)
                end = offset + len(view)
            started = time.monotonic()
            encoded = compressor.flush()
            self.outer_metrics["outer_compress_s"] += time.monotonic() - started
            self._write_outer_bytes(encoded)
            self.outer_metrics["outer_streamed_tensors"] += 1
        encoded_size = self._file.tell() - start
        entry["outer"] = dict(
            codec="zstd", file=self._filename, encoded_offset=start, encoded_bytes=encoded_size, decoded_bytes=size
        )
        self.outer_metrics["outer_input_bytes"] += size
        self.outer_metrics["outer_output_bytes"] += encoded_size

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
