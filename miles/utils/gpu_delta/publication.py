"""Immutable, independently framed canonical publications for direct GPU apply.

All mutable tensors carry XOR masks. Matrix masks use GPU Snappy/LZ4/Bitcomp
frames with optional GPU Zstd wrapping; scalar and vector masks are uncompressed.
These publications are not checkpoint patches for the disk-delta loader.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import orjson
from safetensors.numpy import load as load_columns
from safetensors.numpy import save as save_columns

FRAME_BYTES = 1 << 20
CODECS = {
    "snappy-zstd": ("snappy", "zstd"),
    "lz4-zstd": ("lz4", "zstd"),
    "lz4": ("lz4", None),
    "bitcomp": ("bitcomp", None),
    "bitcomp-zstd": ("bitcomp", "zstd"),
}
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


def _checkpoint_locations(checkpoint):
    """Select indexed checkpoint shards and reject ambiguous tensor names."""
    checkpoint = Path(checkpoint)
    index = checkpoint / "model.safetensors.index.json"
    files = (
        [checkpoint / name for name in sorted(set(json.loads(index.read_text())["weight_map"].values()))]
        if index.exists()
        else sorted(checkpoint.glob("*.safetensors"))
    )
    locations = {}
    for path in files:
        with path.open("rb") as source:
            header_bytes = struct.unpack("<Q", source.read(8))[0]
            header = json.loads(source.read(header_bytes))
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if name in locations:
                raise ValueError(f"Ambiguous canonical checkpoint tensor: {name}")
            start, end = metadata["data_offsets"]
            locations[name] = (
                path,
                8 + header_bytes + start,
                end - start,
                metadata["dtype"],
                tuple(metadata["shape"]),
            )
    return locations


def make_checkpoint_reader(checkpoint):
    """Read bytes from the same indexed prior used by the canonical contract."""
    locations = _checkpoint_locations(checkpoint)

    def read(name, expected_dtype, expected_shape):
        path, offset, size, dtype, shape = locations[name]
        if dtype != expected_dtype or shape != expected_shape:
            raise ValueError(f"Checkpoint tensor metadata changed: {name}")
        with path.open("rb") as source:
            source.seek(offset)
            return np.frombuffer(source.read(size), dtype=np.uint8)

    return read


def read_checkpoint_plan(checkpoint):
    """Canonical bytes agreed by the immutable HF checkpoint, not a receiver.

    Frozen draft weights, computed rotary buffers and static activation
    calibration are not trainer updates. Rank slicing and backend layouts are
    receiver-local and never appear in this contract.
    """
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text())
    quantization = config.get("quantization_config") or {}
    block_fp8 = quantization.get("quant_method") == "fp8" and quantization.get("weight_block_size") == [128, 128]
    locations = _checkpoint_locations(checkpoint)
    plan = []
    for name, (_, _, _, dtype, shape) in sorted(locations.items()):
        layer = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if (layer and int(layer[1]) >= config["num_hidden_layers"]) or name.endswith(
            ("rotary_emb.inv_freq", ".input_scale")
        ):
            continue
        spec = dict(
            name=name, dtype=dtype, shape=list(shape), encoding="xor_raw_bytes" if len(shape) <= 1 else "xor_bytes"
        )
        weight_name = name.removesuffix("_scale_inv")
        projection = re.search(r"\.(?:experts\.\d+|shared_experts)\.(?:gate|up|down)_proj\.weight$", weight_name)
        if block_fp8 and projection and locations[weight_name][3] == "F8_E4M3":
            spec["format"] = "fp8-block128-ue8m0"
            if name.endswith(".weight_scale_inv"):
                if dtype == "F32":
                    spec["checkpoint_dtype"] = dtype
                elif dtype != "U8":
                    raise ValueError(f"Block128 scales require exact UE8M0 storage: {name}")
                spec.update(dtype="U8", encoding="xor_raw_bytes")
            elif not name.endswith(".weight") or dtype != "F8_E4M3":
                raise ValueError(f"Unsupported block128 canonical tensor: {name}")
        plan.append(spec)
    if not plan:
        raise ValueError("GPU-delta checkpoint has no canonical tensors")
    return plan


def plan_digest(plan):
    return sha256(canonical_json(sorted(plan, key=lambda spec: spec["name"])))


def read_canonical_tensor(reader, name, spec):
    """Read exact checkpoint bytes, with explicit lossless UE8M0 normalization."""
    dtype = spec.get("checkpoint_dtype", spec["dtype"])
    if dtype != spec["dtype"] and not (
        spec.get("format") == "fp8-block128-ue8m0" and dtype == "F32" and spec["dtype"] == "U8"
    ):
        raise ValueError(f"Unsupported GPU-delta checkpoint representation for {name!r}")
    raw = reader(name, expected_dtype=dtype, expected_shape=tuple(spec["shape"]))
    if dtype == spec["dtype"]:
        if spec.get("format") == "fp8-block128-ue8m0" and dtype == "U8" and raw.max(initial=0) == 255:
            raise ValueError(f"Checkpoint scale {name!r} contains UE8M0 NaN")
        return raw
    bits = np.frombuffer(raw, dtype="<u4")
    exponent = (bits >> 23) & 255
    normal = ((bits & 0x807FFFFF) == 0) & (exponent > 0) & (exponent < 255)
    # UE8M0 exponent zero is 2**-127, an exactly representable FP32 subnormal.
    smallest = bits == 0x00400000
    if not np.all(normal | smallest):
        raise ValueError(f"Checkpoint scale {name!r} is not exact positive finite UE8M0")
    return exponent.astype(np.uint8)


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(data) -> str:
    return hashlib.sha256(data).hexdigest()


def configured_codec(initial_sync: bool = False) -> str:
    variable = "GPU_DELTA_INITIAL_SYNC_CODEC" if initial_sync else "GPU_DELTA_CODEC"
    codec = os.environ.get(variable, "bitcomp-zstd")
    if codec not in CODECS:
        raise ValueError(f"Expected {variable} to be one of {tuple(CODECS)}")
    return codec


def configured_skip_payload_hash() -> bool:
    return os.environ.get("GPU_DELTA_SKIP_PAYLOAD_HASH") == "1"


def tensor_metadata(name, dtype, shape, encoding="xor_bytes", format=None, checkpoint_dtype=None):
    """Describe canonical bytes without any receiver topology or layout."""
    metadata = {
        "name": name,
        "dtype": dtype,
        "shape": list(shape),
        "nbytes": math.prod(shape) * DTYPE_BYTES[dtype],
        "byte_order": "little",
        "encoding": encoding,
        "frames": [],
    }
    if format is not None:
        metadata["format"] = format
    if checkpoint_dtype is not None:
        metadata["checkpoint_dtype"] = checkpoint_dtype
    return metadata


def _write_exclusive(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


class PublicationWriter:
    """One owner's append-only payload; manifest is sealed after all owners finish.

    A CPU worker writes scalar/vector XOR masks while matrix compression runs
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
        if codec not in CODECS:
            raise ValueError(f"GPU-delta codec must be one of {tuple(CODECS)}")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.payload_metrics = dict(matrix_hash_write_s=0.0, matrix_inner_arena_bytes=0, matrix_payload_bytes=0)
        self._hash = None if configured_skip_payload_hash() else hashlib.sha256()
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

    def add_raw_xor_tensor(self, name, mask, dtype, shape, format=None, checkpoint_dtype=None):
        """Write an uncompressed scalar/vector XOR mask, omitting all-zero masks."""
        entry = tensor_metadata(name, dtype, shape, "xor_raw_bytes", format, checkpoint_dtype)
        mask = np.frombuffer(mask, dtype=np.uint8)
        if mask.size != entry["nbytes"]:
            raise ValueError(f"Canonical tensor byte count differs for {name}")
        entry["changed_bytes"] = int(np.count_nonzero(mask))
        with self._lock:
            if entry["changed_bytes"]:
                payload = memoryview(mask)
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

    def add_encoded_tensor(
        self, name, frames, payload, outer, changed_bytes, dtype, shape, format=None, checkpoint_dtype=None
    ):
        """Publish finalized matrix bytes; optional CPU hashing covers only final wire bytes."""
        entry = tensor_metadata(name, dtype, shape, "xor_bytes", format, checkpoint_dtype)
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
    return write_publication_metadata(directory, metadata | {"tensors": tensors, "files": files})


_CATALOG_FIELDS = ("name", "dtype", "shape", "nbytes", "encoding", "byte_order", "format", "checkpoint_dtype")
_FRAME_FIELDS = ("encoded_offset", "encoded_bytes", "decoded_offset", "decoded_bytes")


def write_publication_metadata(directory, manifest):
    """Seal readable static catalog and named numeric columns, then the manifest.

    Payload files already exist. This also supports explicit metadata-only conversion
    of saved benchmark publications; receivers accept only the current wire format.
    """
    directory = Path(directory)
    entries = sorted(manifest["tensors"], key=lambda entry: entry["name"])
    files = sorted(manifest["files"], key=lambda entry: entry["name"])
    file_ids = {entry["name"]: i for i, entry in enumerate(files)}
    catalog = {
        "plan_digest": manifest["plan_digest"],
        "tensors": [{key: entry[key] for key in _CATALOG_FIELDS if key in entry} for entry in entries],
    }
    columns = {
        "tensors." + key: []
        for key in (
            "changed_bytes",
            "payload_file",
            "payload_offset",
            "payload_nbytes",
            "inner_nbytes",
            "frame_end",
            "outer_end",
            "gap_end",
        )
    }
    columns.update({group + "." + field: [] for group in ("frames", "outer_frames") for field in _FRAME_FIELDS})
    columns.update({"gaps.offset": [], "gaps.nbytes": []})
    for entry in entries:
        raw = entry["encoding"] == "xor_raw_bytes"
        payload = entry.get("raw" if raw else "outer")
        values = {
            "changed_bytes": entry["changed_bytes"],
            "payload_file": file_ids[payload["file"]] if payload else -1,
            "payload_offset": payload["encoded_offset"] if payload else 0,
            "payload_nbytes": payload["encoded_bytes"] if payload else 0,
            "inner_nbytes": payload["decoded_bytes"] if payload and not raw else 0,
        }
        for key, value in values.items():
            columns["tensors." + key].append(value)
        for group, frames in (
            ("frames", entry["frames"]),
            ("outer_frames", payload.get("frames", []) if payload else []),
        ):
            for field in _FRAME_FIELDS:
                columns[group + "." + field].extend(frame[field] for frame in frames)
        columns["tensors.frame_end"].append(len(columns["frames.encoded_offset"]))
        columns["tensors.outer_end"].append(len(columns["outer_frames.encoded_offset"]))
        if not raw:
            cursor = 0
            for frame in entry["frames"]:
                if cursor < frame["decoded_offset"]:
                    columns["gaps.offset"].append(cursor)
                    columns["gaps.nbytes"].append(frame["decoded_offset"] - cursor)
                cursor = frame["decoded_offset"] + frame["decoded_bytes"]
            if cursor < entry["nbytes"]:
                columns["gaps.offset"].append(cursor)
                columns["gaps.nbytes"].append(entry["nbytes"] - cursor)
        columns["tensors.gap_end"].append(len(columns["gaps.offset"]))
    header = {key: value for key, value in manifest.items() if key not in {"tensors", "files"}}
    records = {}
    for kind, filename, content in (
        ("catalog", "catalog.json", orjson.dumps(catalog, option=orjson.OPT_SORT_KEYS)),
        (
            "metadata",
            "metadata.safetensors",
            save_columns({key: np.asarray(values, dtype="<i8") for key, values in columns.items()}),
        ),
    ):
        _write_exclusive(directory / filename, content)
        records[kind] = {"file": filename, "nbytes": len(content), "sha256": sha256(content)}
    content = orjson.dumps(header | {"files": files, **records}, option=orjson.OPT_SORT_KEYS)
    temporary = directory / "manifest.json.pending"
    _write_exclusive(temporary, content)
    os.link(temporary, directory / "manifest.json")
    temporary.unlink()
    return header | {"manifest_path": str((directory / "manifest.json").resolve()), "manifest_sha256": sha256(content)}


def read_publication_metadata(path):
    """Expand named columns for benchmark/oracle inspection, not receiver preparation."""
    path = Path(path)
    manifest = orjson.loads(path.read_bytes())
    values = {}
    for kind in ("catalog", "metadata"):
        record = manifest[kind]
        source = (path.parent / record["file"]).resolve(strict=True)
        if source.parent != path.parent.resolve() or source.stat().st_size != record["nbytes"]:
            raise ValueError("GPU-delta metadata path or size differs")
        raw = source.read_bytes()
        if sha256(raw) != record["sha256"]:
            raise ValueError("GPU-delta metadata SHA256 differs")
        values[kind] = orjson.loads(raw) if kind == "catalog" else load_columns(raw)
    columns = values["metadata"]
    entries = values["catalog"]["tensors"]
    starts = {key: np.r_[0, columns["tensors." + key][:-1]] for key in ("frame_end", "outer_end")}

    def frames(group, end, i):
        return [
            {field: int(columns[group + "." + field][j]) for field in _FRAME_FIELDS}
            for j in range(int(starts[end][i]), int(columns["tensors." + end][i]))
        ]

    for i, entry in enumerate(entries):
        entry["changed_bytes"] = int(columns["tensors.changed_bytes"][i])
        entry["frames"] = frames("frames", "frame_end", i)
        file_id = int(columns["tensors.payload_file"][i])
        if file_id >= 0:
            payload = {
                "file": manifest["files"][file_id]["name"],
                "encoded_offset": int(columns["tensors.payload_offset"][i]),
                "encoded_bytes": int(columns["tensors.payload_nbytes"][i]),
            }
            if entry["encoding"] == "xor_raw_bytes":
                entry["raw"] = payload
            else:
                entry["outer"] = payload | {
                    "decoded_bytes": int(columns["tensors.inner_nbytes"][i]),
                    "frames": frames("outer_frames", "outer_end", i),
                }
    return {key: value for key, value in manifest.items() if key not in {"catalog", "metadata"}} | {"tensors": entries}


def write_checkpoint_ready(directory, descriptor):
    """Publish completion after the payload and any paired checkpoint are sealed."""
    path = Path(directory) / "READY.json"
    _write_exclusive(path, canonical_json(descriptor))
