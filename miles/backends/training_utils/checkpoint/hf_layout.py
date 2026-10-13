"""Write HF tensors into the base checkpoint's exact shard layout.

Every shard keeps the base file's name, header and size, so a model whose weights did not change is written byte for
byte as the base checkpoint. Each shard file has exactly one writer, so several workers write in parallel without two
of them ever writing one file.
"""

import json
import logging
import os
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_INDEX = "model.safetensors.index.json"
_SINGLE_SHARD = "model.safetensors"
_COPY_CHUNK_BYTES = 64 * 1024**2
_SAFETENSORS_DTYPES = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e5m2: "F8_E5M2",
}


@dataclass(frozen=True)
class CheckpointLayout:
    """Where an exported tensor lands in the base checkpoint.

    ``locate(name)`` returns ``(checkpoint_name, byte_offset)``: the base tensor the exported tensor belongs to and
    its byte offset inside it, so one base tensor can be written as several slices (e.g. one expert of a stacked
    expert tensor). Base tensors under ``frozen_prefixes`` are never trained; they are copied from the base checkpoint
    and skipped when the export carries them.
    """

    locate: Callable[[str], tuple[str, int]]
    frozen_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _TensorSpan:
    shard: str
    begin: int  # absolute file offsets
    end: int
    dtype: str
    shape: tuple[int, ...]


def _read_header(path: Path) -> tuple[bytes, dict]:
    with open(path, "rb") as f:
        prefix = f.read(8)
        (length,) = struct.unpack("<Q", prefix)
        header = f.read(length)
    return prefix + header, json.loads(header)


def _base_weight_map(base: Path) -> dict[str, str]:
    index = base / _INDEX
    if index.is_file():
        return json.loads(index.read_text())["weight_map"]
    _, header = _read_header(base / _SINGLE_SHARD)
    return {name: _SINGLE_SHARD for name in header if name != "__metadata__"}


class CheckpointLayoutWriter:
    """One worker's writer for the shards it owns (shard ``i`` of the sorted shard names belongs to worker
    ``i % world_size``); call :meth:`write` for every tensor, then :meth:`finish`."""

    def __init__(self, out_dir: Path, base: Path, layout: CheckpointLayout, *, rank: int, world_size: int) -> None:
        self._base = base
        self._out_dir = out_dir
        self._layout = layout
        self._weight_map = _base_weight_map(base)
        shards = sorted(set(self._weight_map.values()))
        self._owned = shards[rank::world_size]
        self._spans: dict[str, _TensorSpan] = {}
        self._fds: dict[str, int] = {}
        self._written: dict[str, list[tuple[int, int]]] = {}
        self._bytes = 0
        self._start = time.monotonic()
        for shard in self._owned:
            self._open_shard(shard)

    def _open_shard(self, shard: str) -> None:
        header_bytes, header = _read_header(self._base / shard)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            begin, end = entry["data_offsets"]
            self._spans[name] = _TensorSpan(
                shard=shard,
                begin=len(header_bytes) + begin,
                end=len(header_bytes) + end,
                dtype=entry["dtype"],
                shape=tuple(entry["shape"]),
            )
        fd = os.open(self._out_dir / shard, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd, (self._base / shard).stat().st_size)
        _pwrite_all(fd, header_bytes, 0)
        self._fds[shard] = fd

    def write(self, name: str, tensor: torch.Tensor) -> None:
        checkpoint_name, offset = self._layout.locate(name)
        if checkpoint_name not in self._weight_map:
            raise KeyError(f"exported tensor {name} maps to {checkpoint_name}, which the base checkpoint lacks")
        if checkpoint_name.startswith(self._layout.frozen_prefixes) or checkpoint_name not in self._spans:
            return
        span = self._spans[checkpoint_name]
        if _SAFETENSORS_DTYPES.get(tensor.dtype) != span.dtype:
            raise ValueError(f"{name}: dtype {tensor.dtype} differs from the base checkpoint's {span.dtype}")
        data = tensor.detach().contiguous().cpu().view(-1).view(torch.uint8).numpy()
        if offset == 0 and data.nbytes == span.end - span.begin and tuple(tensor.shape) != span.shape:
            raise ValueError(f"{name}: shape {tuple(tensor.shape)} differs from the base checkpoint's {span.shape}")
        if span.begin + offset + data.nbytes > span.end:
            raise ValueError(f"{name}: {data.nbytes} bytes at offset {offset} overflow {checkpoint_name}")
        _pwrite_all(self._fds[span.shard], data, span.begin + offset)
        self._written.setdefault(checkpoint_name, []).append((offset, offset + data.nbytes))
        self._bytes += data.nbytes

    def finish(self) -> None:
        """Copy the frozen tensors, check every other tensor was fully written, and close the shards."""
        try:
            for checkpoint_name, span in self._spans.items():
                if checkpoint_name.startswith(self._layout.frozen_prefixes):
                    self._copy_from_base(span)
                else:
                    _check_tiles(checkpoint_name, self._written.get(checkpoint_name, []), span.end - span.begin)
        finally:
            for fd in self._fds.values():
                os.close(fd)
        logger.info(
            f"Wrote {len(self._owned)} shards ({self._bytes / 1024**3:.1f} GiB) in {time.monotonic() - self._start:.1f} s"
        )

    def _copy_from_base(self, span: _TensorSpan) -> None:
        source = os.open(self._base / span.shard, os.O_RDONLY)
        try:
            for start in range(span.begin, span.end, _COPY_CHUNK_BYTES):
                chunk = os.pread(source, min(_COPY_CHUNK_BYTES, span.end - start), start)
                _pwrite_all(self._fds[span.shard], chunk, start)
        finally:
            os.close(source)


def _pwrite_all(fd: int, data, offset: int) -> None:
    # A single pwrite on Linux stops at about 2 GiB, below the size of an embedding table.
    view = memoryview(data).cast("B")
    while view:
        written = os.pwrite(fd, view, offset)
        view, offset = view[written:], offset + written


def _check_tiles(name: str, slices: list[tuple[int, int]], size: int) -> None:
    """The written ``[begin, end)`` byte slices must cover ``[0, size)`` exactly once."""
    position = 0
    for begin, end in sorted(slices):
        if begin != position:
            problem = "overlap" if begin < position else "gap"
            raise ValueError(f"{name}: export left a {problem} at byte {min(begin, position)} of {size}")
        position = end
    if position != size:
        raise ValueError(f"{name}: export wrote {position} of {size} bytes")
