"""Asynchronous GPU compression using nvCOMP's prebuilt batched C API.

Compression runs on CUDA SMs. Blackwell's fixed-function engine can decompress
the inner Snappy or LZ4 frames at the receiver.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
from dataclasses import dataclass

import torch


class _Options(ctypes.Structure):
    # nvCOMP 5.3 Snappy and Zstd compression options have the same public ABI.
    _fields_ = [("reserved", ctypes.c_char * 64)]


class _LZ4Options(ctypes.Structure):
    # Zero initialization selects NVCOMP_TYPE_CHAR and NVCOMP_BITSHUFFLE_NONE.
    _fields_ = [("data_type", ctypes.c_int), ("bitshuffle_mode", ctypes.c_int), ("reserved", ctypes.c_char * 56)]


class _Alignments(ctypes.Structure):
    _fields_ = [("input", ctypes.c_size_t), ("output", ctypes.c_size_t), ("temp", ctypes.c_size_t)]


@dataclass(frozen=True)
class CompressionBatch:
    """Retain until sizes/statuses and payload transfers finish on the caller stream."""

    outputs: list[torch.Tensor]
    sizes: torch.Tensor
    statuses: torch.Tensor
    keepalive: tuple


class NvcompCompressor:
    """One device/codec, with caller-owned memory and no host synchronization.

    Inputs are already framed contiguous uint8 CUDA views. The caller performs
    XOR and checks returned per-frame statuses after its own completion event.
    Only pointer/length metadata is copied here; no weights or hashes cross CPU.
    """

    def __init__(self, codec: str, device: torch.device):
        if codec not in ("zstd", "snappy", "lz4"):
            raise ValueError("GPU delta compression requires zstd, snappy or lz4")
        self.codec, self.device = codec, torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("GPU delta compressor requires an explicit CUDA device")
        package = f"nvidia-libnvcomp-cu{torch.version.cuda.split('.')[0]}"
        distribution = importlib.metadata.distribution(package)
        version = tuple(int(v) for v in distribution.version.split(".")[:2])
        if not (5, 3) <= version < (6, 0) or ctypes.sizeof(ctypes.c_size_t) != 8:
            raise RuntimeError("GPU deltas require the 64-bit nvCOMP 5.3+ ABI")
        self.version = distribution.version
        self._library = ctypes.CDLL(str(distribution.locate_file("nvidia/libnvcomp/lib64/libnvcomp.so.5")))
        self._symbol, options = {
            "snappy": ("Snappy", _Options),
            "lz4": ("LZ4", _LZ4Options),
            "zstd": ("Zstd", _Options),
        }[codec]
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        self._options = options()
        self._bound = self._bind("GetMaxOutputChunkSize", [size, options, ctypes.POINTER(size)])
        self._temporary = self._bind("GetTempSizeAsync", [size, size, options, ctypes.POINTER(size), size])
        align = self._bind("GetRequiredAlignments", [options, ctypes.POINTER(_Alignments)])
        self._compress = self._bind(
            "Async", [pointer, pointer, size, size, pointer, size, pointer, pointer, options, pointer, pointer]
        )
        self._alignments = _Alignments()
        self._check(align(self._options, ctypes.byref(self._alignments)))
        self._size_cache, self._bound_cache = {}, {}

    def _bind(self, suffix, arguments):
        function = getattr(self._library, "nvcompBatched" + self._symbol + "Compress" + suffix)
        function.argtypes, function.restype = arguments, ctypes.c_int
        return function

    def _check(self, status):
        if status:
            raise RuntimeError(f"nvCOMP {self.codec} compression failed: status={status}")

    def _allocation_sizes(self, count, maximum, total):
        key = count, maximum, total
        if key in self._size_cache:
            return self._size_cache[key]
        bound, temporary = ctypes.c_size_t(), ctypes.c_size_t()
        self._check(self._bound(maximum, self._options, ctypes.byref(bound)))
        self._check(self._temporary(count, maximum, self._options, ctypes.byref(temporary), total))
        alignment = self._alignments.output
        stride = (bound.value + alignment - 1) // alignment * alignment
        result = stride, temporary.value
        self._size_cache[key] = result
        return result

    def compress(
        self, frames: list[torch.Tensor], stream: torch.cuda.Stream, compact_outputs=False
    ) -> CompressionBatch:
        if stream.device != self.device:
            raise ValueError("Compression stream/device mismatch")
        for frame in frames:
            if (
                frame.device != self.device
                or frame.dtype != torch.uint8
                or not frame.is_contiguous()
                or not 0 < frame.numel() <= 1 << 24
                or frame.data_ptr() % self._alignments.input
            ):
                raise ValueError("nvCOMP frames must be aligned contiguous CUDA uint8, with 1..16 MiB bytes")
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            return self._enqueue(frames, stream, compact_outputs=compact_outputs)

    def _compact_outputs(self, lengths):
        offsets, total = [], 0
        for length in lengths:
            if length not in self._bound_cache:
                bound = ctypes.c_size_t()
                self._check(self._bound(length, self._options, ctypes.byref(bound)))
                self._bound_cache[length] = bound.value
            size = self._bound_cache[length]
            total = (total + self._alignments.output - 1) // self._alignments.output * self._alignments.output
            offsets.append((total, size))
            total += size
        arena = torch.empty(total, dtype=torch.uint8, device=self.device)
        return arena, [arena[offset : offset + size] for offset, size in offsets]

    def _enqueue(self, frames, stream, compact_outputs=False):
        count = len(frames)
        sizes = torch.empty(count, dtype=torch.int64, device=self.device)
        statuses = torch.empty(count, dtype=torch.int32, device=self.device)
        if not count:
            return CompressionBatch([], sizes, statuses, ())
        lengths = [frame.numel() for frame in frames]
        stride, temporary_bytes = self._allocation_sizes(count, max(lengths), sum(lengths))
        # One output allocation avoids per-frame allocator traffic. Rounded
        # strides preserve the independently queried nvCOMP output alignment.
        if compact_outputs:
            # Outer chunks vary with tensor compression ratio. Reserving the
            # maximum bound for every tiny tensor can waste model-sized HBM.
            output, outputs = self._compact_outputs(lengths)
        else:
            output = torch.empty((count, stride), dtype=torch.uint8, device=self.device)
            outputs = list(output.unbind())
        temporary = torch.empty(temporary_bytes, dtype=torch.uint8, device=self.device)
        if output.data_ptr() % self._alignments.output or temporary.data_ptr() % self._alignments.temp:
            raise RuntimeError("CUDA allocator does not satisfy nvCOMP compression alignment")
        parameters = torch.empty((3, count), dtype=torch.int64, device="cpu", pin_memory=True)
        parameters.numpy()[:] = [[frame.data_ptr() for frame in frames], lengths, [out.data_ptr() for out in outputs]]
        device_parameters = parameters.to(self.device, non_blocking=True)
        try:
            self._check(
                self._compress(
                    device_parameters[0].data_ptr(),
                    device_parameters[1].data_ptr(),
                    max(lengths),
                    count,
                    temporary.data_ptr(),
                    temporary_bytes,
                    device_parameters[2].data_ptr(),
                    sizes.data_ptr(),
                    self._options,
                    statuses.data_ptr(),
                    stream.cuda_stream,
                )
            )
        except Exception:
            # An immediate API failure may follow metadata copies or a partial
            # launch. Drain this stream before releasing caller-owned buffers.
            stream.synchronize()
            raise
        return CompressionBatch(outputs, sizes, statuses, (frames, parameters, device_parameters, temporary, output))
