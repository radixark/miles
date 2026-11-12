"""Shared nvCOMP 6 ABI and asynchronous CUDA compression.

Compression runs on CUDA SMs. The receiver uses SMs for Bitcomp and other
non-DE decoding, or Blackwell's fixed-function engine for Snappy/LZ4.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
from dataclasses import dataclass

import torch


class _Options(ctypes.Structure):
    # nvCOMP 6 Snappy and Zstd compression options have the same public ABI.
    _fields_ = [("reserved", ctypes.c_char * 64)]


class _LZ4Options(ctypes.Structure):
    # Zero initialization selects NVCOMP_TYPE_CHAR and NVCOMP_BITSHUFFLE_NONE.
    _fields_ = [("data_type", ctypes.c_int), ("bitshuffle_mode", ctypes.c_int), ("reserved", ctypes.c_char * 56)]


class _BitcompOptions(ctypes.Structure):
    _fields_ = [
        ("algorithm", ctypes.c_int),
        ("data_type", ctypes.c_int),
        ("delta", ctypes.c_double),
        ("mode", ctypes.c_int),
        ("reserved", ctypes.c_char * 44),
    ]


class _Alignments(ctypes.Structure):
    _fields_ = [("input", ctypes.c_size_t), ("output", ctypes.c_size_t), ("temp", ctypes.c_size_t)]


def _load_library():
    package = f"nvidia-libnvcomp-cu{torch.version.cuda.split('.')[0]}"
    distribution = importlib.metadata.distribution(package)
    if int(distribution.version.split(".")[0]) != 6 or ctypes.sizeof(ctypes.c_size_t) != 8:
        raise RuntimeError("GPU deltas require the 64-bit nvCOMP 6 ABI")
    library = ctypes.CDLL(str(distribution.locate_file("nvidia/libnvcomp/lib64/libnvcomp.so.6")))
    return library, distribution.version


class _NvcompApi:
    """One codec and direction of the nvCOMP 6 batched C API.

    Subclasses load the library and set ``_prefix``, ``_label`` and ``_options``
    before ``_bind_queries`` binds the shared workspace and alignment queries.
    """

    def _bind(self, suffix, arguments):
        function = getattr(
            self._library, self._prefix + suffix
        )  # config-access-exempt: nvCOMP C symbol is selected by codec and operation
        function.argtypes, function.restype = arguments, ctypes.c_int
        return function

    def _check(self, status):
        if status:
            raise RuntimeError(f"nvCOMP {self._label} failed: status={status}")

    def _bind_queries(self):
        options, pointer, size = type(self._options), ctypes.c_void_p, ctypes.c_size_t
        self._temporary = self._bind("GetTempSize", [size, size, options, ctypes.POINTER(size), size, pointer])
        align = self._bind("GetRequiredAlignments", [options, ctypes.POINTER(_Alignments)])
        self.alignments = _Alignments()
        self._check(align(self._options, ctypes.byref(self.alignments)))
        self._temporary_sizes = {}

    def temporary_bytes(self, geometry: tuple[int, int, int], cuda_stream: int) -> int:
        """Workspace bytes for (frame count, maximum frame, total bytes) on one stream."""
        count, maximum, total = geometry
        if not count:
            return 0
        key = count, maximum, total, cuda_stream
        if key not in self._temporary_sizes:
            size = ctypes.c_size_t()
            # The query enqueues no work. Cache its result for the batch geometry
            # and stream on this instance's fixed device/options.
            self._check(self._temporary(count, maximum, self._options, ctypes.byref(size), total, cuda_stream))
            self._temporary_sizes[key] = size.value
        return self._temporary_sizes[key]


@dataclass(frozen=True)
class CompressionBatch:
    """Retain until sizes/statuses and payload transfers finish on the caller stream."""

    outputs: list[torch.Tensor]
    sizes: torch.Tensor
    statuses: torch.Tensor
    keepalive: tuple


class NvcompCompressor(_NvcompApi):
    """One device/codec, with caller-owned memory and no host synchronization.

    Inputs are already framed contiguous uint8 CUDA views. The caller performs
    XOR and checks returned per-frame statuses after its own completion event.
    Only pointer/length metadata is copied here; no weights or hashes cross CPU.
    """

    def __init__(self, codec: str, device: torch.device):
        if codec not in ("zstd", "snappy", "lz4", "bitcomp"):
            raise ValueError("GPU delta compression requires zstd, snappy, lz4 or bitcomp")
        self.codec, self.device = codec, torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("GPU delta compressor requires an explicit CUDA device")
        self._library, self.version = _load_library()
        symbol, options = {
            "snappy": ("Snappy", _Options),
            "lz4": ("LZ4", _LZ4Options),
            "zstd": ("Zstd", _Options),
            "bitcomp": ("Bitcomp", _BitcompOptions),
        }[codec]
        self._prefix, self._label = f"nvcompBatched{symbol}Compress", f"{codec} compression"
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        self._options = options()
        if codec == "bitcomp":
            # Sparse, U8, lossless. XOR payloads are bytes, never floating-point values.
            self._options.algorithm = 1
            self._options.data_type = 1  # NVCOMP_TYPE_UCHAR
            self._options.mode = 0  # BITCOMP_LOSSLESS; delta remains zero.
        elif codec in ("snappy", "lz4", "zstd"):
            # Keep zeroed options: Snappy/Zstd fields are all reserved; LZ4 zeros select CHAR without bitshuffle.
            pass
        else:
            raise ValueError(f"unsupported GPU delta compression codec: {codec}")
        self._bound = self._bind("GetMaxOutputChunkSize", [size, options, ctypes.POINTER(size)])
        self._compress = self._bind(
            "Async", [pointer, pointer, size, size, pointer, size, pointer, pointer, options, pointer, pointer]
        )
        self._bind_queries()
        self._bound_cache = {}

    def _output_bound(self, length):
        if length not in self._bound_cache:
            bound = ctypes.c_size_t()
            self._check(self._bound(length, self._options, ctypes.byref(bound)))
            self._bound_cache[length] = bound.value
        return self._bound_cache[length]

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
                or frame.data_ptr() % self.alignments.input
            ):
                raise ValueError("nvCOMP frames must be aligned contiguous CUDA uint8, with 1..16 MiB bytes")
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            return self._enqueue(frames, stream, compact_outputs=compact_outputs)

    def _compact_outputs(self, lengths):
        offsets, total = [], 0
        for length in lengths:
            size = self._output_bound(length)
            total = (total + self.alignments.output - 1) // self.alignments.output * self.alignments.output
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
        maximum = max(lengths)
        temporary_bytes = self.temporary_bytes((count, maximum, sum(lengths)), stream.cuda_stream)
        # One output allocation avoids per-frame allocator traffic. Rounded
        # strides preserve the independently queried nvCOMP output alignment.
        if compact_outputs:
            # Outer chunks vary with tensor compression ratio. Reserving the
            # maximum bound for every tiny tensor can waste model-sized HBM.
            output, outputs = self._compact_outputs(lengths)
        else:
            alignment = self.alignments.output
            stride = (self._output_bound(maximum) + alignment - 1) // alignment * alignment
            output = torch.empty((count, stride), dtype=torch.uint8, device=self.device)
            outputs = list(output.unbind())
        temporary = torch.empty(temporary_bytes, dtype=torch.uint8, device=self.device)
        if output.data_ptr() % self.alignments.output or temporary.data_ptr() % self.alignments.temp:
            raise RuntimeError("CUDA allocator does not satisfy nvCOMP compression alignment")
        parameters = torch.empty((3, count), dtype=torch.int64, device="cpu", pin_memory=True)
        parameters.numpy()[:] = [[frame.data_ptr() for frame in frames], lengths, [out.data_ptr() for out in outputs]]
        device_parameters = parameters.to(self.device, non_blocking=True)
        try:
            self._check(
                self._compress(
                    device_parameters[0].data_ptr(),
                    device_parameters[1].data_ptr(),
                    maximum,
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
