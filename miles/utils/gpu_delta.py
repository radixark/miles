"""Lossless GPU XOR/Zstd deltas with CPU-compatible Adler32 checksums.

Compression stays on GPU. Only per-tensor counts/checksums and compressed bytes
cross to CPU. Input lengths stay fixed until compression, avoiding CUDA nonzero
or boolean-indexing operations that synchronize to discover sparse output sizes.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import threading
from dataclasses import dataclass

import numpy as np
import torch

_ADLER_MODULUS = 65521
_ADLER_BLOCK_BYTES = 4096
_ADLER_WINDOW_BYTES = 1 << 20
_ZSTD_MAX_INPUT_BYTES = (1 << 31) - 1


@dataclass(frozen=True)
class EncodedTensor:
    payload: np.ndarray
    checksum: str
    changed: int
    total: int


class _ZstdOptions(ctypes.Structure):
    # nvCOMP 5.x's public nvcompBatchedZstdCompressOpts_t ABI.
    _fields_ = [("reserved", ctypes.c_char * 64)]


class _NvcompZstd:
    """Public batched C API with caller-owned storage and no global allocator changes."""

    def __init__(self):
        cuda_major = torch.version.cuda.split(".")[0]
        package = f"nvidia-libnvcomp-cu{cuda_major}"
        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"GPU deltas require {package}>=5.3,<6") from error
        version = tuple(int(part) for part in distribution.version.split(".")[:2])
        if not (5, 3) <= version < (6, 0) or ctypes.sizeof(ctypes.c_size_t) != 8:
            raise RuntimeError("GPU deltas require the 64-bit nvCOMP 5.3+ C API")
        self._library = ctypes.CDLL(str(distribution.locate_file("nvidia/libnvcomp/lib64/libnvcomp.so.5")))
        size, pointer, options = ctypes.c_size_t, ctypes.c_void_p, _ZstdOptions
        self._bound = self._bind("GetMaxOutputChunkSize", [size, options, ctypes.POINTER(size)])
        self._temporary = self._bind("GetTempSizeAsync", [size, size, options, ctypes.POINTER(size), size])
        self._compress = self._bind(
            "Async", [pointer, pointer, size, size, pointer, size, pointer, pointer, options, pointer, pointer]
        )
        self._options = options()

    def _bind(self, suffix, arguments):
        function = getattr(self._library, "nvcompBatchedZstdCompress" + suffix)
        function.argtypes, function.restype = arguments, ctypes.c_int
        return function

    @staticmethod
    def _check(status):
        if status != 0:
            raise RuntimeError(f"nvCOMP Zstd compression failed with status {status}")

    def _output_bytes(self, length):
        bound = ctypes.c_size_t()
        self._check(self._bound(length, self._options, ctypes.byref(bound)))
        return bound.value

    def compress(self, tensors, stream):
        lengths = [tensor.numel() for tensor in tensors]
        count = len(tensors)
        sizes = torch.empty(count, dtype=torch.int64, device=stream.device)
        statuses = torch.empty(count, dtype=torch.int32, device=stream.device)
        if not count:
            return [], sizes, statuses, ()
        temporary_bytes = ctypes.c_size_t()
        self._check(self._temporary(count, max(lengths), self._options, ctypes.byref(temporary_bytes), sum(lengths)))
        outputs = [torch.empty(self._output_bytes(n), dtype=torch.uint8, device=stream.device) for n in lengths]
        temporary = torch.empty(temporary_bytes.value, dtype=torch.uint8, device=stream.device)
        # Only pointers and fixed lengths go H2D. All data buffers come from
        # fresh Torch allocations, satisfying nvCOMP's 4-byte alignment contract.
        parameters = torch.empty((3, count), dtype=torch.int64, device="cpu", pin_memory=True)
        parameters.numpy()[:] = [[t.data_ptr() for t in tensors], lengths, [t.data_ptr() for t in outputs]]
        device_parameters = parameters.to(stream.device, non_blocking=True)
        self._check(
            self._compress(
                device_parameters[0].data_ptr(),
                device_parameters[1].data_ptr(),
                max(lengths),
                count,
                temporary.data_ptr(),
                temporary_bytes.value,
                device_parameters[2].data_ptr(),
                sizes.data_ptr(),
                self._options,
                statuses.data_ptr(),
                stream.cuda_stream,
            )
        )
        return outputs, sizes, statuses, (parameters, device_parameters, temporary)


def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("GPU delta inputs must be contiguous CUDA tensors; implicit copies are not permitted")
    return tensor.detach().reshape(-1).view(torch.uint8)


def gpu_adler32(data: torch.Tensor) -> torch.Tensor:
    """Return zlib.adler32(data)'s unsigned value in a CUDA int64 scalar.

    Weighted sums are reduced modulo 65521 in 4096-byte blocks. Windows cap the
    int64 temporary at 8 MiB; combining windows on device avoids host scalar
    reads and overflow even for buffers much larger than one expert tensor.
    """
    data = _byte_view(data)
    first = torch.ones((), dtype=torch.int64, device=data.device)
    second = torch.zeros((), dtype=torch.int64, device=data.device)
    weights = torch.arange(_ADLER_BLOCK_BYTES, 0, -1, dtype=torch.int64, device=data.device)
    for start in range(0, data.numel(), _ADLER_WINDOW_BYTES):
        window = data[start : start + _ADLER_WINDOW_BYTES]
        length = window.numel()
        padded_length = (length + _ADLER_BLOCK_BYTES - 1) // _ADLER_BLOCK_BYTES * _ADLER_BLOCK_BYTES
        integers = torch.zeros(padded_length, dtype=torch.int64, device=data.device)
        integers[:length].copy_(window)
        blocks = integers.view(-1, _ADLER_BLOCK_BYTES)
        sums = blocks.sum(dim=1).remainder_(_ADLER_MODULUS)
        weighted = (blocks * weights).sum(dim=1).remainder_(_ADLER_MODULUS)
        remaining = length - torch.arange(
            _ADLER_BLOCK_BYTES, padded_length + 1, _ADLER_BLOCK_BYTES, dtype=torch.int64, device=data.device
        )
        contribution = (weighted + sums * remaining.remainder(_ADLER_MODULUS)).remainder_(_ADLER_MODULUS).sum()
        second = (second + length * first + contribution).remainder_(_ADLER_MODULUS)
        first = (first + sums.sum()).remainder_(_ADLER_MODULUS)
    return torch.bitwise_or(torch.bitwise_left_shift(second, 16), first)


class PendingBatch:
    """Own all buffers until compression and exact-size D2H transfers complete.

    finish() is the publication boundary: first resolve scalar metadata, then
    transfer only compressed payloads on a separate copy stream. The public C
    API writes sizes/statuses into our metadata buffer; opaque nvCOMP Python
    arrays can synchronize later work when their lazy metadata is destroyed.
    """

    def __init__(self, *, encoded, metadata, totals, stream, ready, keepalive):
        self._encoded = encoded
        self._metadata = metadata
        self._totals = totals
        self._stream = stream
        self._ready = ready
        self._keepalive = keepalive
        self._result: list[EncodedTensor] | None = None
        self._closed = False
        self._copy_pending = False
        self._lock = threading.Lock()

    def finish(self) -> list[EncodedTensor]:
        with self._lock:
            return self._finish()

    def _finish(self) -> list[EncodedTensor]:
        if self._result is not None:
            return self._result
        if self._closed:
            raise RuntimeError("Cannot finish an abandoned GPU delta batch")
        self._ready.synchronize()
        metadata = self._metadata.numpy()
        copies = self._copy_payloads(metadata)
        self._result = [
            EncodedTensor(payload=host.numpy(), checksum=f"{int(digest):08x}", changed=int(changed), total=total)
            for host, (changed, digest, _, _), total in zip(copies, metadata, self._totals, strict=True)
        ]
        self._keepalive = ()
        self._encoded = ()
        self._closed = True
        return self._result

    def _copy_payloads(self, metadata):
        copies = []
        copied = torch.cuda.Event()
        with torch.cuda.device(self._stream.device), torch.cuda.stream(self._stream):
            try:
                for device, (changed, _, size, status) in zip(self._encoded, metadata, strict=True):
                    if status:
                        raise RuntimeError(f"nvCOMP Zstd chunk failed with status {status}")
                    size = int(size)
                    if device.device != self._stream.device or not 0 < size <= device.numel():
                        raise RuntimeError("nvCOMP returned an invalid compressed device buffer")
                    if not changed:
                        copies.append(torch.empty(0, dtype=torch.uint8, device="cpu"))
                        continue
                    # RAW Zstd has no nvCOMP container header, so existing CPU
                    # receivers can decompress exactly one canonical tensor.
                    host = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
                    device.record_stream(self._stream)
                    self._copy_pending = True
                    host.copy_(device[:size], non_blocking=True)
                    copies.append(host)
            finally:
                # Later status/allocation failures still drain earlier copies.
                # If this wait fails, close() retries before releasing sources.
                if self._copy_pending:
                    copied.record(self._stream)
                    copied.synchronize()
                    self._copy_pending = False
        return copies

    def close(self) -> None:
        """Abandon payload publication only after all GPU readers have completed."""
        with self._lock:
            if not self._closed:
                self._ready.synchronize()
                if self._copy_pending:
                    self._stream.synchronize()
                    self._copy_pending = False
                self._keepalive = ()
                self._encoded = ()
                self._closed = True

    def __del__(self):
        # The explicit finish/close paths propagate failures. This last-resort
        # finalizer drains outstanding work if its owner is discarded.
        try:
            self.close()
        except Exception:
            pass


class GpuDeltaCodec:
    """One caller thread's GPU Zstd compressor; no CPU compression fallback."""

    def __init__(self):
        self._compressor = _NvcompZstd()
        self._thread = threading.get_ident()
        self._copy_streams = {}

    def encode_batch(
        self,
        previous: list[torch.Tensor],
        current: list[torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> PendingBatch:
        if threading.get_ident() != self._thread:
            raise RuntimeError("Each caller thread must own its own GpuDeltaCodec")
        if len(previous) != len(current):
            raise ValueError("Previous and current delta batches must have equal lengths")
        old = [_byte_view(tensor) for tensor in previous]
        new = [_byte_view(tensor) for tensor in current]
        for before, after in zip(old, new, strict=True):
            if before.device != stream.device or after.device != stream.device or before.numel() != after.numel():
                raise ValueError("Delta byte lengths and devices must match the selected CUDA stream")
            if not 0 < after.numel() <= _ZSTD_MAX_INPUT_BYTES:
                raise ValueError("Each RAW Zstd tensor must contain between 1 and 2 GiB - 1 bytes")
        with torch.cuda.device(stream.device):
            if stream.device not in self._copy_streams:
                self._copy_streams[stream.device] = torch.cuda.Stream(device=stream.device)
            producer = torch.cuda.current_stream(stream.device)
            if stream != producer:
                stream.wait_stream(producer)
            with torch.cuda.stream(stream):
                return self._encode(old, new, stream)

    def _encode(self, old, new, stream):
        # Record input consumers before the first kernel: a later allocation or
        # checksum failure must not let their producer-stream storage be reused.
        for tensor in old + new:
            tensor.record_stream(stream)
        differences = [torch.bitwise_xor(before, after) for before, after in zip(old, new, strict=True)]
        metrics = [
            torch.stack((torch.count_nonzero(difference), gpu_adler32(after)))
            for difference, after in zip(differences, new, strict=True)
        ]
        # Compression writes exact sizes/statuses on device, then piggybacks them
        # onto the same small metadata transfer as counts and checksums.
        try:
            encoded, sizes, statuses, compression_storage = self._compressor.compress(differences, stream)
            device_metadata = (
                torch.cat((torch.stack(metrics), sizes[:, None], statuses[:, None]), dim=1)
                if metrics
                else torch.empty((0, 4), dtype=torch.int64, device=stream.device)
            )
            host_metadata = torch.empty(device_metadata.shape, dtype=torch.int64, device="cpu", pin_memory=True)
            host_metadata.copy_(device_metadata, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(stream)
            return PendingBatch(
                encoded=encoded,
                metadata=host_metadata,
                totals=[tensor.numel() for tensor in new],
                stream=self._copy_streams[stream.device],
                ready=ready,
                keepalive=(self, old, new, differences, compression_storage, sizes, statuses, device_metadata),
            )
        except BaseException:
            # Drain after a partial submission while caller-owned storage remains
            # alive. Only the failure path waits on the compression stream.
            stream.synchronize()
            raise
