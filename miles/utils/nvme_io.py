"""Bounded pinned-memory staging between local direct I/O and CUDA tensors."""

from __future__ import annotations

import os
import threading
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from pathlib import Path

import torch

NVME_ALIGNMENT = 4096
_STAGING_BYTES = 8 << 20


def _cuda_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("NVMe staging requires an explicitly selected CUDA device")
    return torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)


def _aligned_view(storage: torch.Tensor, nbytes: int) -> torch.Tensor:
    return storage.narrow(0, (-storage.data_ptr()) % NVME_ALIGNMENT, nbytes)


def allocate_aligned_buffer(nbytes: int, device: torch.device | str) -> torch.Tensor:
    """Allocate an owned GPU byte buffer matching the aligned disk layout."""
    if nbytes <= 0 or nbytes % NVME_ALIGNMENT:
        raise ValueError("NVMe buffer size must be a positive multiple of 4096 bytes")
    storage = torch.empty(nbytes + NVME_ALIGNMENT - 1, dtype=torch.uint8, device=_cuda_device(device))
    return _aligned_view(storage, nbytes)


class _StagingSlot:
    def __init__(self, nbytes: int, device: torch.device):
        with torch.cuda.device(device):
            storage = torch.empty(nbytes + NVME_ALIGNMENT - 1, dtype=torch.uint8, device="cpu", pin_memory=True)
            self.buffer = _aligned_view(storage, nbytes)
            self.stream = torch.cuda.Stream(device=device)
        # ndarray and memoryview both share the pinned tensor's storage.
        self.bytes = memoryview(self.buffer.numpy())
        self.lock = threading.Lock()


class NvmeStagingPool:
    """One reusable pinned read chunk and one write chunk for an entire owner.

    Share this object across baseline and initial checkpoint file handles. The
    default uses 16 MiB plus alignment padding, independent of model/expert size.
    Separate slots and copy streams allow reads and writes to progress together.
    """

    def __init__(self, device: torch.device | str, *, buffer_bytes: int = _STAGING_BYTES):
        if buffer_bytes <= 0 or buffer_bytes % NVME_ALIGNMENT:
            raise ValueError("Pinned staging capacity must be a positive multiple of 4096 bytes")
        self.device = _cuda_device(device)
        self.buffer_bytes = buffer_bytes
        self._read = _StagingSlot(buffer_bytes, self.device)
        self._write = _StagingSlot(buffer_bytes, self.device)


def _read_exact(fd: int, buffer: memoryview, offset: int, expected: int) -> None:
    count = os.preadv(fd, [buffer], offset)
    if count != expected:
        raise OSError(f"Direct NVMe read at {offset}: expected {expected} bytes, received {count}")


def _write_exact(fd: int, buffer: memoryview, offset: int) -> None:
    count = os.pwritev(fd, [buffer], offset)
    if count != len(buffer):
        raise OSError(f"Direct NVMe write at {offset}: expected {len(buffer)} bytes, received {count}")


class NvmeBackend:
    """One direct-I/O file handle; Futures own every disk and CUDA transfer.

    Reads complete only after disk -> pinned CPU -> GPU, and writes complete
    only after GPU -> pinned CPU -> disk. CUDA event waits happen on I/O workers,
    never as a device-wide synchronization. The owner must not reuse GPU buffers
    until their Futures and other GPU consumers finish.

    Linux O_DIRECT is mandatory to avoid retaining full baselines in page cache.
    This backend requires ordinary filesystem direct I/O, not a GDS driver, and
    never silently falls back to buffered I/O. It does not fsync ephemeral data.
    """

    def __init__(
        self,
        path: str | Path,
        device: torch.device | str,
        writable: bool = False,
        *,
        staging: NvmeStagingPool | None = None,
        executor: Executor | None = None,
        max_inflight: int = 4,
    ):
        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        if not all(hasattr(os, name) for name in ("O_DIRECT", "preadv", "pwritev")):
            raise RuntimeError("NVMe staging requires Linux O_DIRECT, preadv, and pwritev")
        self.device = _cuda_device(device)
        self.path = Path(path)
        self.writable = writable
        self._staging = staging if staging is not None else NvmeStagingPool(self.device)
        if self._staging.device != self.device:
            raise ValueError("Pinned staging pool and NVMe backend must use the same CUDA device")
        flags = os.O_RDWR | os.O_CREAT if writable else os.O_RDONLY
        self._fd = os.open(self.path, flags | os.O_DIRECT, 0o600)
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="delta-nvme")
        self._slots = threading.BoundedSemaphore(max_inflight)
        self._lock = threading.Lock()
        self._futures: list[Future[int]] = []
        self._closed = False

    def read_into(self, offset: int, dst_u8: torch.Tensor, *, expected_bytes: int | None = None) -> Future[int]:
        """Read aligned bytes; result() guarantees that the GPU destination is ready.

        An exact known EOF short read is allowed for canonical checkpoint tails.
        Bytes beyond expected_bytes are left untouched and must not be consumed.
        """
        self._validate_buffer(offset, dst_u8)
        expected = dst_u8.numel() if expected_bytes is None else expected_bytes
        if not 0 < expected <= dst_u8.numel():
            raise ValueError("Expected NVMe read length must be positive and fit the destination")
        if expected != dst_u8.numel() and offset + expected != os.fstat(self._fd).st_size:
            raise ValueError("A known short NVMe read is permitted only for the exact file tail")
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        return self._submit(offset, dst_u8, ready, write=False, expected=expected)

    def write_from(self, offset: int, src_u8: torch.Tensor, ready_event: torch.cuda.Event) -> Future[int]:
        """Write after the producer event; result() guarantees disk I/O completion."""
        if not self.writable:
            raise ValueError(f"NVMe file is read-only: {self.path}")
        self._validate_buffer(offset, src_u8)
        if ready_event.device != self.device:
            raise ValueError("NVMe write requires a recorded readiness event on the buffer's device")
        return self._submit(offset, src_u8, ready_event, write=True, expected=src_u8.numel())

    def _validate_buffer(self, offset: int, tensor: torch.Tensor) -> None:
        if tensor.device != self.device or tensor.dtype != torch.uint8 or tensor.ndim != 1:
            raise ValueError("NVMe buffers must be flat uint8 tensors on the configured CUDA device")
        if not tensor.is_contiguous():
            raise ValueError("NVMe buffers must be contiguous; implicit copies are not permitted")
        if offset < 0 or offset % NVME_ALIGNMENT:
            raise ValueError("NVMe file offsets must be 4096-byte aligned")
        if tensor.numel() == 0 or tensor.numel() % NVME_ALIGNMENT:
            raise ValueError("NVMe I/O lengths must be positive multiples of 4096 bytes")

    def _submit(self, offset, tensor, ready_event, *, write: bool, expected: int) -> Future[int]:
        self._slots.acquire()
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Cannot submit I/O to a closed NVMe backend")
                future = self._executor.submit(
                    self._transfer, offset, tensor, ready_event, write=write, expected=expected
                )
                self._futures.append(future)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        return future

    def _transfer(self, offset, tensor, ready_event, *, write: bool, expected: int) -> int:
        slot = self._staging._write if write else self._staging._read
        with slot.lock, torch.cuda.device(self.device), torch.cuda.stream(slot.stream):
            slot.stream.wait_event(ready_event)
            tensor.record_stream(slot.stream)
            copied = torch.cuda.Event()
            copy_pending = False
            try:
                for begin in range(0, expected, self._staging.buffer_bytes):
                    requested = min(self._staging.buffer_bytes, tensor.numel() - begin)
                    count = min(requested, expected - begin)
                    if write:
                        copy_pending = True
                        slot.buffer[:count].copy_(tensor[begin : begin + count], non_blocking=True)
                        copied.record(slot.stream)
                        copied.synchronize()
                        copy_pending = False
                        _write_exact(self._fd, slot.bytes[:count], offset + begin)
                    else:
                        _read_exact(self._fd, slot.bytes[:requested], offset + begin, count)
                        copy_pending = True
                        tensor[begin : begin + count].copy_(slot.buffer[:count], non_blocking=True)
                        copied.record(slot.stream)
                        copied.synchronize()
                        copy_pending = False
            finally:
                # Keep both the GPU tensor and pinned slot alive even if a later
                # disk/copy operation fails after earlier copies were enqueued.
                if copy_pending:
                    copied.record(slot.stream)
                    copied.synchronize()
        return expected

    def drain(self) -> None:
        """Wait every already submitted operation, then report the first failure."""
        with self._lock:
            futures, self._futures = self._futures, []
        error = None
        for future in futures:
            try:
                future.result()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def close(self) -> None:
        """Drain disk and CUDA transfers before closing the file handle."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.drain()
        finally:
            try:
                if self._owns_executor:
                    self._executor.shutdown(wait=True)
            finally:
                os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
