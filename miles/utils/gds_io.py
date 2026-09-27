"""Bounded GPU-only cuFile I/O for versioned, aligned delta baselines.

The caller must qualify the actual mount's direct read AND write paths before
enabling this backend. A successful import or file registration is insufficient.
Compatibility mode must be disabled in cuFile's configuration before startup;
this module never substitutes POSIX reads or host staging for unavailable GDS.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

GDS_ALIGNMENT = 4096
_DRIVER_LOCK = threading.Lock()
_DRIVER_OPENED = False


def allocate_aligned_buffer(nbytes: int, device: torch.device | str) -> torch.Tensor:
    """Allocate an independently owned GPU byte buffer suitable for direct I/O."""
    if nbytes <= 0 or nbytes % GDS_ALIGNMENT:
        raise ValueError("GDS buffer size must be a positive multiple of 4096 bytes")
    device = _cuda_device(device)
    storage = torch.empty(nbytes + GDS_ALIGNMENT - 1, dtype=torch.uint8, device=device)
    offset = (-storage.data_ptr()) % GDS_ALIGNMENT
    return storage.narrow(0, offset, nbytes)


def _cuda_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("GDS requires an explicitly selected CUDA device")
    return torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)


def _require_direct_configuration(cufile) -> None:
    """Fail closed instead of silently accepting cuFile's host fallback mode."""
    global _DRIVER_OPENED
    # Queries before initialization can report staged defaults instead of the
    # cufile.json configuration. The driver is process-scoped, not file-scoped;
    # closing a file must not tear down another user's active cuFile handles.
    with _DRIVER_LOCK:
        if not _DRIVER_OPENED:
            cufile.driver_open()
            _DRIVER_OPENED = True
    required = ("PROPERTIES_ALLOW_COMPAT_MODE", "FORCE_COMPAT_MODE")
    optional = ("PROPERTIES_POSIX_IO_MODE", "GDS_FALLBACK_IO")
    parameters = cufile.BoolConfigParameter
    for name in required + optional:
        parameter = getattr(parameters, name, None)
        if parameter is None:
            if name in required:
                raise RuntimeError(f"cuda.bindings.cufile cannot verify {name}; direct GDS is unqualified")
            continue
        try:
            enabled = cufile.get_parameter_bool(parameter)
        except cufile.cuFileError as error:
            # Bindings can expose flags introduced after the loaded libcufile.
            # Only optional unknown parameters may be ignored; required checks
            # and all other runtime failures remain fatal.
            if name in optional and error.status == cufile.OpError.INVALID_VALUE:
                continue
            raise
        if enabled:
            raise RuntimeError(f"Direct GDS requires cuFile {name}=false; configure it before process startup")


class GdsBackend:
    """One file handle with bounded asynchronous reads/writes to CUDA tensors.

    Official CUDA Python bindings release the GIL around synchronous cuFile I/O,
    allowing these workers to overlap Python-submitted quantization kernels.
    PyTorch's synchronous GdsFile bindings do not provide that guarantee.

    The caller owns GPU stream ordering: schedule prefetch before quantization,
    and do not reuse an I/O buffer until both its Future and its GPU consumers
    complete. Each submitted operation retains its tensor and readiness event.
    """

    def __init__(
        self,
        path: str | Path,
        device: torch.device | str,
        writable: bool = False,
        *,
        executor: Executor | None = None,
        max_inflight: int = 4,
    ):
        # GDS is optional; importing Miles must not require CUDA Python/cuFile.
        from cuda.bindings import cufile

        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        self.device = _cuda_device(device)
        self.path = Path(path)
        self.writable = writable
        self._cufile = cufile
        _require_direct_configuration(cufile)
        flags = os.O_RDWR | os.O_CREAT if writable else os.O_RDONLY
        self._fd = os.open(self.path, flags | os.O_DIRECT, 0o600)
        try:
            descriptor = cufile.Descr.from_data(np.zeros(1, dtype=cufile.descr_dtype))
            descriptor.type = cufile.FileHandleType.OPAQUE_FD
            descriptor.handle["fd"] = self._fd
            with torch.cuda.device(self.device):
                self._handle = cufile.handle_register(descriptor.ptr)
        except BaseException:
            os.close(self._fd)
            raise
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="delta-gds")
        self._slots = threading.BoundedSemaphore(max_inflight)
        self._lock = threading.Lock()
        self._futures: list[Future[int]] = []
        self._closed = False

    def read_into(self, offset: int, dst_u8: torch.Tensor, *, expected_bytes: int | None = None) -> Future[int]:
        """Read exactly dst_u8.numel() bytes; result() establishes CPU I/O completion.

        The event orders allocator reuse and earlier work on the caller's current
        stream. Consumers on other streams must be explicitly ordered by caller.
        """
        self._validate_buffer(offset, dst_u8)
        expected = dst_u8.numel() if expected_bytes is None else expected_bytes
        if not 0 < expected <= dst_u8.numel():
            raise ValueError("Expected GDS read length must be positive and fit the destination")
        if expected != dst_u8.numel() and offset + expected != os.fstat(self._fd).st_size:
            raise ValueError("A known short GDS read is permitted only for the exact file tail")
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        return self._submit(offset, dst_u8, ready, write=False, expected=expected)

    def write_from(self, offset: int, src_u8: torch.Tensor, ready_event: torch.cuda.Event) -> Future[int]:
        """Write exact bytes after the recorded producer event, without a global sync."""
        if not self.writable:
            raise ValueError(f"GDS file is read-only: {self.path}")
        self._validate_buffer(offset, src_u8)
        if ready_event.device != self.device:
            raise ValueError("GDS write requires a recorded readiness event on the buffer's device")
        return self._submit(offset, src_u8, ready_event, write=True, expected=src_u8.numel())

    def _validate_buffer(self, offset: int, tensor: torch.Tensor) -> None:
        if tensor.device != self.device or tensor.dtype != torch.uint8 or tensor.ndim != 1:
            raise ValueError("GDS buffers must be flat uint8 tensors on the configured CUDA device")
        if not tensor.is_contiguous():
            raise ValueError("GDS buffers must be contiguous; implicit copies are not permitted")
        if offset < 0 or offset % GDS_ALIGNMENT or tensor.data_ptr() % GDS_ALIGNMENT:
            raise ValueError("GDS file offsets and buffer pointers must be 4096-byte aligned")
        if tensor.numel() == 0 or tensor.numel() % GDS_ALIGNMENT:
            raise ValueError("GDS I/O lengths must be positive multiples of 4096 bytes")

    def _submit(self, offset, tensor, ready_event, *, write: bool, expected: int) -> Future[int]:
        self._slots.acquire()
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Cannot submit I/O to a closed GDS backend")
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
        with torch.cuda.device(self.device):
            ready_event.synchronize()
            operation = self._cufile.write if write else self._cufile.read
            nbytes = tensor.numel()
            result = operation(self._handle, tensor.data_ptr(), nbytes, offset, 0)
        if result != expected:
            raise OSError(f"cuFile {'write' if write else 'read'} {self.path}: expected {expected}, received {result}")
        return result

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
        """Drain before deregistration, even if an operation failed. Does not fsync."""
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
                self._cufile.handle_deregister(self._handle)
            finally:
                os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
