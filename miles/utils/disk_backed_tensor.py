"""Host tensors backed by unlinked files on node-local NVMe, for optimizer state that does not fit RAM.

The pages live in reclaimable page cache, so the kernel writes them back instead of the process
hitting its memory limit, and an unlinked file leaves nothing behind when the run dies.
"""

import ctypes
import errno
import os
import shutil
import tempfile

import torch


def reserve_file(fd: int, nbytes: int) -> None:
    """Reserve blocks up front, so a full filesystem fails here as ENOSPC.

    Sizing a file with ftruncate alone leaves it sparse: the mapping succeeds and the
    process dies on SIGBUS at first touch instead, with nothing to point at.
    """
    try:
        os.posix_fallocate(fd, 0, nbytes)
    except OSError as e:
        if e.errno not in (errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL):
            raise
        os.ftruncate(fd, nbytes)


def disk_backed_like(tensor: torch.Tensor, directory: str) -> torch.Tensor:
    nbytes = max(tensor.numel() * tensor.element_size(), 1)
    fd, path = tempfile.mkstemp(dir=directory, suffix=".bin")
    try:
        reserve_file(fd, nbytes)
    finally:
        os.close(fd)
    storage = torch.UntypedStorage.from_file(path, shared=True, nbytes=nbytes)
    os.unlink(path)
    buffer = torch.empty(0, dtype=tensor.dtype).set_(storage, 0, tensor.shape)
    buffer._miles_disk_backed = True
    return buffer


def is_disk_backed(tensor: torch.Tensor) -> bool:
    return getattr(tensor, "_miles_disk_backed", False)


_MS_SYNC = 4
_libc = ctypes.CDLL(None, use_errno=True)


def flush_mapping(tensor: torch.Tensor) -> int:
    """msync one file-backed buffer, returning the bytes it covered.

    Checkpointing calls os.fsync on its own files, which waits on the kernel's writeback
    queue -- and our mappings are rewritten every step, so that queue is carrying gigabytes
    of our dirty pages by then. Flushing them here keeps that cost attributable and cheap
    to repeat: msync over an already-clean mapping returns immediately.
    """
    storage = tensor.untyped_storage()
    nbytes = storage.nbytes()
    if _libc.msync(ctypes.c_void_p(storage.data_ptr()), ctypes.c_size_t(nbytes), _MS_SYNC) != 0:
        raise OSError(ctypes.get_errno(), "msync of optimizer state mapping failed")
    return nbytes


def optimizer_state_dir_root(args) -> str:
    return os.path.join(args.offload_train_disk_dir, "optimizer_state")


def purge_rank_dir(dir_root: str) -> str:
    """Drop everything this rank left behind, before any store claims its own path.

    A store only removes the exact path it is about to use, so state written under a
    different layout -- another parallelism, a renamed directory scheme -- survives
    forever, and a run killed by the scheduler never reaches its atexit cleanup either.
    On a 744B DP1 model that is hundreds of GB per rank per stale run, and node-local
    NVMe fills up until allocation fails. The rank subtree is exclusively this rank's,
    so clearing it whole is safe, and it must happen before the chained dense and
    expert stores are constructed, since they share it.
    """
    rank_dir = os.path.join(dir_root, f"rank{torch.distributed.get_rank():05d}")
    shutil.rmtree(rank_dir, ignore_errors=True)
    os.makedirs(rank_dir, exist_ok=True)
    return rank_dir
