import ctypes
import mmap
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


@pytest.fixture
def nvme_setup_chain(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    original_empty = torch.empty

    def unpinned_empty(*args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", unpinned_empty)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    optimizer = object.__new__(DistributedOptimizer)
    optimizer.is_stub_optimizer = False
    optimizer.config = SimpleNamespace(
        defer_main_param_initialization=True,
        use_precision_aware_optimizer=False,
        optimizer_cpu_offload=False,
        offload_optimizer_states=False,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        log_num_zeros_in_grad=False,
    )
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    optimizer.model_float16_groups = []
    optimizer.shard_fp32_from_float16_groups = []
    optimizer.model_fp32_groups = []
    optimizer.shard_fp32_groups = []
    optimizer.grad_scaler = None
    optimizer.distributed_optimizer_instance_id = 0
    return SimpleNamespace(chained_optimizers=[optimizer], get_grad_norm=lambda: 0.0)


def make_nvme_setup_args(*, directory: Path, no_load_optim: bool) -> Namespace:
    return Namespace(
        offload_train_disk_dir=str(directory),
        offload_train_disk_chunk_mb=1,
        stream_optimizer_state_moment_dtype="fp32",
        stream_optimizer_state_page_cache_mb=None,
        no_load_optim=not no_load_optim,
        backend=Namespace(no_load_optim=no_load_optim),
    )


def resident_pages(fd: int) -> int:
    """Pages of ``fd`` resident in the host page cache, from mincore over a shared mapping."""
    size = os.fstat(fd).st_size
    libc = ctypes.CDLL(None, use_errno=True)
    with mmap.mmap(fd, size, access=mmap.ACCESS_WRITE) as mapping:
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        pages = (ctypes.c_ubyte * ((size + mmap.PAGESIZE - 1) // mmap.PAGESIZE))()
        assert libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(size), pages) == 0, ctypes.get_errno()
        return sum(page & 1 for page in pages)


def require_evictable(directory) -> None:
    """Skip unless files under ``directory`` can leave the page cache, which tmpfs pages, being the storage, cannot."""
    if not hasattr(os, "posix_fadvise"):
        pytest.skip("page-cache control is Linux-only")
    fd = os.open(os.path.join(directory, "evictable-probe.bin"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.pwrite(fd, b"\x01" * mmap.PAGESIZE, 0)
        # Raw syscalls rather than the store's helper, so a broken helper fails the tests instead of skipping them.
        os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        evictable = resident_pages(fd) == 0
    finally:
        os.close(fd)
    if not evictable:
        pytest.skip(f"{directory} keeps its pages resident (tmpfs), so nothing can be evicted")
