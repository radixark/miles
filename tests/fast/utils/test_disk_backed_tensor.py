"""The file-backed buffers behind disk-resident optimizer state (Megatron Muon, torchtitan Adam).

Guards a silent failure: if the allocator stops returning file-backed storage, the offloader
keeps working against pinned host memory while the log still claims otherwise.
"""

import os

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from miles.utils import disk_backed_tensor

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def test_disk_buffer_matches_shape_and_dtype_and_is_not_pinned(tmp_path):
    src = torch.randn(64, 32, dtype=torch.float32)

    buf = disk_backed_tensor.disk_backed_like(src, str(tmp_path))

    assert buf.shape == src.shape
    assert buf.dtype == src.dtype
    assert buf.device.type == "cpu"
    # The inherited offloader picks its sync/async copy path off is_pinned().
    assert not buf.is_pinned()


def test_disk_buffer_round_trip_is_bit_exact(tmp_path):
    src = torch.randn(128, 64, dtype=torch.float32)
    buf = disk_backed_tensor.disk_backed_like(src, str(tmp_path))

    buf.copy_(src)
    out = torch.empty_like(src)
    out.copy_(buf)

    assert torch.equal(out, src)


def test_disk_buffer_leaves_no_file_behind(tmp_path):
    disk_backed_tensor.disk_backed_like(torch.zeros(8), str(tmp_path))

    # Unlinked at creation, so a killed run leaves no residue.
    assert os.listdir(tmp_path) == []


def test_disk_buffer_is_recognized_as_already_managed(tmp_path):
    """Megatron's checkpoint adoption reallocates non-pinned CPU state; ours must be exempt."""
    buf = disk_backed_tensor.disk_backed_like(torch.zeros(32, 8), str(tmp_path))

    assert disk_backed_tensor.is_disk_backed(buf)
    assert not disk_backed_tensor.is_disk_backed(torch.zeros(32, 8))


def test_flush_mapping_covers_the_buffer_and_repeats_cheaply(tmp_path):
    """Checkpointing fsyncs its own files behind the kernel's writeback of ours."""
    buf = disk_backed_tensor.disk_backed_like(torch.zeros(1024, 256), str(tmp_path))
    nbytes = buf.numel() * buf.element_size()
    buf.fill_(1.0)

    assert disk_backed_tensor.flush_mapping(buf) == nbytes
    # Already clean, so the repeat is the cheap case the checkpoint hook relies on.
    assert disk_backed_tensor.flush_mapping(buf) == nbytes


def test_reserve_sizes_the_file(tmp_path):
    path = tmp_path / "f.bin"
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        disk_backed_tensor.reserve_file(fd, 4096)
        assert os.fstat(fd).st_size == 4096
    finally:
        os.close(fd)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_disk_buffer_preserves_dtype(tmp_path, dtype):
    src = torch.zeros(16, 4, dtype=dtype)

    buf = disk_backed_tensor.disk_backed_like(src, str(tmp_path))

    assert buf.dtype is dtype
    assert buf.numel() == src.numel()
