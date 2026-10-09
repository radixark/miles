"""File-backed Adam moments must update exactly like in-memory ones and actually live in the file."""

import socket

import numpy as np
import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cpu_ci
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from miles.backends.torchtitan_utils import disk_optimizer_state

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

_MOMENTS = ("exp_avg", "exp_avg_sq")


def _make_params() -> list[torch.nn.Parameter]:
    torch.manual_seed(0)
    return [torch.nn.Parameter(torch.randn(16, 8)), torch.nn.Parameter(torch.randn(5))]


def _train(optimizer: torch.optim.Optimizer, params: list[torch.nn.Parameter], *, steps: int) -> None:
    gradients = torch.Generator().manual_seed(1)
    for _ in range(steps):
        for param in params:
            param.grad = (torch.randn(param.shape, generator=gradients) * 1e-2).to(param.dtype)
            if isinstance(param, DTensor):
                param.grad = distribute_tensor(param.grad, param.device_mesh, param.placements)
        optimizer.step()


@pytest.fixture
def single_rank_cpu_mesh():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def test_file_backed_moments_update_like_in_memory_adamw(tmp_path):
    in_memory, file_backed = _make_params(), _make_params()
    reference = torch.optim.AdamW(in_memory, lr=1e-2, fused=True)
    optimizer = torch.optim.AdamW(file_backed, lr=1e-2, fused=True)
    disk_optimizer_state.install([optimizer], directory=str(tmp_path), rank=3)

    _train(reference, in_memory, steps=3)
    _train(optimizer, file_backed, steps=3)

    for expected, actual in zip(in_memory, file_backed, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    on_disk = np.fromfile(tmp_path / "rank00003_optimizer0.bin", dtype=np.float32)
    in_state = torch.cat([optimizer.state[param][name].flatten() for param in file_backed for name in _MOMENTS])
    np.testing.assert_array_equal(on_disk, in_state.numpy(), err_msg="the moments are not the file's pages")


def test_dtensor_params_get_file_backed_dtensor_moments(tmp_path, single_rank_cpu_mesh):
    params = [
        torch.nn.Parameter(distribute_tensor(p.detach(), single_rank_cpu_mesh, [Shard(0)])) for p in _make_params()
    ]
    optimizer = torch.optim.AdamW(params, lr=1e-2, fused=True)
    disk_optimizer_state.install([optimizer], directory=str(tmp_path), rank=0)

    for param in params:
        for name in _MOMENTS:
            moment = optimizer.state[param][name]
            assert isinstance(moment, DTensor), f"{name} lost the param's DTensor spec"
            assert moment.placements == param.placements and moment.to_local().device.type == "cpu"
    _train(optimizer, params, steps=2)
    assert np.fromfile(tmp_path / "rank00000_optimizer0.bin", dtype=np.float32).any()


def test_install_after_the_first_step_is_refused(tmp_path):
    params = _make_params()
    optimizer = torch.optim.AdamW(params, lr=1e-2, fused=True)
    _train(optimizer, params, steps=1)
    with pytest.raises(RuntimeError, match="before the optimizer has state"):
        disk_optimizer_state.install([optimizer], directory=str(tmp_path), rank=0)
