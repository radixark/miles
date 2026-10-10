"""File-backed Adam moments must update exactly like in-memory ones, live in file pages, and survive a resume."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cpu_ci
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from miles.backends.torchtitan_utils.disk_optimizer_state import move_adam_moments_to_disk

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

_MOMENTS = ("exp_avg", "exp_avg_sq")


def _make_params() -> list[torch.nn.Parameter]:
    torch.manual_seed(0)
    return [torch.nn.Parameter(torch.randn(16, 8)), torch.nn.Parameter(torch.randn(5))]


def _train(optimizer: torch.optim.Optimizer, params: list[torch.nn.Parameter], *, steps: int, seed: int = 1) -> None:
    gradients = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        for param in params:
            param.grad = (torch.randn(param.shape, generator=gradients) * 1e-2).to(param.dtype)
            if isinstance(param, DTensor):
                param.grad = distribute_tensor(param.grad, param.device_mesh, param.placements)
        optimizer.step()


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _is_file_backed(tensor: torch.Tensor) -> bool:
    return bool(_local(tensor).untyped_storage().filename)


@pytest.fixture
def single_rank_gloo():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_file_backed_moments_update_like_in_memory_adamw(tmp_path, single_rank_gloo):
    in_memory, file_backed = _make_params(), _make_params()
    reference = torch.optim.AdamW(in_memory, lr=1e-2, fused=True)
    optimizer = torch.optim.AdamW(file_backed, lr=1e-2, fused=True)
    move_adam_moments_to_disk([optimizer], state_dir_root=str(tmp_path))

    _train(reference, in_memory, steps=3)
    _train(optimizer, file_backed, steps=3)

    for expected, actual in zip(in_memory, file_backed, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for param in file_backed:
        for name in _MOMENTS:
            assert _is_file_backed(optimizer.state[param][name]), f"{name} drifted off the file pages"
    assert os.listdir(tmp_path / "rank00000") == [], "the backing file outlives its mapping"


def test_dtensor_params_get_file_backed_dtensor_moments(tmp_path, single_rank_gloo):
    mesh = init_device_mesh("cpu", (1,))
    params = [torch.nn.Parameter(distribute_tensor(p.detach(), mesh, [Shard(0)])) for p in _make_params()]
    optimizer = torch.optim.AdamW(params, lr=1e-2, fused=True)
    move_adam_moments_to_disk([optimizer], state_dir_root=str(tmp_path))

    _train(optimizer, params, steps=2)
    for param in params:
        for name in _MOMENTS:
            moment = optimizer.state[param][name]
            assert isinstance(moment, DTensor), f"{name} lost the param's DTensor spec"
            assert moment.placements == param.placements and _is_file_backed(moment)


def test_moments_a_checkpoint_loaded_carry_over(tmp_path, single_rank_gloo):
    """Resuming loads the moments before they move to disk; moving must keep them, not zero them."""
    reference, resumed = _make_params(), _make_params()
    reference_optimizer = torch.optim.AdamW(reference, lr=1e-2, fused=True)
    resumed_optimizer = torch.optim.AdamW(resumed, lr=1e-2, fused=True)
    _train(reference_optimizer, reference, steps=2)
    _train(resumed_optimizer, resumed, steps=2)

    move_adam_moments_to_disk([resumed_optimizer], state_dir_root=str(tmp_path))
    _train(reference_optimizer, reference, steps=2, seed=2)
    _train(resumed_optimizer, resumed, steps=2, seed=2)

    for expected, actual in zip(reference, resumed, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
