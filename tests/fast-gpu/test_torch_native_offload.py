import torch
import torch.distributed as dist
import torch.nn as nn
from tests.ci.ci_register import register_cuda_ci
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from miles.backends.training_utils.torch_native.offload import move_train_state

register_cuda_ci(est_time=30, suite="stage-b-2-gpu-h200", labels=["fsdp", "torchtitan"], hardware=["hopper"])


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _train_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    tensors = [_local(param) for param in model.parameters()]
    for state in optimizer.state.values():
        tensors += [_local(value) for value in state.values() if isinstance(value, torch.Tensor) and value.dim() > 0]
    return tensors


def test_offload_round_trips_fsdp_state_through_pinned_host_memory(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29541")
    dist.init_process_group("nccl", rank=0, world_size=1)
    try:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(64, 64, bias=False), nn.Linear(64, 64, bias=False)).cuda()
        fully_shard(model, mesh=init_device_mesh("cuda", (1,)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        inputs = torch.randn(4, 64, device="cuda")
        model(inputs).pow(2).mean().backward()
        optimizer.step()
        before = [tensor.detach().cpu().clone() for tensor in _train_state(model, optimizer)]

        move_train_state([model], [optimizer], "cpu")
        offloaded = _train_state(model, optimizer)
        assert all(tensor.device.type == "cpu" and tensor.is_pinned() for tensor in offloaded)
        assert all(isinstance(param, DTensor) for param in model.parameters())

        move_train_state([model], [optimizer], "cuda")
        restored = _train_state(model, optimizer)
        assert all(tensor.device.type == "cuda" for tensor in restored)
        assert all(torch.equal(a, b.cpu()) for a, b in zip(before, restored, strict=True))

        model(inputs).pow(2).mean().backward()
        optimizer.step()
    finally:
        dist.destroy_process_group()
