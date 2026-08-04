"""Four-rank worker for FSDP2 hybrid-shard gradient parity."""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from miles.backends.experimental.fsdp_utils.dtensor import gather_full_param
from miles.backends.experimental.fsdp_utils.parallel import build_fsdp_meshes


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


class _TinyModel(nn.Module):
    def __init__(self, dim: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block(dim) for _ in range(depth))
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def _make_model() -> _TinyModel:
    torch.manual_seed(1234)
    return _TinyModel().cuda()


def _materialize_gradient(param: nn.Parameter) -> torch.Tensor:
    grad = param.grad
    assert grad is not None
    return grad.full_tensor() if isinstance(grad, DTensor) else grad


def _reference_gradients(inputs: torch.Tensor, world_size: int) -> dict[str, torch.Tensor]:
    model = _make_model()
    model(inputs).square().mean().backward()

    gradients = {}
    for name, param in model.named_parameters():
        assert param.grad is not None
        gradient = param.grad.detach().clone()
        dist.all_reduce(gradient)
        gradient /= world_size
        gradients[name] = gradient
    return gradients


def _fully_shard_model(model: _TinyModel, mesh) -> None:
    for block in model.blocks:
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    shard_size = world_size // args.replicate_size

    meshes = build_fsdp_meshes(
        device_type="cuda",
        world_size=world_size,
        context_parallel_size=1,
        dp_replicate_size=args.replicate_size,
    )
    fsdp_mesh = meshes["fsdp"]
    assert fsdp_mesh.ndim == (1 if args.replicate_size == 1 else 2)

    generator = torch.Generator(device="cuda").manual_seed(9000 + rank)
    inputs = torch.randn(8, 32, generator=generator, device="cuda")
    expected_gradients = _reference_gradients(inputs, world_size)

    model = _make_model()
    initial_weights = {name: param.detach().clone() for name, param in model.named_parameters()}
    _fully_shard_model(model, fsdp_mesh)
    model(inputs).square().mean().backward()

    gradients = {}
    for name, param in model.named_parameters():
        actual = _materialize_gradient(param).detach()
        torch.testing.assert_close(actual, expected_gradients[name], rtol=2e-5, atol=2e-6)

        peers = [torch.empty_like(actual) for _ in range(world_size)]
        dist.all_gather(peers, actual)
        for peer in peers[1:]:
            torch.testing.assert_close(peers[0], peer, rtol=0, atol=0)
        gradients[name] = actual.cpu()

        full_param = gather_full_param(param)
        torch.testing.assert_close(full_param, initial_weights[name])

    if rank == 0:
        torch.save(gradients, args.output)
        print(f"PASS r{args.replicate_size}s{shard_size}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
