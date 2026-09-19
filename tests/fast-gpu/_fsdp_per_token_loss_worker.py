"""Two-rank worker: per-token loss scaling lands the FSDP2 gradient on the global token mean."""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from miles.backends.fsdp_utils.loss_scaling import get_per_token_loss_scales
from miles.backends.fsdp_utils.parallel import create_fsdp_parallel_state
from miles.backends.training_utils.data import DataIterator
from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.utils.distributed_utils import init_gloo_group


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


class _TinyModel(nn.Module):
    def __init__(self, dim: int = 32, depth: int = 2) -> None:
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


def _fully_shard_model(model: _TinyModel, mesh) -> None:
    for block in model.blocks:
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)


def _token_loss_sums(model: nn.Module, samples) -> list[torch.Tensor]:
    """Masked sum of per-token scalar losses, one entry per single-sample micro-batch."""
    return [(model(tokens.unsqueeze(0)).sum(dim=-1).squeeze(0) * mask).sum() for tokens, mask in samples]


def _full_gradient(param: nn.Parameter) -> torch.Tensor:
    grad = param.grad
    assert grad is not None
    return grad.full_tensor() if isinstance(grad, DTensor) else grad


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    init_gloo_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2

    set_parallel_state(create_fsdp_parallel_state(argparse.Namespace(dp_replicate_size=1)))

    generator = torch.Generator(device="cuda").manual_seed(5000 + rank)
    # the two ranks own unequal token counts, so any per-rank normalization would disagree
    lengths = (3, 5) if rank == 0 else (2, 8)
    masks = [torch.full((n,), 1, dtype=torch.int, device="cuda") for n in lengths]
    samples = [
        (torch.randn(n, 32, generator=generator, device="cuda"), mask)
        for n, mask in zip(lengths, masks, strict=True)
    ]

    global_num_tokens = torch.tensor(sum(int(mask.sum()) for mask in masks), device="cuda")
    dist.all_reduce(global_num_tokens, op=dist.ReduceOp.SUM)

    # reference: the global token mean, from an unsharded model on summed local gradients
    reference_model = _make_model()
    sum(_token_loss_sums(reference_model, samples)).backward()
    reference_gradients = {}
    for name, param in reference_model.named_parameters():
        gradient = param.grad.detach().clone()
        dist.all_reduce(gradient)
        gradient /= global_num_tokens
        reference_gradients[name] = gradient

    # scaled: the production scan + pre-scaling on the FSDP2-wrapped model
    data_iterator = DataIterator({"loss_masks": masks}, micro_batch_size=1)
    (scale,) = get_per_token_loss_scales(data_iterator, [2])

    mesh = get_parallel_state().get_mesh("fsdp")
    model = _make_model()
    _fully_shard_model(model, mesh)
    for microbatch_loss in _token_loss_sums(model, samples):
        (microbatch_loss * scale).backward()
    for name, param in model.named_parameters():
        torch.testing.assert_close(_full_gradient(param), reference_gradients[name], rtol=2e-5, atol=2e-6)

    # unscaled control: the legacy path leaves the gradient global_num_tokens / world_size too large
    unscaled_model = _make_model()
    _fully_shard_model(unscaled_model, mesh)
    for microbatch_loss in _token_loss_sums(unscaled_model, samples):
        microbatch_loss.backward()
    legacy_ratio = global_num_tokens / world_size
    for name, param in unscaled_model.named_parameters():
        torch.testing.assert_close(
            _full_gradient(param), reference_gradients[name] * legacy_ratio, rtol=2e-5, atol=2e-6
        )

    if rank == 0:
        print("PASS fsdp-per-token-loss", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
