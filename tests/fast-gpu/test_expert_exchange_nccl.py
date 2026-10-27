from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, suite="stage-b-2-gpu-h200", labels=["weight-update"], hardware=["hopper", "blackwell"], num_gpus=2
)

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.fast.dist_utils import find_free_port

from miles.backends.megatron_utils.update_weight.expert_exchange import ExpertExchange
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfNameRoute

_WORLD_SIZE = 2
_NUM_BATCHES = 3
_EXPERTS_PER_RANK = 3
# global expert index -> the ranks that load it: shared, unneeded, self-only and cross-rank units
_LOADING_RANKS_BY_EXPERT = {0: {0, 1}, 1: set(), 2: {0}, 3: {1}, 4: {0, 1}, 5: {0, 1}}


def _expert_units(rank: int, batch: int) -> list[list[tuple[str, torch.Tensor]]]:
    generator = torch.Generator().manual_seed(1000 * batch + rank)
    units = []
    for expert in range(_EXPERTS_PER_RANK * rank, _EXPERTS_PER_RANK * (rank + 1)):
        prefix = f"model.layers.{batch}.mlp.experts.{expert}"
        units.append(
            [
                (f"{prefix}.down_proj.weight", torch.randn(64, 48, generator=generator).to(torch.bfloat16)),
                (f"{prefix}.down_proj.weight_scale_inv", torch.randn(1, 3, generator=generator)),
                # an odd byte count, so the next segment starts misaligned unless the plan pads it
                (
                    f"{prefix}.down_proj.bias_bytes",
                    torch.randint(0, 256, (5,), dtype=torch.uint8, generator=generator),
                ),
            ]
        )
    return units


def _expert_of(unit) -> int:
    return int(unit[0][0].split(".experts.")[1].split(".")[0])


def _route() -> HfNameRoute:
    return HfNameRoute(
        hf_names_by_rank=tuple(
            frozenset(
                name
                for batch in range(_NUM_BATCHES)
                for src_rank in range(_WORLD_SIZE)
                for unit in _expert_units(src_rank, batch)
                if rank in _LOADING_RANKS_BY_EXPERT[_expert_of(unit)]
                for name, _ in unit
            )
            for rank in range(_WORLD_SIZE)
        )
    )


def _exchange_with_lookahead(exchanges: list[ExpertExchange], rank: int) -> list[list]:
    """Starts batch k+1 before finishing batch k, as the direct iterator does, and reuses freed memory while a batch
    is in flight, so a payload released too early gets overwritten."""
    received_by_batch, previous_batch = [], None
    for batch, exchange in enumerate(exchanges):
        units = [[(name, tensor.cuda()) for name, tensor in unit] for unit in _expert_units(rank, batch)]
        in_flight = exchange.start(units, device="cuda")
        del units
        torch.empty(1 << 20, dtype=torch.uint8, device="cuda").fill_(255)
        if previous_batch is not None:
            received_by_batch.append(previous_batch[0].finish(previous_batch[1]))
        previous_batch = (exchange, in_flight)
    received_by_batch.append(previous_batch[0].finish(previous_batch[1]))
    torch.cuda.synchronize()
    return received_by_batch


def _run_rank(rank: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=_WORLD_SIZE)
    try:
        route = _route()
        exchanges = [ExpertExchange(group=dist.group.WORLD, route=route) for _ in range(_NUM_BATCHES)]
        # the second sync reuses each exchange's cached plan
        for _ in range(2):
            received_by_batch = _exchange_with_lookahead(exchanges, rank)
            for batch, received in enumerate(received_by_batch):
                expected = [
                    unit
                    for src_rank in range(_WORLD_SIZE)
                    for unit in _expert_units(src_rank, batch)
                    if rank in _LOADING_RANKS_BY_EXPERT[_expert_of(unit)]
                ]
                assert [[name for name, _ in unit] for unit in received] == [
                    [name for name, _ in unit] for unit in expected
                ], f"rank {rank} batch {batch} received the wrong units"
                for received_unit, expected_unit in zip(received, expected, strict=True):
                    for (name, tensor), (_, expected_tensor) in zip(received_unit, expected_unit, strict=True):
                        assert torch.equal(tensor.cpu(), expected_tensor), f"rank {rank}: {name} bytes differ"
    finally:
        dist.destroy_process_group()


def test_each_rank_receives_exactly_the_bytes_of_the_units_its_route_names():
    """A missing, misrouted or duplicated unit leaves an engine weight stale or wrong, and a payload freed while its
    sends are in flight corrupts the next batch; the update would serve those bytes silently."""
    mp.spawn(_run_rank, args=(find_free_port(),), nprocs=_WORLD_SIZE, join=True)
