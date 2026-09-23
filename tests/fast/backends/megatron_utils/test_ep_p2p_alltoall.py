"""The P2P ring must reproduce all_to_all_single exactly, including on changing uneven splits."""

import random

import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.megatron_utils.ep_p2p_alltoall import ring_all_to_all_single

HIDDEN = 3


def _split_matrix(world_size: int, seed: int, *, allow_zero: bool) -> list[list[int]]:
    """counts[src][dst], identical on every rank because the seed is shared."""
    rng = random.Random(seed)
    low = 0 if allow_zero else 1
    return [[rng.randint(low, 7) for _ in range(world_size)] for _ in range(world_size)]


def _payload(src: int, dst: int, count: int) -> torch.Tensor:
    """Rows that name their sender, receiver and position, so misplacement is visible."""
    base = torch.arange(count, dtype=torch.float32).unsqueeze(1).expand(count, HIDDEN)
    return base + 1000.0 * src + 100.0 * dst


def _exchange(rank: int, world_size: int, counts: list[list[int]], peer_ranks: list[int], group) -> None:
    my_index = peer_ranks.index(rank)
    input_splits = counts[my_index]
    output_splits = [counts[src][my_index] for src in range(world_size)]
    send = torch.cat([_payload(my_index, dst, n) for dst, n in enumerate(input_splits)])
    recv = torch.full((sum(output_splits), HIDDEN), -1.0)

    ring_all_to_all_single(
        recv,
        send,
        output_splits,
        input_splits,
        peer_ranks=peer_ranks,
        my_index=my_index,
        transport_group=group,
    )

    expected = torch.cat([_payload(src, my_index, n) for src, n in enumerate(output_splits)])
    torch.testing.assert_close(recv, expected, rtol=0, atol=0)


def _worker_changing_splits(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        peer_ranks = list(range(world_size))
        for iteration in range(20):
            counts = _split_matrix(world_size, seed=iteration, allow_zero=iteration % 2 == 0)
            _exchange(rank, world_size, counts, peer_ranks, group=None)
    finally:
        dist.destroy_process_group()


def _worker_subgroup(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        # Group order differs from global order, so peers must be addressed by global rank.
        order = [2, 0, 3, 1]
        group = dist.new_group(ranks=order)
        for iteration in range(5):
            _exchange(rank, world_size, _split_matrix(world_size, seed=100 + iteration, allow_zero=True), order, group)
    finally:
        dist.destroy_process_group()


def test_ring_matches_all_to_all_on_changing_uneven_splits():
    run_multiprocess(_worker_changing_splits, world_size=4)


def test_ring_addresses_peers_by_global_rank_in_a_reordered_group():
    run_multiprocess(_worker_subgroup, world_size=4)
