from dataclasses import FrozenInstanceError

import pytest

from miles_plugins.models.cp_layout import build_cp_layout_metadata


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("multipliers", [(2, 6, 1), (1,), (8, 2, 10, 3)])
def test_peer_plans_cover_every_token(cp_size, multipliers):
    boundaries = [0]
    for multiplier in multipliers:
        boundaries.append(boundaries[-1] + multiplier * cp_size)
    layouts = [build_cp_layout_metadata(boundaries, cp_size, rank) for rank in range(cp_size)]
    for plans in (
        [layout.zigzag_to_contiguous for layout in layouts],
        [layout.contiguous_to_zigzag for layout in layouts],
    ):
        for rank, plan in enumerate(plans):
            local_total = layouts[rank].local_total_len
            assert sum(plan.send_sizes) == sum(plan.recv_sizes) == local_total
            sources = sorted(src + offset for peer in plan.send for src, _, count in peer for offset in range(count))
            destinations = sorted(
                dst + offset for peer in plan.recv for _, dst, count in peer for offset in range(count)
            )
            assert sources == destinations == list(range(local_total))
            for peer, peer_plan in enumerate(plans):
                assert plan.send[peer] == peer_plan.recv[rank]
                assert plan.send_sizes[peer] == peer_plan.recv_sizes[rank]
    with pytest.raises(FrozenInstanceError):
        layouts[0].cp_rank = 10


@pytest.mark.parametrize(
    "boundaries,cp_size,cp_rank",
    [([0, 8], 0, 0), ([0, 8], 2, 2), ([1, 8], 2, 0), ([0, 0, 8], 2, 0), ([0, 7], 2, 0), ([0, 8.0], 2, 0)],
)
def test_invalid_layout_rejected(boundaries, cp_size, cp_rank):
    with pytest.raises(ValueError):
        build_cp_layout_metadata(boundaries, cp_size, cp_rank)


def test_zero_peer_splits_and_cross_sequence_shards():
    layout = build_cp_layout_metadata([0, 8, 12], 2, 0)
    assert layout.zigzag_to_contiguous.send_sizes == layout.zigzag_to_contiguous.recv_sizes == (2, 4)
    layout = build_cp_layout_metadata([0, 4], 4, 0)
    assert layout.zigzag_to_contiguous.send_sizes == (1, 0, 0, 0)
