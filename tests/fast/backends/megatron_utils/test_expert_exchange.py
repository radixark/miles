import pytest
import torch

from miles.backends.megatron_utils.update_weight.expert_exchange import plan_expert_exchange
from miles.backends.megatron_utils.update_weight.expert_payload import build_payload_layout, pack_units, unpack_units
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfNameRoute


def _expert_unit(rank: int, expert: int) -> list[tuple[str, torch.Tensor]]:
    prefix = f"model.layers.3.mlp.experts.{4 * rank + expert}"
    generator = torch.Generator().manual_seed(100 * rank + expert)
    return [
        (f"{prefix}.down_proj.weight", torch.randn(4, 6, generator=generator).to(torch.bfloat16)),
        (f"{prefix}.down_proj.weight_scale_inv", torch.randn(1, 2, generator=generator)),
        # an odd byte count, so the next segment starts misaligned unless the plan pads it
        (f"{prefix}.down_proj.bias_bytes", torch.randint(0, 256, (3,), dtype=torch.uint8, generator=generator)),
    ]


def _exchange_by_plans(units_by_rank, route):
    """Runs every rank's plan, then hands rank r segment r of each rank's send payload, as the pairwise sends do."""
    layouts_by_rank = [build_payload_layout(units) for units in units_by_rank]
    plans = [plan_expert_exchange(layouts_by_rank, route, rank) for rank in range(len(units_by_rank))]
    send_segments_by_rank = []
    for units, plan in zip(units_by_rank, plans, strict=True):
        payload = torch.empty(sum(plan.send_split_sizes), dtype=torch.uint8)
        segments = payload.split(plan.send_split_sizes)
        for unit_indices, layout, segment in zip(
            plan.unit_indices_by_dst_rank, plan.send_layouts_by_dst_rank, segments, strict=True
        ):
            pack_units([units[index] for index in unit_indices], layout, segment)
        send_segments_by_rank.append(segments)
    received_by_rank = []
    for rank, plan in enumerate(plans):
        incoming = [segments[rank] for segments in send_segments_by_rank]
        assert [segment.numel() for segment in incoming] == plan.recv_split_sizes
        payload = torch.cat(incoming)
        received_by_rank.append(
            [
                unit
                for layout, segment in zip(
                    plan.recv_layouts_by_src_rank, payload.split(plan.recv_split_sizes), strict=True
                )
                for unit in unpack_units(layout, segment)
            ]
        )
    return received_by_rank


def test_each_rank_receives_exactly_the_units_its_route_names():
    """A rank missing a needed expert leaves its engine weights stale; a rank receiving extras wastes the exchange.
    Send and receive sizes must agree pairwise, or a receive reads the wrong bytes."""
    units_by_rank = [[_expert_unit(rank, expert) for expert in range(4)] for rank in range(4)]
    all_units = {unit[0][0]: unit for units in units_by_rank for unit in units}
    # engine rank r loads experts of two different sources; rank 3 sends to no engine
    needed_weight_names = [
        {f"model.layers.3.mlp.experts.{expert}.down_proj.weight" for expert in experts}
        for experts in ([0, 5], [1, 2, 15], [9], [])
    ]
    route = HfNameRoute(
        hf_names_by_rank=tuple(
            frozenset(name for weight in names for name, _ in all_units[weight]) for names in needed_weight_names
        )
    )

    received_by_rank = _exchange_by_plans(units_by_rank, route)

    for received, names in zip(received_by_rank, needed_weight_names, strict=True):
        assert {unit[0][0] for unit in received} == names
        for unit in received:
            for (name, tensor), (expected_name, expected) in zip(unit, all_units[unit[0][0]], strict=True):
                assert name == expected_name and tensor.dtype == expected.dtype and torch.equal(tensor, expected)


def test_a_rank_needing_part_of_a_unit_is_refused():
    """Delivering a weight without its scales would load unscaled fp8 bytes."""
    unit = _expert_unit(0, 0)
    layouts_by_rank = [build_payload_layout([unit])]

    with pytest.raises(AssertionError, match="a unit travels whole"):
        plan_expert_exchange(layouts_by_rank, HfNameRoute(hf_names_by_rank=(frozenset({unit[0][0]}),)), 0)
