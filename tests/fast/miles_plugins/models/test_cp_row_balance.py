"""CPU tests for the causal row balancer shared by CP indexers.

Every rank's plan is built on the host and the two all-to-alls are replayed in Python, so the
routing, the balance and the gate are checked without GPUs; the exchange then runs over a gloo
process group. Rows carry their own global position, which makes a row routed to the wrong rank
or returned out of order visible.
"""

import random
from functools import partial

import pytest
import torch
import torch.distributed as dist

from tests.ci.ci_register import register_cpu_ci
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles_plugins.models.cp_row_balance import plan_causal_row_balance, scoring_rank_of_chunk, send_rows_to_scorers

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])


def _plans(seq_lens, cp_size, min_gain=0.1):
    return [
        plan_causal_row_balance(tuple(seq_lens), cp_rank=rank, cp_size=cp_size, device="cpu", min_gain=min_gain)
        for rank in range(cp_size)
    ]


def _reference_plan(seq_lens, cp_rank, cp_size, min_gain):
    """The balancer spelled out one row at a time: each row's chunk, scoring rank and owner."""
    total = sum(seq_lens)
    rank_rows = total // cp_size
    lens = torch.tensor(seq_lens)
    positions = torch.arange(total)
    seq = torch.repeat_interleave(torch.arange(len(seq_lens)), lens)
    offset = positions - (torch.cumsum(lens, 0) - lens)[seq]
    scorer = scoring_rank_of_chunk(offset * 2 * cp_size // lens[seq], cp_size)
    owner = positions // rank_rows
    cost = (offset + 1).double()
    contiguous = torch.zeros(cp_size, dtype=torch.float64).index_add_(0, owner, cost)
    balanced = torch.zeros(cp_size, dtype=torch.float64).index_add_(0, scorer, cost)
    if balanced.max() > (1 - min_gain) * contiguous.max():
        return None
    local_scorer = scorer[cp_rank * rank_rows : (cp_rank + 1) * rank_rows]
    mine = scorer == cp_rank
    return dict(
        send_rows=torch.argsort(local_scorer, stable=True),
        input_splits=tuple(torch.bincount(local_scorer, minlength=cp_size).tolist()),
        output_splits=tuple(torch.bincount(owner[mine], minlength=cp_size).tolist()),
        scored_positions=positions[mine],
    )


def _all_to_all(send_buffers, send_splits, recv_splits):
    """What all_to_all_single delivers: rank r gets each peer's slice for r, in peer order."""
    cp_size = len(send_buffers)
    chunks = [list(buffer.split(list(splits))) for buffer, splits in zip(send_buffers, send_splits, strict=True)]
    received = []
    for dst in range(cp_size):
        pieces = [chunks[src][dst] for src in range(cp_size)]
        assert [piece.shape[0] for piece in pieces] == list(recv_splits[dst]), "sender and receiver disagree"
        received.append(torch.cat(pieces))
    return received


def _costs(seq_lens):
    return torch.cat([torch.arange(1, n + 1) for n in seq_lens]).double()


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("seed", range(12))
def test_matches_the_row_by_row_reference(cp_size, seed):
    rng = random.Random(seed)
    # empty, shorter-than-2cp and long sequences; the last one pads the stream to a multiple of cp
    seq_lens = [rng.choice([0, 1, 3, rng.randint(1, 64), rng.randint(1, 3000)]) for _ in range(rng.randint(1, 10))]
    seq_lens.append(cp_size - sum(seq_lens) % cp_size)
    for min_gain in (-1.0, 0.1):
        for rank in range(cp_size):
            plan = plan_causal_row_balance(seq_lens, cp_rank=rank, cp_size=cp_size, device="cpu", min_gain=min_gain)
            reference = _reference_plan(seq_lens, rank, cp_size, min_gain)
            assert (plan is None) == (reference is None), f"gate differs for {seq_lens}"
            if plan is not None:
                assert torch.equal(plan.send_rows, reference["send_rows"])
                assert plan.input_splits == reference["input_splits"]
                assert plan.output_splits == reference["output_splits"]
                assert torch.equal(plan.scored_positions, reference["scored_positions"])


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("seq_lens", [[4096], [3000, 1096], [2048, 1024, 512, 512], [4000, 90, 6]])
def test_rows_reach_their_scorer_and_come_back_in_order(cp_size, seq_lens):
    plans = _plans(seq_lens, cp_size, min_gain=-1.0)
    rank_rows = sum(seq_lens) // cp_size
    local_rows = [torch.arange(r * rank_rows, (r + 1) * rank_rows) for r in range(cp_size)]

    sent = [rows[plan.send_rows] for rows, plan in zip(local_rows, plans, strict=True)]
    scored = _all_to_all(sent, [p.input_splits for p in plans], [p.output_splits for p in plans])
    for plan, rows in zip(plans, scored, strict=True):
        assert torch.equal(rows, plan.scored_positions)

    results = [rows * 10 for rows in scored]
    returned = _all_to_all(results, [p.output_splits for p in plans], [p.input_splits for p in plans])
    for plan, rows, back in zip(plans, local_rows, returned, strict=True):
        local = torch.empty_like(back)
        local[plan.send_rows] = back
        assert torch.equal(local, rows * 10)


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_one_sequence_balances_to_within_a_percent(cp_size):
    seq_lens = [cp_size * 8192]
    costs = _costs(seq_lens)
    per_rank = torch.stack([costs[plan.scored_positions].sum() for plan in _plans(seq_lens, cp_size)])

    assert per_rank.max() / per_rank.min() < 1.01


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_one_sequence_is_a_single_pairwise_swap(cp_size):
    """Rank r keeps its first half and trades its second half with rank cp - 1 - r only."""
    rank_rows = 4096
    for rank, plan in enumerate(_plans([cp_size * rank_rows], cp_size)):
        peers = {dst for dst, rows in enumerate(plan.input_splits) if rows}
        assert peers <= {rank, cp_size - 1 - rank}
        assert plan.input_splits[rank] == rank_rows // 2


def test_chunk_pairing_pairs_early_with_late():
    """Chunk 2r goes to rank r and chunk 2r + 1 to rank cp - 1 - r, so costs sum to 2cp - 1 per rank."""
    cp_size = 4
    ranks = scoring_rank_of_chunk(torch.arange(2 * cp_size), cp_size).tolist()
    by_rank = [[chunk for chunk, rank in enumerate(ranks) if rank == r] for r in range(cp_size)]

    assert by_rank == [[0, 7], [2, 5], [3, 4], [1, 6]]
    assert {sum(chunks) for chunks in by_rank} == {2 * cp_size - 1}


def test_a_pack_of_equal_short_documents_keeps_the_contiguous_layout():
    """Each rank already holds the same mix of positions, so moving rows would only add traffic."""
    assert all(plan is None for plan in _plans([512] * 64, 4))


def test_a_long_document_in_a_pack_is_balanced():
    seq_lens = [24576, 2048, 2048, 2048, 2048]
    rank_rows = sum(seq_lens) // 4
    costs = _costs(seq_lens)
    contiguous = torch.stack([costs[r * rank_rows : (r + 1) * rank_rows].sum() for r in range(4)])
    balanced = torch.stack([costs[plan.scored_positions].sum() for plan in _plans(seq_lens, 4)])

    assert balanced.max() < 0.75 * contiguous.max()


def test_the_stream_must_split_evenly_over_the_ranks():
    with pytest.raises(ValueError, match="split evenly"):
        plan_causal_row_balance((100, 21), cp_rank=0, cp_size=2, device="cpu")


def test_differentiable_inputs_are_rejected():
    """The exchange carries no autograd, so a gradient would be dropped without a word."""
    plan = plan_causal_row_balance((64,), cp_rank=0, cp_size=2, device="cpu")
    with pytest.raises(ValueError, match="no autograd"):
        send_rows_to_scorers([torch.zeros(32, 4, requires_grad=True)], plan, cp_group=None)


def _exchange_worker(rank, world_size, port, seq_lens):
    init_gloo(rank, world_size, port=port)
    try:
        rank_rows = sum(seq_lens) // world_size
        plan = plan_causal_row_balance(seq_lens, cp_rank=rank, cp_size=world_size, device="cpu")
        assert plan is not None
        positions = torch.arange(rank * rank_rows, (rank + 1) * rank_rows)
        # mixed dtypes and trailing shapes, each row tagged with its global position
        tagged = positions.view(-1, 1, 1).expand(-1, 2, 3).to(torch.bfloat16).contiguous()
        weights = positions.double().view(-1, 1) * 0.5

        exchange = send_rows_to_scorers([positions, tagged, weights], plan, dist.group.WORLD)
        received = exchange.wait()

        scored = exchange.plan.scored_positions
        assert torch.equal(received[0], scored)
        assert torch.equal(received[1], scored.view(-1, 1, 1).expand(-1, 2, 3).to(torch.bfloat16))
        assert torch.equal(received[2], scored.double().view(-1, 1) * 0.5)
        # results with rows along dim 1, as the indexer's [batch, rows, topk] picks
        picks = (scored * 10).view(1, -1, 1).expand(2, -1, 4).int()
        back = exchange.return_to_owners(picks, dim=1)
        assert torch.equal(back, (positions * 10).view(1, -1, 1).expand(2, -1, 4).int())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("seq_lens", [(512,), (320, 96, 64, 32)])
def test_exchange_round_trips_over_a_process_group(world_size, seq_lens):
    run_multiprocess(partial(_exchange_worker, seq_lens=seq_lens), world_size=world_size)
