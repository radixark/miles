"""Balance causal per-row work across contiguous context-parallel ranks.

A causal indexer (the DSA/NSA lightning indexer, DeepSeek-V4's CSA indexer) scores every key
before each query, so a row's cost grows with its position inside its own sequence. Under
contiguous CP, rank r holds rows [r * L, (r + 1) * L) of the packed stream; for one long sequence
the last rank then does about (2 * cp - 1) / cp of the mean work, and the others wait for it.

Balancing moves only the rows being scored, never the layout. Every sequence is cut into
2 * cp near-equal chunks; rank r scores chunks 2r and 2 * cp - 1 - 2r of each one, an early chunk
paired with a late one. For a stream holding one sequence, rank r keeps its own first half and
swaps its second half with rank cp - 1 - r, so the exchange is a single pairwise swap. Every rank
derives the same plan from the global sequence lengths, so no metadata travels; a plan is used
only when it lowers the busiest rank's work by ``min_gain``, since a pack of many short documents
is already balanced under contiguous CP and moving it would only add traffic.

A caller starts the exchange with ``send_rows_to_scorers``, which returns at once so it can overlap
other work, scores the rows at ``plan.scored_positions`` once ``RowExchange.wait()`` hands them
over, and sends the results back with ``RowExchange.return_to_owners``. Each input travels in its
own all-to-all straight into the buffer the scorer reads, so a row is copied once on the way out
and never on the way in. The exchange carries no autograd: it is for work whose output is
selection indices, and it rejects inputs that require grad.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(frozen=True)
class RowBalancePlan:
    """This rank's side of a balanced exchange; every field is identical in meaning on all ranks.

    Attributes:
        send_rows: local row indices in send order (grouped by scoring rank, ascending position).
        input_splits: rows sent to each rank.
        output_splits: rows received from each rank.
        scored_positions: global stream positions of the rows this rank scores, in received order.
    """

    send_rows: Tensor
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    scored_positions: Tensor

    @property
    def num_scored(self) -> int:
        return sum(self.output_splits)


def scoring_rank_of_chunk(chunk: Tensor, cp_size: int) -> Tensor:
    """Chunk 2r goes to rank r and chunk 2r + 1 to rank cp - 1 - r: chunk costs pair up evenly."""
    return torch.where(chunk % 2 == 0, chunk // 2, cp_size - 1 - chunk // 2)


def plan_causal_row_balance(
    seq_lens: tuple[int, ...] | list[int],
    *,
    cp_rank: int,
    cp_size: int,
    device: torch.device | str,
    min_gain: float = 0.1,
) -> RowBalancePlan | None:
    """The balanced exchange for a stream of ``seq_lens`` split contiguously over ``cp_size`` ranks.

    ``seq_lens`` must tile the whole stream, padding included, and the stream must split evenly
    over the ranks. Returns None when balancing would not lower the busiest rank's causal cost
    (``offset + 1`` per row) by ``min_gain``. Host work is proportional to sequences times ranks;
    the row indices are generated on ``device``.
    """
    if min(seq_lens) < 0:
        raise ValueError(f"sequence lengths must be non-negative, got {min(seq_lens)}")
    total = sum(seq_lens)
    if total % cp_size:
        raise ValueError(f"a stream of {total} rows does not split evenly over {cp_size} ranks")
    rank_rows = total // cp_size

    pieces = _split_at_rank_boundaries(_chunks(seq_lens, cp_size), rank_rows)
    if not _worth_balancing(pieces, cp_size, min_gain):
        return None
    return _plan_for_rank(pieces, cp_rank=cp_rank, cp_size=cp_size, rank_rows=rank_rows, device=device)


@dataclass(frozen=True)
class _Intervals:
    """Row intervals ``[start, start + rows)`` of the stream, in ascending position.

    ``offset`` is where each interval starts inside its own sequence, ``scorer`` the rank that
    scores it, and ``owner`` the rank that holds it (set once intervals are cut at rank boundaries).
    """

    start: Tensor
    rows: Tensor
    offset: Tensor
    scorer: Tensor
    owner: Tensor | None = None


def _chunks(seq_lens, cp_size: int) -> _Intervals:
    """Every sequence cut into 2 * cp near-equal chunks, each with its scoring rank; empty ones dropped.

    Chunk c of a sequence of n rows holds offsets [ceil(c n / 2cp), ceil((c + 1) n / 2cp)).
    """
    n_chunks = 2 * cp_size
    lens = torch.tensor(seq_lens, dtype=torch.int64)
    edges = (torch.arange(n_chunks + 1) * lens[:, None] + n_chunks - 1) // n_chunks
    lo, hi = edges[:, :-1].reshape(-1), edges[:, 1:].reshape(-1)
    seq_start = (torch.cumsum(lens, 0) - lens).repeat_interleave(n_chunks)
    scorer = scoring_rank_of_chunk(torch.arange(n_chunks).repeat(len(seq_lens)), cp_size)
    keep = hi > lo
    return _Intervals(start=(seq_start + lo)[keep], rows=(hi - lo)[keep], offset=lo[keep], scorer=scorer[keep])


def _split_at_rank_boundaries(chunks: _Intervals, rank_rows: int) -> _Intervals:
    """Cut every chunk where a rank's rows end, so each piece has one owner and one scorer."""
    first_owner = chunks.start // rank_rows
    n_pieces = (chunks.start + chunks.rows - 1) // rank_rows - first_owner + 1
    chunk = torch.repeat_interleave(n_pieces)
    owner = first_owner[chunk] + _concat_ranges(torch.zeros_like(n_pieces), n_pieces)
    start = torch.maximum(chunks.start[chunk], owner * rank_rows)
    end = torch.minimum(chunks.start[chunk] + chunks.rows[chunk], (owner + 1) * rank_rows)
    return _Intervals(
        start=start,
        rows=end - start,
        offset=chunks.offset[chunk] + (start - chunks.start[chunk]),
        scorer=chunks.scorer[chunk],
        owner=owner,
    )


def _worth_balancing(pieces: _Intervals, cp_size: int, min_gain: float) -> bool:
    """Whether scoring on ``scorer`` cuts the busiest rank's causal cost by ``min_gain``."""
    lo, hi = pieces.offset, pieces.offset + pieces.rows
    cost = (hi * (hi + 1) - lo * (lo + 1)) // 2  # sum of offset + 1 over each piece
    contiguous = torch.zeros(cp_size, dtype=torch.int64).index_add_(0, pieces.owner, cost).max().item()
    balanced = torch.zeros(cp_size, dtype=torch.int64).index_add_(0, pieces.scorer, cost).max().item()
    return balanced <= (1 - min_gain) * contiguous


def _plan_for_rank(
    pieces: _Intervals, *, cp_rank: int, cp_size: int, rank_rows: int, device: torch.device | str
) -> RowBalancePlan:
    sent = pieces.owner == cp_rank
    # stable: rows bound for one rank keep ascending position, the order the receiver expects
    order = torch.argsort(pieces.scorer[sent], stable=True)
    received = pieces.scorer == cp_rank
    return RowBalancePlan(
        send_rows=_concat_ranges(pieces.start[sent][order] - cp_rank * rank_rows, pieces.rows[sent][order], device),
        input_splits=_rows_per_rank(pieces.scorer[sent], pieces.rows[sent], cp_size),
        output_splits=_rows_per_rank(pieces.owner[received], pieces.rows[received], cp_size),
        scored_positions=_concat_ranges(pieces.start[received], pieces.rows[received], device),
    )


def _concat_ranges(starts: Tensor, lengths: Tensor, device: torch.device | str = "cpu") -> Tensor:
    """``cat([arange(s, s + n) for s, n in zip(starts, lengths)])``, expanded on ``device``.

    Only the per-range table crosses to the device; the rows are generated there.
    """
    total = int(lengths.sum())
    base, lengths = torch.stack([starts - (torch.cumsum(lengths, 0) - lengths), lengths]).to(device)
    return base.repeat_interleave(lengths, output_size=total) + torch.arange(total, device=device)


def _rows_per_rank(rank: Tensor, rows: Tensor, cp_size: int) -> tuple[int, ...]:
    return tuple(torch.zeros(cp_size, dtype=torch.int64).index_add_(0, rank, rows).tolist())


class RowExchange:
    """One balanced exchange: rows in flight to this rank's scorer, and the way back for results."""

    def __init__(self, plan: RowBalancePlan, cp_group: dist.ProcessGroup, received: list[Tensor], works: list):
        self.plan = plan
        self._cp_group = cp_group
        self._received = received
        self._works = works

    def wait(self) -> list[Tensor]:
        """The sent tensors' rows at ``plan.scored_positions``, in that order."""
        for work in self._works:
            work.wait()
        self._works = []
        return self._received

    def return_to_owners(self, results: Tensor, *, dim: int = 0) -> Tensor:
        """Per-row results for the scored rows (along ``dim``) back to the local row order."""
        if results.requires_grad:
            raise ValueError("the row exchange carries no autograd; return detached results")
        self.wait()
        rows = results.movedim(dim, 0).contiguous()
        assert rows.shape[0] == self.plan.num_scored, f"{rows.shape[0]} results for {self.plan.num_scored} rows"
        received = rows.new_empty((self.plan.send_rows.numel(), *rows.shape[1:]))
        dist.all_to_all_single(
            received,
            rows,
            output_split_sizes=list(self.plan.input_splits),
            input_split_sizes=list(self.plan.output_splits),
            group=self._cp_group,
        )
        # rows come back grouped by scoring rank in ascending position, i.e. in send order
        local = torch.empty_like(received)
        local[self.plan.send_rows] = received
        return local.movedim(0, dim).contiguous()


def send_rows_to_scorers(tensors: list[Tensor], plan: RowBalancePlan, cp_group: dist.ProcessGroup) -> RowExchange:
    """Start sending each tensor's local rows (dim 0) to the ranks that score them."""
    for rows in tensors:
        if rows.requires_grad:
            raise ValueError("the row exchange carries no autograd; send detached tensors")
        assert rows.shape[0] == plan.send_rows.numel(), "every tensor needs one row per local position"
    received, works = [], []
    for rows in tensors:
        buffer = rows.new_empty((plan.num_scored, *rows.shape[1:]))
        works.append(
            dist.all_to_all_single(
                buffer,
                rows.index_select(0, plan.send_rows),
                output_split_sizes=list(plan.output_splits),
                input_split_sizes=list(plan.input_splits),
                group=cp_group,
                async_op=True,
            )
        )
        received.append(buffer)
    return RowExchange(plan, cp_group, received, works)
