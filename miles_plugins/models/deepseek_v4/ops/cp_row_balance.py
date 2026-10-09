"""Balance causal per-row work across contiguous context-parallel ranks.

Every sequence is cut into 2 * cp chunks and rank r scores chunks 2r and 2 * cp - 1 - 2r, pairing an
early chunk with a late one; for one sequence the exchange is a single pairwise swap. Every rank
derives the same plan from the global sequence lengths. ``send_rows_to_scorers`` starts the
exchange, ``wait()`` hands over the rows at ``scored_positions`` and ``return_to_owners`` sends the
results back; ``LocalRows`` is the same interface without an exchange. Nothing carries autograd.
"""

from dataclasses import dataclass
from functools import lru_cache

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


def scoring_rank_of_chunk(chunk: int, cp_size: int) -> int:
    """Chunk 2r goes to rank r and chunk 2r + 1 to rank cp - 1 - r: an early chunk pairs with a late one.

    This is Miles' zigzag layout with the ranks relabeled so that each rank keeps its own first chunk.
    """
    return chunk // 2 if chunk % 2 == 0 else cp_size - 1 - chunk // 2


# enough for every micro-batch in flight under pipeline parallelism; a miss only rebuilds the plan
@lru_cache(maxsize=32)
def plan_causal_row_balance(
    seq_lens: tuple[int, ...],
    *,
    cp_rank: int,
    cp_size: int,
    device: torch.device | str,
    min_gain: float = 0.1,
) -> RowBalancePlan | None:
    """The balanced exchange for a stream of ``seq_lens`` split contiguously over ``cp_size`` ranks.

    ``seq_lens`` must tile the whole stream, padding included, and the stream must split evenly
    over the ranks. Returns None when balancing would not lower the busiest rank's causal cost
    (``offset + 1`` per row) by ``min_gain``. A rank may be left with no rows to score. Host work is
    proportional to sequences times ranks; the row indices are generated on ``device``. Plans are
    memoized, so every layer and every recompute of a micro-batch shares one.
    """
    if min(seq_lens) < 0:
        raise ValueError(f"sequence lengths must be non-negative, got {min(seq_lens)}")
    total = sum(seq_lens)
    if total % cp_size:
        raise ValueError(f"a stream of {total} rows does not split evenly over {cp_size} ranks")
    rank_rows = total // cp_size

    pieces = _pieces(seq_lens, cp_size, rank_rows)
    if not _worth_balancing(pieces, cp_size, min_gain):
        return None
    return _plan_for_rank(pieces, cp_rank=cp_rank, cp_size=cp_size, rank_rows=rank_rows, device=device)


@dataclass(frozen=True)
class _Piece:
    """Stream rows ``[start, start + rows)``, held by ``owner`` and scored by ``scorer``.

    ``offset`` is where the piece starts inside its own sequence.
    """

    start: int
    rows: int
    offset: int
    owner: int
    scorer: int


def _pieces(seq_lens, cp_size: int, rank_rows: int) -> list[_Piece]:
    """Every sequence's 2 * cp chunks in stream order, cut where a rank's rows end.

    Chunk c of a sequence of n rows holds offsets [ceil(c n / 2cp), ceil((c + 1) n / 2cp)), so it is
    at most half a rank's rows, rounded up, and crosses at most one rank boundary.
    """
    n_chunks = 2 * cp_size
    pieces = []
    seq_start = 0
    for n in seq_lens:
        for chunk in range(n_chunks):
            lo, hi = _ceil_div(chunk * n, n_chunks), _ceil_div((chunk + 1) * n, n_chunks)
            scorer = scoring_rank_of_chunk(chunk, cp_size)
            while lo < hi:
                owner = (seq_start + lo) // rank_rows
                cut = min(hi, (owner + 1) * rank_rows - seq_start)
                pieces.append(_Piece(start=seq_start + lo, rows=cut - lo, offset=lo, owner=owner, scorer=scorer))
                lo = cut
        seq_start += n
    return pieces


def _worth_balancing(pieces: list[_Piece], cp_size: int, min_gain: float) -> bool:
    """Whether scoring each piece on its scorer cuts the busiest rank's causal cost by ``min_gain``."""
    contiguous, balanced = [0] * cp_size, [0] * cp_size
    for piece in pieces:
        end = piece.offset + piece.rows
        cost = (end * (end + 1) - piece.offset * (piece.offset + 1)) // 2  # sum of offset + 1 over the piece
        contiguous[piece.owner] += cost
        balanced[piece.scorer] += cost
    return max(balanced) <= (1 - min_gain) * max(contiguous)


def _plan_for_rank(
    pieces: list[_Piece], *, cp_rank: int, cp_size: int, rank_rows: int, device: torch.device | str
) -> RowBalancePlan:
    # sorted() is stable: rows bound for one rank keep ascending position, the order the receiver expects
    sent = sorted((piece for piece in pieces if piece.owner == cp_rank), key=lambda piece: piece.scorer)
    received = [piece for piece in pieces if piece.scorer == cp_rank]
    input_splits, output_splits = [0] * cp_size, [0] * cp_size
    for piece in sent:
        input_splits[piece.scorer] += piece.rows
    for piece in received:
        output_splits[piece.owner] += piece.rows
    return RowBalancePlan(
        send_rows=_concat_ranges([p.start - cp_rank * rank_rows for p in sent], [p.rows for p in sent], device),
        input_splits=tuple(input_splits),
        output_splits=tuple(output_splits),
        scored_positions=_concat_ranges([p.start for p in received], [p.rows for p in received], device),
    )


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _concat_ranges(starts: list[int], lengths: list[int], device: torch.device | str) -> Tensor:
    """``cat([arange(s, s + n) for s, n in zip(starts, lengths)])``, expanded on ``device``.

    Only the per-range table crosses to the device; the rows are generated there.
    """
    starts_t, lengths_t = torch.tensor([starts, lengths], dtype=torch.int64)
    total = int(lengths_t.sum())
    table = torch.stack([starts_t - (torch.cumsum(lengths_t, 0) - lengths_t), lengths_t])
    if torch.device(device).type == "cuda":
        table = table.pin_memory()  # a copy from pageable memory would wait for the stream to drain
    base, lengths_t = table.to(device, non_blocking=True)
    return base.repeat_interleave(lengths_t, output_size=total) + torch.arange(total, device=device)


class RowExchange:
    """One balanced exchange: rows in flight to this rank's scorer, and the way back for results."""

    def __init__(self, plan: RowBalancePlan, cp_group: dist.ProcessGroup, received: list[Tensor], works: list):
        self.plan = plan
        self._cp_group = cp_group
        self._received = received
        self._works = works

    @property
    def scored_positions(self) -> Tensor:
        return self.plan.scored_positions

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


class LocalRows:
    """``RowExchange`` without the exchange: this rank scores its own rows, already in local order."""

    def __init__(self, tensors: list[Tensor], scored_positions: Tensor):
        self.scored_positions = scored_positions
        self._tensors = tensors

    def wait(self) -> list[Tensor]:
        return self._tensors

    def return_to_owners(self, results: Tensor, *, dim: int = 0) -> Tensor:
        return results


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
