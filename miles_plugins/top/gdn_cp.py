"""Exact context parallelism for the Qwen GDN TOP trainer.

Every CP rank builds the same packed samples. They are cut into contiguous pieces of about equal estimated cost, only
where serving's chunk prefill can resume: on its 64-row grid, where its kernel carries the FP32 state from one tile to
the next. Rank r's first piece may continue rank r-1's last one: each GDN layer hands the FP32 state and the last 3
convolution inputs forward, and their gradients back. Full attention all-gathers the rotated keys and values; rank r's
rows sit at r * max_rows + local row in that CP order, which also orders the all-gathered final hidden rows the LM head
reads. Under full recompute, each layer's recompute reuses what its first forward received (RecomputeStash) instead of
exchanging again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from miles_plugins.top.host_copy import device_tensor

_HALO = 3


class RecomputeStash:
    """The GDN chain's incoming state and conv halo from a checkpointed layer's training forward, reused by its
    recompute.

    Full recompute runs a layer's forward twice: without grad in the training forward, with grad in backward. Exact
    CP chains GDN serially across ranks, so recomputing would run the forward chain again inside the backward chain
    (a round trip over all ranks per layer). The forward is deterministic, so the first pass keeps what it received
    and the recompute skips the exchange; every rank checkpoints the same layers, so both ends skip together. Only
    the TOP-specific serial exchanges use it: attention's all-gather re-runs as in native CP, keeping the TOP vs
    native comparison like for like."""

    def __init__(self, training: bool):
        self.training = training
        self._kept = {}

    def keeping(self) -> bool:
        """Call outside autograd.Function.forward (which always runs without grad)."""
        return self.training and not torch.is_grad_enabled()

    def take(self, key):
        """(True, value) in a recompute whose first pass kept ``key``; else (False, None)."""
        if not (self.training and torch.is_grad_enabled() and key in self._kept):
            return False, None
        return True, self._kept.pop(key)

    def keep(self, key, value):
        self._kept[key] = value.detach() if value is not None else None


def legal_cut(length: int, target: int, lo: int, hi: int, *, grid: int) -> int | None:
    """A cut c with lo < c < hi where serving's chunk prefill of a ``length``-row document can resume (its ``grid``,
    or the document's end): the largest at or below ``target``, else the smallest above it."""
    if target >= length:
        c = length
    elif target >= 0:
        c = target // grid * grid
    else:
        c = None
    if c is not None and lo < c < hi:
        return c
    # None at or below the target lies above lo: the smallest above both.
    floor = max(target, lo)
    if length <= floor:
        return None
    c = 0 if floor < 0 else min((floor // grid + 1) * grid, length)
    return c if lo < c < hi else None


# Relative cost of one row: projections, MLP and GDN (1) plus its full-attention keys (per key). Arbitrary from FLOPs
# (16 of 64 layers attend; x2 for attention's lower efficiency and the decode split's recompute); it only moves exact
# cut points, so a rough value is safe.
_ATTN_COST_PER_KEY = 1.5e-5


def _row_costs(lengths):
    """Per document, cumulative cost up to each row."""
    import itertools

    return [[0.0, *itertools.accumulate(1.0 + _ATTN_COST_PER_KEY * (i + 1) for i in range(n))] for n in lengths]


def _latest_starts(length: int, count: int, *, grid: int) -> list[int]:
    """The k-th entry is the last row from which [row, length) still splits into k + 1 legal pieces of at least 3
    rows (taken greedily from the end, which fits the most pieces); at most ``count`` entries."""
    starts, c = [], length
    while len(starts) < count:
        c = legal_cut(length, c - _HALO, -1, c - _HALO + 1, grid=grid)
        if c is None:
            break
        starts.append(c)
    return starts


def partition(lengths, cp_size: int, *, grid: int) -> list[list[tuple[int, int, int]]]:
    """Per rank, its pieces (document, first row, end row): documents cut near equal shares of estimated cost (rows
    plus attention keys; every piece at least 3 rows, so a piece's last conv inputs are its own). A cut always leaves
    the later ranks enough legal pieces, so every rank gets one whenever the documents allow it."""
    import bisect

    cum = _row_costs(lengths)
    total = sum(c[-1] for c in cum)
    bounds = [k * total / cp_size for k in range(1, cp_size)]
    # Pieces the documents after each one can hold (up to cp_size).
    later, held = [0] * len(lengths), 0
    for d in reversed(range(len(lengths))):
        later[d] = held
        held += len(_latest_starts(lengths[d], cp_size, grid=grid))
    pieces: list[list[tuple[int, int, int]]] = [[] for _ in range(cp_size)]
    rank, done = 0, 0.0
    for d, n in enumerate(lengths):
        a, c = 0, cum[d]
        while a < n:
            # Pieces the rest of this document must hold after this rank's, for the later ranks to get one each.
            need = cp_size - 1 - rank - later[d]
            if rank < cp_size - 1 and (done + c[-1] - c[a] > bounds[rank] or need > 0):
                hi = n - _HALO + 1
                if need > 0:
                    starts = _latest_starts(n, need, grid=grid)
                    hi = min(hi, starts[-1] + 1) if len(starts) == need else a
                target = bisect.bisect_right(c, bounds[rank] - done + c[a]) - 1  # last row within the share
                cut = legal_cut(n, min(max(target, a), hi - 1), a + _HALO - 1, hi, grid=grid)
                if cut is not None and cut > a:
                    pieces[rank].append((d, a, cut))
                    done += c[cut] - c[a]
                    a = cut
                    rank += 1
                    continue
                if pieces[rank]:
                    rank += 1
                    continue
                # No legal cut in this document's rest: an empty rank takes all of it rather than being skipped.
            pieces[rank].append((d, a, n))
            done += c[-1] - c[a]
            a = n
    if any(not p for p in pieces):
        raise ValueError(f"[top] GDN CP: {sum(lengths)} rows cannot give each of {cp_size} ranks a piece")
    return pieces


@dataclass
class GDNCPLayout:
    """This rank's view of the CP split (host ints unless noted)."""

    group: object
    rank: int
    size: int
    prev: int | None  # global rank sending this rank's incoming piece its state, or None
    next: int | None  # global rank receiving this rank's outgoing piece's state, or None
    incoming: int | None  # local document continuing the previous rank's last piece
    outgoing: int | None  # local document continued by the next rank's first piece
    max_rows: int  # rows of the largest rank's slice; CP order = rank * max_rows + local row
    local_rows: int
    tail_end: int  # local end row of the outgoing piece (its last 3 conv inputs go forward)
    keys: tuple  # (CP-order key rows [K] int64 tensor, cu_seqlens_k int32 tensor, max_seqlen_k)
    conv_init: tuple  # per local document: None or 3 indices into cat(halo, local inputs)
    output_rows: torch.Tensor  # CP-order rows of this rank's slice of the original stream
    stash: RecomputeStash = field(default_factory=lambda: RecomputeStash(False))


def build_layout(*, lengths, sample_rows: int, local_original_rows: int, device, group, grid: int) -> tuple:
    """This rank's pieces (document, first row, end row) of the packed documents ``lengths`` and its layout. The
    first ``sample_rows`` document rows are the original stream's rows in order; its padding rows carry no loss."""
    import bisect

    size, rank = dist.get_world_size(group), dist.get_rank(group)
    pieces = partition(lengths, size, grid=grid)
    max_rows = max(sum(b - a for _, a, b in p) for p in pieces)
    where = {}  # per document, (first row, end row, CP-order row of document row 0 for that piece)
    for r, ps in enumerate(pieces):
        off = 0
        for d, a, b in ps:
            where.setdefault(d, []).append((a, b, r * max_rows + off - a))
            off += b - a

    def cp_rows(d, a, b):
        out = []
        for pa, pb, base in where[d]:
            lo, hi = max(a, pa), min(b, pb)
            if lo < hi:
                out += range(base + lo, base + hi)
        return out

    mine = pieces[rank]
    incoming = next((i for i, (_, a, _) in enumerate(mine) if a > 0), None)
    outgoing = next((i for i, (d, _, b) in enumerate(mine) if b < lengths[d]), None)
    ranks = dist.get_process_group_ranks(group)
    local_start, off = [], 0
    for _, a, b in mine:
        local_start.append(off)
        off += b - a
    index, kv_lengths = [], []
    for i, (d, a, b) in enumerate(mine):
        own = rank * max_rows + local_start[i]
        index += cp_rows(d, 0, a) + list(range(own, own + b - a))
        kv_lengths.append(b)  # the document's rows before the piece, then its own
    cu_k = [0]
    for n in kv_lengths:
        cu_k.append(cu_k[-1] + n)
    # Original rows of this rank's slice of the (padded) original stream -> CP order; its padding rows carry no
    # loss, and document 0's row 0 serves.
    doc_start = [0]
    for n in lengths:
        doc_start.append(doc_start[-1] + n)
    pad_row = cp_rows(0, 0, 1)[0]
    owner_of = []
    for g in range(rank * local_original_rows, (rank + 1) * local_original_rows):
        if g >= sample_rows:
            owner_of.append(pad_row)
            continue
        d = bisect.bisect_right(doc_start, g) - 1
        owner_of.append(cp_rows(d, g - doc_start[d], g - doc_start[d] + 1)[0])
    layout = GDNCPLayout(
        group=group,
        rank=rank,
        size=size,
        prev=ranks[rank - 1] if incoming is not None else None,
        next=ranks[rank + 1] if outgoing is not None else None,
        incoming=incoming,
        outgoing=outgoing,
        max_rows=max_rows,
        local_rows=off,
        tail_end=local_start[outgoing] + mine[outgoing][2] - mine[outgoing][1] if outgoing is not None else 0,
        keys=(
            device_tensor(index, device=device),
            device_tensor(cu_k, device=device, dtype=torch.int32),
            max(kv_lengths),
        ),
        conv_init=tuple(list(range(_HALO)) if a > 0 else None for _, a, _ in mine),
        output_rows=device_tensor(owner_of, device=device),
    )
    return mine, layout


def post_p2p(send=None, to=None, recv=None, frm=None, group=None) -> list:
    """Post a send and/or a receive together without waiting; ``wait_p2p`` the returned requests before reading
    ``recv`` (under NCCL the wait orders the current stream after the transfer; the host does not block)."""
    ops = []
    if send is not None:
        ops.append(dist.P2POp(dist.isend, send.contiguous(), to, group))
    if recv is not None:
        ops.append(dist.P2POp(dist.irecv, recv, frm, group))
    return dist.batch_isend_irecv(ops) if ops else []


def wait_p2p(reqs) -> None:
    for req in reqs:
        req.wait()


def _p2p(send=None, to=None, recv=None, frm=None, group=None):
    wait_p2p(post_p2p(send=send, to=to, recv=recv, frm=frm, group=group))


class _Halo(torch.autograd.Function):
    """Send the outgoing piece's last 3 rows of ``x`` forward; receive the incoming piece's 3 previous rows."""

    @staticmethod
    def forward(ctx, x, layout: GDNCPLayout, tail_end: int, cached):
        ctx.layout, ctx.tail_end, ctx.shape = layout, tail_end, x.shape
        if cached is not None:
            return cached.clone()
        send = x[tail_end - _HALO : tail_end] if layout.next is not None else None
        halo = x.new_empty((_HALO, x.shape[1])) if layout.prev is not None else x.new_empty((0, x.shape[1]))
        _p2p(
            send=send,
            to=layout.next,
            recv=halo if layout.prev is not None else None,
            frm=layout.prev,
            group=layout.group,
        )
        return halo

    @staticmethod
    def backward(ctx, d_halo):
        layout = ctx.layout
        dx = d_halo.new_zeros(ctx.shape)
        recv = dx.new_empty((_HALO, ctx.shape[1])) if layout.next is not None else None
        _p2p(
            send=d_halo if layout.prev is not None else None,
            to=layout.prev,
            recv=recv,
            frm=layout.next,
            group=layout.group,
        )
        if recv is not None:
            dx[ctx.tail_end - _HALO : ctx.tail_end] = recv
        return dx, None, None, None


def exchange_halo(x, layout: GDNCPLayout, key=None):
    """[3, D] inputs before this rank's incoming piece (or [0, D]); differentiable through the exchange.
    ``key`` (the calling layer) lets a recompute reuse the first pass's halo."""
    hit, cached = layout.stash.take((key, "halo")) if key is not None else (False, None)
    keeping = key is not None and layout.stash.keeping()
    out = _Halo.apply(x, layout, layout.tail_end, cached if hit else None)
    if keeping:
        layout.stash.keep((key, "halo"), out)
    return out


def all_gather_rows(x, layout: GDNCPLayout):
    """[local rows, ...] -> [size * max_rows, ...] in CP order (autograd: gradients return to their rank)."""
    from torch.distributed.nn.functional import all_gather

    pad = layout.max_rows - x.shape[0]
    if pad:
        x = torch.cat((x, x.new_zeros((pad, *x.shape[1:]))))
    return torch.cat(all_gather(x.contiguous(), group=layout.group))
