"""Trainer side of TOP's decode KV split (sglang.srt.true_on_policy.attention.decode_kv_split).

Serving decodes a row against its prefix split at fixed key positions (multiples of the split): the shared FA4
forward on each piece, FA's combine across pieces. The trainer computes the rows serving decoded (the GDN history's
step rows) the same way: pieces before the row's own are full slices of its keys (non-causal), the row's own piece is
causal over the document's rows in that piece. Prefill rows keep the ordinary attention. A document's key i is
absolute position i (keys start at position 0: own rows, or a fork chain's / CP prefix rows before them).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from miles_plugins.top.host_copy import device_tensor

# Decode rows per chunk: bounds the FP32 partial buffer (rows x pieces x heads x dim), not the math.
CHUNK_ROWS = 1024

# The split this trainer's architecture uses, bound when its TOP spec is built (bind_split); 0 is one-pass decode.
_SPLIT = 0


def bind_split(model_type: str, stamp: list) -> int:
    """Bind the trainer to serving's decode KV split for the checkpoint's text architecture (same function, same
    override, so the two sides cannot disagree)."""
    global _SPLIT
    from sglang.srt.true_on_policy.attention import decode_kv_split

    _SPLIT = decode_kv_split(model_type)
    stamp.append(("decode_kv_split", "0", str(_SPLIT)))
    return _SPLIT


def bound_split() -> int:
    return _SPLIT


@dataclass(frozen=True)
class _Calls:
    q: torch.Tensor  # packed query rows, grouped per entry
    cu_q: torch.Tensor
    max_q: int
    k: torch.Tensor  # rows of the document-ordered key tensor, grouped per entry
    cu_k: torch.Tensor
    max_k: int
    src: torch.Tensor  # positions in the call's output that feed a decode row's piece
    dst: torch.Tensor  # their flat (decode slot * pieces + piece) index


@dataclass(frozen=True)
class _Chunk:
    rows: torch.Tensor  # packed decode rows of this chunk
    pieces: int
    full: _Calls | None
    own: _Calls | None


@dataclass(frozen=True)
class DecodeSplitPlan:
    rows: torch.Tensor  # every decode row (packed query row)
    chunks: tuple


def _calls(entries, device):
    """entries: [(query rows, key rows, [(index in query rows, dst)])] -> _Calls, or None."""
    if not entries:
        return None
    q, k, cu_q, cu_k, src, dst = [], [], [0], [0], [], []
    for qrows, krows, picks in entries:
        base = cu_q[-1]
        q += qrows
        k.extend(krows)
        cu_q.append(base + len(qrows))
        cu_k.append(cu_k[-1] + len(krows))
        for i, d in picks:
            src.append(base + i)
            dst.append(d)
    t = lambda x, dt=torch.int64: device_tensor(x, device=device, dtype=dt)  # noqa: E731
    return _Calls(
        q=t(q),
        cu_q=t(cu_q, torch.int32),
        max_q=max(b - a for a, b in zip(cu_q, cu_q[1:], strict=False)),
        k=t(k),
        cu_k=t(cu_k, torch.int32),
        max_k=max(b - a for a, b in zip(cu_k, cu_k[1:], strict=False)),
        src=t(src),
        dst=t(dst),
    )


def build_plan(*, decode_rows, cu, doc_starts, cu_k, split: int, device) -> DecodeSplitPlan | None:
    """``decode_rows`` sorted packed rows serving decoded; ``cu`` packed query document bounds (host ints);
    ``doc_starts`` each document's first absolute position; ``cu_k`` bounds of each document's keys in the
    document-ordered key tensor (host ints; ``cu`` when keys are the documents' own rows)."""
    if not split or not decode_rows:
        return None
    import bisect

    doc_of = lambda r: bisect.bisect_right(cu, r) - 1  # noqa: E731
    pos_of = lambda r, d: doc_starts[d] + r - cu[d]  # noqa: E731
    needs = [(r, d, pos_of(r, d)) for r in decode_rows for d in (doc_of(r),)]
    if all(p < split for _, _, p in needs):
        return None  # every decode row's prefix fits one piece: ordinary attention is the serving program
    chunks = []
    for lo in range(0, len(needs), CHUNK_ROWS):
        part = needs[lo : lo + CHUNK_ROWS]
        pieces = max(p // split + 1 for _, _, p in part)
        full, own = {}, {}
        for slot, (r, d, p) in enumerate(part):
            if cu_k[d + 1] - cu_k[d] < p + 1:
                raise ValueError(
                    f"[top] decode split: document {d} has {cu_k[d + 1] - cu_k[d]} keys below position {p}"
                )
            for j in range(p // split):
                e = full.get((d, j))
                if e is None:  # built once per (document, piece): setdefault would build the key range every call
                    e = full[(d, j)] = ([], range(cu_k[d] + j * split, cu_k[d] + (j + 1) * split), [])
                e[2].append((len(e[0]), slot * pieces + j))
                e[0].append(r)
            j = p // split
            if (d, j) not in own:
                # The document's rows in piece j, as the causal queries over keys [j*split, last row + 1).
                first = max(cu[d], cu[d] + j * split - doc_starts[d])
                last = min(cu[d + 1], cu[d] + (j + 1) * split - doc_starts[d])
                qrows = list(range(first, last))
                hi = doc_starts[d] + last - cu[d]
                own[(d, j)] = (qrows, range(cu_k[d] + j * split, cu_k[d] + hi), [])
            e = own[(d, j)]
            e[2].append((r - e[0][0], slot * pieces + j))
        chunks.append(
            _Chunk(
                rows=device_tensor([r for r, _, _ in part], device=device),
                pieces=pieces,
                full=_calls(list(full.values()), device),
                own=_calls(list(own.values()), device),
            )
        )
    return DecodeSplitPlan(rows=device_tensor(decode_rows, device=device), chunks=tuple(chunks))


def _run(calls: _Calls, q, k, v, scale, causal):
    from sglang.srt.true_on_policy.attention import _forward

    out, lse = _forward(
        q=q.index_select(0, calls.q),
        k=k.index_select(0, calls.k),
        v=v.index_select(0, calls.k),
        causal=causal,
        cu_seqlens_q=calls.cu_q,
        cu_seqlens_k=calls.cu_k,
        max_seqlen_q=calls.max_q,
        max_seqlen_k=calls.max_k,
        softmax_scale=scale,
    )[:2]
    return out.index_select(0, calls.src), lse.index_select(1, calls.src)


def apply(plan: DecodeSplitPlan | None, out, *, q, k, v, scale):
    """``out`` [T, Hq, D] the ordinary attention (with grad); returns it with the decode rows' values replaced by the
    split program's, keeping the ordinary attention's gradient (the same function in exact arithmetic). Activation
    recompute runs it too: the layers after it must see the scored values (an MoE router's choice depends on them)."""
    if plan is None:
        return out
    from sglang.srt.true_on_policy.attention import combine_kv_splits

    values = []
    with torch.no_grad():
        qd, kd, vd = q.detach(), k.detach(), v.detach()
        for c in plan.chunks:
            n, h, d = c.rows.numel(), out.shape[1], out.shape[2]
            part_o = torch.zeros(n * c.pieces, h, d, device=out.device, dtype=out.dtype)
            part_l = torch.full((h, n * c.pieces), float("-inf"), device=out.device, dtype=torch.float32)
            for calls, causal in ((c.full, False), (c.own, True)):
                if calls is not None:
                    o, lse = _run(calls, qd, kd, vd, scale, causal)
                    part_o.index_copy_(0, calls.dst, o)
                    part_l.index_copy_(1, calls.dst, lse.float())
            values.append(combine_kv_splits(part_o, part_l, c.pieces))
    return _ReplaceRows.apply(out, plan.rows, torch.cat(values).to(out.dtype))


class _ReplaceRows(torch.autograd.Function):
    """``out`` with ``rows`` set to ``values`` exactly; the gradient passes to ``out`` unchanged (straight through).

    ``out + (values - out)`` is not ``values`` in BF16 when the two differ by more than 2x (small entries near
    cancellation): the subtraction rounds."""

    @staticmethod
    def forward(ctx, out, rows, values):
        return out.detach().clone().index_copy_(0, rows, values)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None
