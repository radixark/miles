"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Chunked KDA training backward (SM100a / SM103a).

Deterministic backward pass for the chunked KDA training algorithm
(``chunk_kda`` with in-kernel Q/K L2 normalization, the lower-bound safe gate,
sigmoid beta and ``dt_bias``; chunk size 64, K = V = 128).  The kernels
reproduce the reference chunked dataflow stage by stage -- gate prefix scan
and WY inputs, WY recompute, forward state recurrence, dAv, adjoint state
recurrence, fused dq/dk/dg, intra-chunk backward, and the gate / norm
epilogue -- with fp32 inter-chunk state carriers and fixed-order reductions
(no atomics), so repeated backward passes on identical inputs are
bit-identical.

The forward pass is not part of this module: :func:`chunk_kda_backward`
consumes the tensors the reference forward saves for backward (normalized
q/k with their inverse norms, sigmoid(beta), and the per-chunk ``Aqk`` /
``Akk`` matrices).

Any sequence length and any ``cu_seqlens`` packing are accepted: the stages
run on a chunk-aligned internal row layout (:class:`ChunkLayout`) in which
every sequence starts on a 64-token chunk boundary, the first stage gathers
the token rows into it and the last stages scatter the gradients back, so no
host-side repacking or padding copies are needed.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

from ._jit import device_arch, kernel

CHUNK = 64
HEAD_DIM = 128
PAIR_ROWS = 2 * CHUNK


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def _contiguous(name: str, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if t.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {t.dtype}")
    return t.contiguous()


def _as_dtype(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``t.to(dtype)``, skipping the dispatcher round trip when ``t`` already has ``dtype``
    (``Tensor.to`` returns ``t`` itself in that case)."""
    return t if t.dtype == dtype else t.to(dtype)


def _rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    return x.reshape(rows, *x.shape[2:])


# --------------------------------------------------------------------------- #
# chunk-aligned internal layout
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChunkLayout:
    """Row layout the stages run on: every sequence starts on a 64-token chunk boundary.

    The chunked algorithm chunks each sequence from its own first token (only a sequence's last
    chunk can be partial), so a sequence of ``L`` tokens owns ``ceil(L / 64)`` chunks.  Internally
    every chunk is a full 64-row tile; the tail rows of a partial chunk are zero pads that the
    stages gate off (they never touch a real token's gradient, and their own gradients are exact
    zeros or dropped).  A trailing all-zero chunk keeps the count even for the pair-tiled stages.

    ``chunk_bos[c]``: first input row of chunk ``c``; ``chunk_len[c]``: its valid tokens (0 for the
    trailing pad chunk); ``seq_chunk_start[n]``: first chunk of sequence ``n`` (``N + 1`` entries);
    ``internal_rows[r]``: internal row of input row ``r``.

    Layouts are immutable and cached per packing, so the derived counts below are computed once per
    layout rather than on every access from the backward's hot path.
    """

    lengths: tuple[int, ...]
    offsets: tuple[int, ...]
    rows_external: int
    seq_chunk_start_cpu: tuple[int, ...]
    chunk_bos: torch.Tensor
    chunk_len: torch.Tensor
    seq_chunk_start: torch.Tensor
    internal_rows: torch.Tensor

    @cached_property
    def num_sequences(self) -> int:
        return len(self.lengths)

    @cached_property
    def num_chunks(self) -> int:
        return self.seq_chunk_start_cpu[-1]

    @cached_property
    def num_chunks_padded(self) -> int:
        return -(-self.num_chunks // 2) * 2

    @cached_property
    def rows(self) -> int:
        return self.num_chunks_padded * CHUNK

    @cached_property
    def has_pad_chunk(self) -> bool:
        return self.num_chunks_padded != self.num_chunks

    @cached_property
    def identity(self) -> bool:
        """Input rows and internal rows coincide (64-multiple lengths laid out back to back, even chunk count)."""
        return (
            self.rows == self.rows_external
            and not self.has_pad_chunk
            and all(length % CHUNK == 0 for length in self.lengths)
            and all(offset == sum(self.lengths[:i]) for i, offset in enumerate(self.offsets))
        )

    def scatter(self, x: torch.Tensor) -> torch.Tensor:
        """Internal ``[rows, ...]`` -> input ``[rows_external, ...]`` (pads dropped)."""
        return x if self.identity else x.index_select(0, self.internal_rows)


class _LRUCache:
    """Bounded dict that evicts its least recently used entry (a hit refreshes the entry)."""

    def __init__(self, limit: int):
        self.limit = limit
        self._data: OrderedDict = OrderedDict()

    def get(self, key):
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
        return value

    def put(self, key, value) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.limit:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()


_TABLE_ALIGN = 64  # int32 elements: every int32 table view starts 256-byte aligned inside the shared buffer
_ROW_ALIGN = 32  # int64 elements: the int32 tables start 256-byte aligned behind the int64 rows


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _host_tables(lengths: tuple[int, ...], offsets: tuple[int, ...], rows_external: int, rows_out, bos_out, clen_out):
    """Fill the layout tables straight into their (pinned) staging views, vectorised: ``rows_out``
    (``rows_external`` int64, the internal row of every input row), ``bos_out`` / ``clen_out`` (``nc``
    int32 each). Returns ``starts`` (``N + 1`` Python ints). Per-sequence scalars stay in Python (a
    pack has a few sequences), only the per-chunk and per-row arrays go through numpy, and the per-row
    array is written with a single ``np.add(..., out=)`` when the sequences lie back to back (packed
    and fixed-length input: offsets = exclusive cumsum of the lengths)."""
    n = len(lengths)
    nch = [(length + (CHUNK - 1)) // CHUNK for length in lengths]
    starts = [0] * (n + 1)
    excl = [0] * n
    for s in range(n):
        starts[s + 1] = starts[s] + nch[s]
        if s:
            excl[s] = excl[s - 1] + lengths[s - 1]
    nc = starts[-1]
    # per chunk: index within its sequence (by repeat), first input row, valid tokens
    i = np.arange(nc, dtype=np.int64) - np.repeat(np.asarray(starts[:-1], dtype=np.int64), nch)
    bos_out[:] = np.repeat(np.asarray(offsets, dtype=np.int64), nch) + i * CHUNK
    clen_out[:] = np.minimum(CHUNK, np.repeat(np.asarray(lengths, dtype=np.int64), nch) - i * CHUNK)
    # per input row: internal row = row + (first internal row of its sequence - its exclusive cumsum)
    shift = np.asarray([starts[s] * CHUNK - excl[s] for s in range(n)], dtype=np.int64)
    if tuple(excl) == offsets:
        np.add(np.arange(rows_external, dtype=np.int64), np.repeat(shift, lengths), out=rows_out)
    else:
        j = np.arange(rows_external, dtype=np.int64) - np.repeat(np.asarray(excl, dtype=np.int64), lengths)
        rows_out[np.repeat(np.asarray(offsets, dtype=np.int64), lengths) + j] = np.repeat(shift + np.asarray(excl, dtype=np.int64), lengths) + j
    return starts


def _build_layout(lengths, offsets, rows_external: int, device) -> ChunkLayout:
    """Build the layout of ``lengths`` / ``offsets`` on ``device``.

    Host cost is a handful of vectorised numpy ops writing straight into ONE pinned staging buffer
    (int64: ``internal_rows`` first, then ``chunk_bos``, ``chunk_len`` and ``seq_chunk_start`` as
    256-byte-aligned int32 views of its tail); the device side is exactly one device allocation and one
    ``non_blocking`` host-to-device copy, so building a layout never drains the stream (the U0-1 gate
    allows up to two such copies). The staging buffer comes from torch's caching pinned-memory
    allocator, which hands it out again once the copy has retired. On a CPU device the buffer is
    pageable and the copy is a plain copy (tests).
    """
    lengths = tuple(int(x) for x in lengths)
    offsets = tuple(int(x) for x in offsets)
    _check(
        len(lengths) == len(offsets) and bool(lengths),
        "sequence lengths and offsets must be non-empty and equally long",
    )
    _check(all(length > 0 for length in lengths), "every packed sequence must have at least one token")
    _check(sum(lengths) == rows_external, "the packed sequences must cover the token rows exactly once")
    device = torch.device(device) if not isinstance(device, torch.device) else device
    pinned = device.type == "cuda"

    n = len(lengths)
    nc = sum((length + (CHUNK - 1)) // CHUNK for length in lengths)
    nc_pad = _round_up(nc, 2)
    table = _round_up(nc_pad, _TABLE_ALIGN)
    start_table = _round_up(n + 1, _TABLE_ALIGN)
    rows64 = _round_up(rows_external, _ROW_ALIGN)
    host = torch.empty(rows64 + (2 * table + start_table) // 2, dtype=torch.int64, pin_memory=pinned)
    a32 = host[rows64:].view(torch.int32).numpy()
    starts = _host_tables(lengths, offsets, rows_external, host.numpy()[:rows_external], a32[:nc], a32[table : table + nc])
    if nc_pad != nc:  # trailing pad chunk: no rows, zero length
        a32[nc] = 0
        a32[table + nc] = 0
    a32[2 * table : 2 * table + n + 1] = starts

    dev = torch.empty_like(host, device=device)
    dev.copy_(host, non_blocking=pinned)
    tables = dev[rows64:].view(torch.int32)
    return ChunkLayout(
        lengths=lengths,
        offsets=offsets,
        rows_external=int(rows_external),
        seq_chunk_start_cpu=tuple(starts),
        chunk_bos=tables[:nc_pad],
        chunk_len=tables[table : table + nc_pad],
        seq_chunk_start=tables[2 * table : 2 * table + n + 1],
        internal_rows=dev[:rows_external],
    )


_LAYOUT_CACHE_LIMIT = 256
_LAYOUT_CACHE: _LRUCache = _LRUCache(_LAYOUT_CACHE_LIMIT)


def host_boundaries(cu_seqlens_cpu) -> tuple[int, ...]:
    """The host copy of ``cu_seqlens`` as a tuple of ints: from a CPU integer tensor (the canonical
    ``cu_seqlens_cpu``) or any sequence of ints. A device tensor is rejected rather than copied."""
    if isinstance(cu_seqlens_cpu, torch.Tensor):
        _check(cu_seqlens_cpu.device.type == "cpu", "cu_seqlens_cpu must be a host (CPU) tensor")
        return tuple(cu_seqlens_cpu.tolist())
    return tuple(int(x) for x in cu_seqlens_cpu)


def _device_key(device):
    device = torch.device(device) if not isinstance(device, torch.device) else device
    index = device.index
    if index is None and device.type == "cuda":
        index = torch.cuda.current_device()
    return (device.type, index)


def chunk_layout(
    batch: int,
    seq_len: int,
    cu_seqlens: torch.Tensor | None,
    device,
    *,
    cu_seqlens_cpu: torch.Tensor | Sequence[int] | None = None,
) -> ChunkLayout:
    """Layout of a ``[batch, seq_len]`` input, or of ``cu_seqlens``-packed sequences (``batch == 1``).

    Layouts are cached per (host boundaries, device) in an LRU of ``_LAYOUT_CACHE_LIMIT`` entries: a
    training run repeats a few packing shapes, and a hit costs no host-to-device copy, no stream
    synchronisation and no device read. The key is the host tuple of boundaries, so every object
    carrying the same boundaries (a new ``PackedSeqParams`` of the next micro-batch, a tuple from a
    direct kernel-level call) hits the same entry.

    ``cu_seqlens_cpu`` is the host copy of ``cu_seqlens``: a CPU int64 tensor (the canonical form the
    layer passes) or any sequence of ints. With it the lookup never touches ``cu_seqlens``. Without it
    the boundaries are read from the device (``cu_seqlens.tolist()``: one device-to-host copy that waits
    for the stream) -- a fallback for direct callers, not the training path.
    """
    dev_key = _device_key(device)
    if cu_seqlens is None:
        key = ("fixed", int(batch), int(seq_len), dev_key)
        cu = None
    else:
        _check(batch == 1, f"packed input must have batch 1, got {batch}")
        if cu_seqlens_cpu is not None:
            cu = host_boundaries(cu_seqlens_cpu)
        else:
            cu = tuple(int(x) for x in cu_seqlens.tolist())
        key = ("packed", cu, dev_key)
    layout = _LAYOUT_CACHE.get(key)
    if layout is None:
        if cu is None:
            lengths, offsets = [seq_len] * batch, [b * seq_len for b in range(batch)]
        else:
            _check(
                len(cu) >= 2 and cu[0] == 0 and cu[-1] == seq_len,
                f"cu_seqlens must run from 0 to the token count {seq_len}",
            )
            lengths, offsets = [b - a for a, b in zip(cu[:-1], cu[1:], strict=True)], cu[:-1]
        layout = _build_layout(lengths, offsets, batch * seq_len, device)
        _LAYOUT_CACHE.put(key, layout)
    return layout


# --------------------------------------------------------------------------- #
# stage launchers (thin host wrappers over the generated kernels)
# --------------------------------------------------------------------------- #
def _launch_prep(arch, g_raw, q_norm, k_norm, v, beta, A_log, dt_bias, aqk, akk, do, *, layout, lower_bound):
    rows_ext, hv, kd = g_raw.shape
    h = q_norm.shape[1]
    _check(
        kd == HEAD_DIM
        and hv % h == 0
        and rows_ext == layout.rows_external
        and aqk.shape == (rows_ext, hv, CHUNK)
        and akk.shape == (rows_ext, hv, CHUNK),
        "prep: expected [R,HV,128] g/v/do, [R,H,128] q/k, [R,HV,64] Aqk/Akk over the layout's token rows",
    )
    dev, bf = g_raw.device, torch.bfloat16
    rows = layout.rows
    copy_inputs = 0 if layout.identity else 1
    # The bf16 [rows, HV, 128] stage outputs are carved from one allocation (each plane is a
    # contiguous view with its own base address): one allocator round trip instead of six / eight.
    planes = torch.empty(6 + 2 * copy_inputs, rows, hv, HEAD_DIM, dtype=bf, device=dev).unbind(0)
    out = {"gk": torch.empty(rows, hv, HEAD_DIM, dtype=torch.float32, device=dev)}
    out["vb"], out["kb"], out["qg"], out["kg"], out["ke"], out["qe"] = planes[:6]
    out["aqk_tril"] = torch.empty(rows, hv, CHUNK, dtype=bf, device=dev)
    if copy_inputs:
        out["akk"] = torch.empty(rows, hv, CHUNK, dtype=bf, device=dev)
        out["do"], out["v"] = planes[6:]
        out["beta"] = torch.empty(rows, hv, dtype=torch.float32, device=dev)
    else:
        out["akk"], out["do"], out["v"], out["beta"] = akk, do, v, beta
    kernel("prep", arch).launch(
        grid=(layout.num_chunks_padded, hv, 1),
        g_raw=g_raw,
        q_norm=q_norm,
        k_norm=k_norm,
        v=v,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        aqk=aqk,
        akk=akk,
        do=do,
        chunk_bos=layout.chunk_bos,
        chunk_len=layout.chunk_len,
        gk_out=out["gk"],
        vb_out=out["vb"],
        kb_out=out["kb"],
        qg_out=out["qg"],
        kg_out=out["kg"],
        ke_out=out["ke"],
        qe_out=out["qe"],
        aqk_tril=out["aqk_tril"],
        akk_int=out["akk"],
        do_int=out["do"],
        v_int=out["v"],
        beta_int=out["beta"],
        num_qk_heads=h,
        num_heads=hv,
        group=hv // h,
        copy_inputs=copy_inputs,
        lower_bound=float(lower_bound),
    )
    return out


def _launch_wy(arch, akk, vb, kb):
    rows, hv, _ = vb.shape
    _check(rows % PAIR_ROWS == 0, "wy: R % 128 == 0 required")
    u, w = torch.empty(2, rows, hv, HEAD_DIM, dtype=torch.bfloat16, device=vb.device).unbind(0)
    kernel("wy", arch).launch(
        grid=(rows // PAIR_ROWS, hv, 1),
        akk_tma=akk,
        vb_tma=vb,
        kb_tma=kb,
        u_out=u,
        w_out=w,
        num_heads=hv,
    )
    return u, w


_SM_COUNT: dict[int, int] = {}
SLICE_BLOCKS = 4  # fwdh/dhu *_slices kernels: one 32-column value slice per CTA


def _sm_count(device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    sm = _SM_COUNT.get(index)
    if sm is None:
        sm = int(torch.cuda.get_device_properties(index).multi_processor_count)
        _SM_COUNT[index] = sm
    return sm


def _serial_walk_stage(stage, pairs, device):
    """Pick the two-half (``stage``) or four-slice (``stage + "_slices"``) kernel for the serial chunk walks.

    The slice kernels shorten each (sequence, head) walk by ~1.3x but quadruple the CTA count; they win
    only while every CTA runs in one wave (``pairs * 4 <= SM count``, measured on B200: b1 t8192
    199/236 -> 153/175 us; b4 t2048 dhu 68 -> 100 us and rlpack 8x2688 133/192 -> 157/276 us once a
    second wave appears).  Returns ``(stage name, grid x)``."""
    sm = _sm_count(device)
    if pairs * SLICE_BLOCKS <= sm:
        return stage + "_slices", pairs * SLICE_BLOCKS
    return stage, pairs * 2


def _launch_fwdh(arch, w, kg, u, gk, *, layout):
    rows, hv, _ = w.shape
    nc = layout.num_chunks
    h_out = torch.empty(nc, hv, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=w.device)
    v_new = torch.empty(rows, hv, HEAD_DIM, dtype=torch.bfloat16, device=w.device)
    if layout.has_pad_chunk:
        v_new[nc * CHUNK :].zero_()
    pairs = layout.num_sequences * hv
    stage, grid = _serial_walk_stage("fwdh", pairs, w.device)
    kernel(stage, arch).launch(
        grid=(grid, 1, 1),
        w_tma=w,
        kg_tma=kg,
        u=u,
        gk=gk,
        h_out=h_out,
        v_new=v_new,
        num_heads=hv,
        seq_chunk_start=layout.seq_chunk_start,
    )
    return h_out, v_new


def _launch_dav(arch, do, v_new, aqk_tril, *, scale):
    rows, hv, _ = do.shape
    dAqk = torch.empty(rows, hv, CHUNK, dtype=torch.float32, device=do.device)
    dv1 = torch.empty(rows, hv, HEAD_DIM, dtype=torch.bfloat16, device=do.device)
    kernel("dav", arch).launch(
        grid=(rows // PAIR_ROWS, hv, 1),
        do_tma=do,
        vnew_tma=v_new,
        aqk_tma=aqk_tril,
        dAqk=dAqk,
        dv1=dv1,
        num_heads=hv,
        scale=float(scale),
    )
    return dAqk, dv1


def _launch_dhu(arch, kg, qg, w, do, dv1, gk, *, layout, scale):
    rows, hv, _ = kg.shape
    nc = layout.num_chunks
    dh_out = torch.empty(nc, hv, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=kg.device)
    dv2 = torch.empty(rows, hv, HEAD_DIM, dtype=torch.bfloat16, device=kg.device)
    if layout.has_pad_chunk:
        dv2[nc * CHUNK :].zero_()
    pairs = layout.num_sequences * hv
    stage, grid = _serial_walk_stage("dhu", pairs, kg.device)
    kernel(stage, arch).launch(
        grid=(grid, 1, 1),
        kg_tma=kg,
        qg_tma=qg,
        w_tma=w,
        do_tma=do,
        dv1=dv1,
        gk=gk,
        dh_out=dh_out,
        dv2=dv2,
        num_heads=hv,
        seq_chunk_start=layout.seq_chunk_start,
        scale=float(scale),
    )
    return dh_out, dv2


def _launch_dqkg(arch, do, v_new, dv2, v, k_e, q_e, h, dh, akk, gk, beta, *, layout, scale):
    rows, hv, _ = do.shape
    nc = layout.num_chunks
    dev = do.device
    # h / dh are the fresh contiguous [nc, HV, 128, 128] state carriers from fwdh / dhu
    h2 = h.view(nc * hv * HEAD_DIM, HEAD_DIM)
    dh2 = dh.view(nc * hv * HEAD_DIM, HEAD_DIM)
    dq, dk, dg = torch.empty(3, rows, hv, HEAD_DIM, dtype=torch.float32, device=dev).unbind(0)
    db = torch.empty(rows, hv, dtype=torch.float32, device=dev)
    dAkk = torch.empty(rows, hv, CHUNK, dtype=torch.float32, device=dev)
    dv = torch.empty(layout.rows_external, hv, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    # dqkg is persistent: grid-stride over the nc * hv (chunk, value head) items, one CTA per SM
    kernel("dqkg", arch).launch(
        grid=(min(nc * hv, _sm_count(dev)), 1, 1),
        do_tma=do,
        vn_tma=v_new,
        dv2_tma=dv2,
        v_tma=v,
        h_tma=h2,
        dh_tma=dh2,
        akk_tma=akk,
        gk_tma=gk,  # v8: gk / k / q are staged into shared memory by TMA for the drains; h / dh / v are read from their tiles
        k_tma=k_e,
        q_tma=q_e,
        beta=beta,
        dq_out=dq,
        dk_out=dk,
        dg_out=dg,
        db_out=db,
        dAkk_out=dAkk,
        dv_out=dv,
        num_heads=hv,
        num_items=nc * hv,
        chunk_bos=layout.chunk_bos,
        chunk_len=layout.chunk_len,
        scale=float(scale),
    )
    return dq, dk, dg, db, dAkk, dv


def _launch_intra(arch, dAqk, dAkk, gk, k_e, q_e, beta, dq_f, dk_f, dg_f, db_f, *, num_chunks):
    rows, hv, _ = gk.shape
    hq = k_e.shape[1]
    dq, dk, dg = torch.empty(3, rows, hv, HEAD_DIM, dtype=torch.float32, device=gk.device).unbind(0)
    db = torch.empty_like(db_f)
    kernel("intra", arch).launch(
        grid=(num_chunks, hv, 1),
        dAqk=dAqk,
        dAkk=dAkk,
        gk=gk,
        k_e=k_e,
        q_e=q_e,
        beta=beta,
        dq_f=dq_f,
        dk_f=dk_f,
        dg_f=dg_f,
        db_f=db_f,
        dq_out=dq,
        dk_out=dk,
        dg_out=dg,
        db_out=db,
        num_heads=hv,
        num_qk_heads=hq,
        group=hv // hq,
    )
    return dq, dk, dg, db


def _launch_intra_qk(
    arch, dAqk, dAkk, gk, k_e, q_e, beta, dq_f, dk_f, dg_f, db_f, *, num_chunks, hq, q_rstd, k_rstd, layout
):
    """intra with the q/k l2norm epilogue folded in (one q/k head per value head): returns bf16 dq / dk already in
    the external (FLA) rows, plus fp32 dg / db in the internal layout; the qk epilogue kernel is then skipped."""
    rows, hv, _ = gk.shape
    if hq != hv:
        raise ValueError("_launch_intra_qk requires one q/k head per value head")
    rows_ext = layout.rows_external
    dq, dk = torch.empty(2, rows_ext, hq, HEAD_DIM, dtype=torch.bfloat16, device=gk.device).unbind(0)
    dg = torch.empty(rows, hv, HEAD_DIM, dtype=torch.float32, device=gk.device)
    db = torch.empty_like(db_f)
    kernel("intra_qk", arch).launch(
        grid=(num_chunks, hv, 1),
        dAqk=dAqk,
        dAkk=dAkk,
        gk=gk,
        k_e=k_e,
        q_e=q_e,
        beta=beta,
        dq_f=dq_f,
        dk_f=dk_f,
        dg_f=dg_f,
        db_f=db_f,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        chunk_bos=layout.chunk_bos,
        chunk_len=layout.chunk_len,
        dq_ext=dq,
        dk_ext=dk,
        dg_out=dg,
        db_out=db,
        num_heads=hv,
        num_qk_heads=hq,
        group=hv // hq,
    )
    return dq, dk, dg, db


def _launch_epilogue(
    arch,
    dq_intra,
    dk_intra,
    dg_intra,
    db_total,
    *,
    q_norm,
    k_norm,
    q_rstd,
    k_rstd,
    g_raw,
    beta_raw,
    A_log,
    dt_bias,
    layout,
    lower_bound,
    qk_done=None,
):
    rows, hv, kd = dg_intra.shape
    h = q_norm.shape[1]
    nc = layout.num_chunks
    rows_ext = layout.rows_external
    dev, bf = dg_intra.device, torch.bfloat16
    dg_out = torch.empty(rows_ext, hv, kd, dtype=bf, device=dev)
    dbeta = torch.empty(rows_ext, hv, dtype=bf, device=dev)
    dA_part, dbias_part = torch.empty(2, nc, hv, kd, dtype=torch.float32, device=dev).unbind(0)
    if qk_done is None:
        dq_out = torch.empty(rows_ext, h, kd, dtype=bf, device=dev)
        dk_out = torch.empty(rows_ext, h, kd, dtype=bf, device=dev)
    else:
        dq_out, dk_out = qk_done  # bf16 external rows from the folded intra epilogue
    dA_log = torch.empty(hv, dtype=torch.float32, device=dev)
    dt_bias_grad = torch.empty(hv * kd, dtype=torch.float32, device=dev)
    kernel("gate_epilogue", arch).launch(
        grid=(nc, hv, 1),
        dg_intra=dg_intra,
        g_raw=g_raw,
        db_total=db_total,
        beta_raw=beta_raw,
        A_log=A_log,
        dt_bias=dt_bias,
        chunk_bos=layout.chunk_bos,
        chunk_len=layout.chunk_len,
        dg_out=dg_out,
        dbeta=dbeta,
        dA_part=dA_part,
        dbias_part=dbias_part,
        num_heads=hv,
        lower_bound=float(lower_bound),
    )
    if qk_done is None:
        kernel("qk_epilogue", arch).launch(
            grid=(nc, h, 1),
            dq_intra=dq_intra,
            dk_intra=dk_intra,
            q_norm=q_norm,
            k_norm=k_norm,
            q_rstd=q_rstd,
            k_rstd=k_rstd,
            chunk_bos=layout.chunk_bos,
            chunk_len=layout.chunk_len,
            dq_out=dq_out,
            dk_out=dk_out,
            num_qk_heads=h,
            num_v_heads=hv,
            group=hv // h,
        )
    kernel("finalize", arch).launch(
        grid=(hv, 1, 1),
        dA_part=dA_part,
        dbias_part=dbias_part,
        dA_log=dA_log,
        dt_bias_grad=dt_bias_grad,
        num_chunks=nc,
        num_heads=hv,
    )
    return dq_out, dk_out, dg_out, dbeta, dA_log, dt_bias_grad


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
_UNUSED_BETA_OPERANDS: dict[tuple, torch.Tensor] = {}
_UNUSED_BETA_OPERANDS_LIMIT = 64


def _unused_beta_operand(rows: int, hv: int, device) -> torch.Tensor:
    """bf16 ``[rows, hv]`` zeros for the gate epilogue's ``beta_raw`` slot when the fused sigmoid
    backward output is discarded (post-sigmoid ``beta``).  The kernel only reads it."""
    key = (int(rows), int(hv), device)
    t = _UNUSED_BETA_OPERANDS.get(key)
    if t is None:
        if len(_UNUSED_BETA_OPERANDS) >= _UNUSED_BETA_OPERANDS_LIMIT:
            _UNUSED_BETA_OPERANDS.clear()
        t = torch.zeros(rows, hv, dtype=torch.bfloat16, device=device)
        _UNUSED_BETA_OPERANDS[key] = t
    return t


def chunk_kda_backward(
    *,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    beta_logits: torch.Tensor | None = None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    lower_bound: float,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Deterministic chunked KDA training backward.

    Arguments follow the chunked KDA forward's saved-for-backward set:

    - ``q_norm``, ``k_norm``: L2-normalized q/k, bf16 ``[B, T, H, 128]``; ``q_rstd``, ``k_rstd``:
      their inverse norms, fp32 ``[B, T, H]``.
    - ``v``, ``g``: bf16 ``[B, T, HV, 128]`` (``g`` are the raw gate pre-activations).
    - ``beta``: fp32 ``[B, T, HV]`` = the beta the forward consumed (sigmoid(beta_logits) when the
      sigmoid is fused into the forward, otherwise the caller's post-sigmoid beta).
    - ``beta_logits``: optional bf16 ``[B, T, HV]`` raw beta. When given, ``dbeta`` is the gradient
      with respect to the logits (bf16, sigmoid backward fused); when ``None``, ``dbeta`` is the
      gradient with respect to ``beta`` itself, returned in ``beta``'s dtype.
    - ``A_log``: fp32 ``[HV]``; ``dt_bias``: fp32 ``[HV * 128]``.
    - ``Aqk``, ``Akk``: bf16 ``[B, T, HV, 64]`` per-chunk matrices saved by the forward.
    - ``do``: bf16 ``[B, T, HV, 128]`` output gradient; ``scale``: attention scale;
      ``lower_bound``: the safe-gate lower bound (negative).
    - ``cu_seqlens``: optional int32 ``[N + 1]`` packed-sequence offsets (``B`` must be 1); the
      sequences may have any lengths, as in the reference's varlen convention.
    - ``cu_seqlens_cpu``: host copy of ``cu_seqlens`` -- a CPU int64 tensor (the layer's canonical
      form) or a sequence of ints. With it the backward issues no device-to-host copy and, once the
      packing's layout is cached, no host-to-device copy and no stream synchronisation either (the
      first backward of a new packing uploads the layout tables with one non-blocking copy). Without
      it ``cu_seqlens`` is read from the device once per call (a fallback for direct callers).

    Any ``T`` is accepted.  ``HV`` must be a positive multiple of ``H`` (grouped value heads).
    Returns ``dq``/``dk`` bf16 ``[B, T, H, 128]``, ``dv`` bf16 ``[B, T, HV, 128]``, ``dbeta`` bf16
    ``[B, T, HV]``, ``dg`` bf16 ``[B, T, HV, 128]``, ``dA_log`` fp32 ``[HV]``, ``dt_bias`` fp32
    ``[HV * 128]``.  Repeated calls on identical inputs are bit-identical.
    """
    arch = device_arch(v.device)
    batch0, seq0, h, kd = q_norm.shape
    hv = v.shape[2]
    _check(kd == HEAD_DIM and v.shape[-1] == HEAD_DIM, "K = V = 128 is required")
    _check(hv % h == 0, "HV must be a multiple of H")
    layout = chunk_layout(batch0, seq0, cu_seqlens, q_norm.device, cu_seqlens_cpu=cu_seqlens_cpu)
    rows = layout.rows_external
    bf, f32 = torch.bfloat16, torch.float32

    qn = _contiguous("q_norm", _rows(q_norm, rows), bf)
    kn = _contiguous("k_norm", _rows(k_norm, rows), bf)
    qr = _as_dtype(_rows(q_rstd, rows).contiguous(), f32)
    kr = _as_dtype(_rows(k_rstd, rows).contiguous(), f32)
    v_r = _contiguous("v", _rows(v, rows), bf)
    g_r = _contiguous("g", _rows(g, rows), bf)
    beta_s = _contiguous("beta", _rows(beta, rows), f32)
    if beta_logits is not None:
        beta_raw = _contiguous("beta_logits", _rows(beta_logits, rows), bf)
    else:
        # post-sigmoid beta: the gate epilogue's fused sigmoid backward output is not used; it
        # still needs a bf16 operand of the right shape, served from a small per-shape cache
        # instead of a per-call cast (one allocation and one copy kernel fewer per backward).
        beta_raw = _unused_beta_operand(rows, hv, q_norm.device)
    do_r = _contiguous("do", _rows(do, rows), bf)
    aqk = _contiguous("Aqk", _rows(Aqk, rows), bf)
    akk = _contiguous("Akk", _rows(Akk, rows), bf)
    A_log = _as_dtype(A_log.contiguous(), f32)
    dt_bias = _as_dtype(dt_bias.contiguous(), f32)

    # prep gathers the token rows into the internal layout; every stage below runs there, and dqkg /
    # the epilogue scatter the gradients back to the token rows.
    pre = _launch_prep(
        arch, g_r, qn, kn, v_r, beta_s, A_log, dt_bias, aqk, akk, do_r, layout=layout, lower_bound=lower_bound
    )
    u, w = _launch_wy(arch, pre["akk"], pre["vb"], pre["kb"])
    h_state, v_new = _launch_fwdh(arch, w, pre["kg"], u, pre["gk"], layout=layout)
    dAqk, dv1 = _launch_dav(arch, pre["do"], v_new, pre["aqk_tril"], scale=scale)
    dh_state, dv2 = _launch_dhu(arch, pre["kg"], pre["qg"], w, pre["do"], dv1, pre["gk"], layout=layout, scale=scale)
    dq_f, dk_f, dg_f, db_f, dAkk, dv = _launch_dqkg(
        arch,
        pre["do"],
        v_new,
        dv2,
        pre["v"],
        pre["ke"],
        pre["qe"],
        h_state,
        dh_state,
        pre["akk"],
        pre["gk"],
        pre["beta"],
        layout=layout,
        scale=scale,
    )
    if hv == h:  # one q/k head per value head (GVA batches keep the two-kernel path)
        # one q/k head per value head: intra applies the l2norm backward itself and writes bf16 dq / dk into
        # the external rows; the qk epilogue kernel is skipped
        dq_i, dk_i, dg_i, db_i = _launch_intra_qk(
            arch,
            dAqk,
            dAkk,
            pre["gk"],
            pre["ke"],
            pre["qe"],
            pre["beta"],
            dq_f,
            dk_f,
            dg_f,
            db_f,
            num_chunks=layout.num_chunks,
            hq=h,
            q_rstd=qr,
            k_rstd=kr,
            layout=layout,
        )
        qk_done = (dq_i, dk_i)
    else:
        dq_i, dk_i, dg_i, db_i = _launch_intra(
            arch,
            dAqk,
            dAkk,
            pre["gk"],
            pre["ke"],
            pre["qe"],
            pre["beta"],
            dq_f,
            dk_f,
            dg_f,
            db_f,
            num_chunks=layout.num_chunks,
        )
        qk_done = None
    dq, dk, dg, dbeta, dA_log_grad, dt_bias_grad = _launch_epilogue(
        arch,
        dq_i,
        dk_i,
        dg_i,
        db_i,
        q_norm=qn,
        k_norm=kn,
        q_rstd=qr,
        k_rstd=kr,
        g_raw=g_r,
        beta_raw=beta_raw,
        A_log=A_log,
        dt_bias=dt_bias,
        layout=layout,
        lower_bound=lower_bound,
        qk_done=qk_done,
    )
    # every gradient below is a fresh contiguous kernel output (or an index_select of one): view, not reshape
    return {
        "dq": dq.view(batch0, seq0, h, kd),
        "dk": dk.view(batch0, seq0, h, kd),
        "dv": dv.view(batch0, seq0, hv, kd),
        "dbeta": (
            dbeta.view(batch0, seq0, hv)
            if beta_logits is not None
            else _as_dtype(layout.scatter(db_i).view(batch0, seq0, hv), beta.dtype)
        ),
        "dg": dg.view(batch0, seq0, hv, kd),
        "dA_log": dA_log_grad,
        "dt_bias": dt_bias_grad,
    }


__all__ = ["ChunkLayout", "chunk_kda_backward", "chunk_layout", "host_boundaries", "CHUNK", "HEAD_DIM"]
