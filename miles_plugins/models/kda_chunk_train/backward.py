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

A layout uploads its device tables once, from one pinned staging buffer with one
``non_blocking`` copy: the per-chunk and per-sequence int32 tables of the stages,
fla's ``chunk_indices`` of the same packing (the fla forward of the
``deterministic_fla_fwd`` mode consumes them) and the two-entry ``cu_seqlens`` of
the first and last sequence that fla's CP pre-process kernels index. The per-row
scatter table (``internal_rows``, int64) is built and uploaded on first use only.

Context parallelism (fla ``FLACPContext`` with a process group) follows the deterministic kernels' CP host
protocol (fla's state-exchange pre-process around the Cake stages, fed with fla's own recompute of its operands),
mirrored here stage for stage from the Cake compiler's reference host wrapper: the forward seeds the first local
sequence with the compressed state fla's forward pre-process exchanged, the backward runs fla's backward
pre-process between dAv and the adjoint recurrence and seeds the last local sequence's adjoint state with
the exchanged ``dht``; the exchange kernels and the one ``all_gather`` per call are fla's own.
"""

from __future__ import annotations

import functools
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
_WALKS = ("auto", "v1", "v2")  # serial_walk: the launchers' dispatch rule, or the pinned fwdh / dhu kernel


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
    ``chunk_indices[c]``: ``(sequence, chunk index within it)`` of real chunk ``c`` -- the rows of fla's
    ``prepare_chunk_indices`` for the same packing, handed to the fla forward of the ``deterministic_fla_fwd``
    mode (``chunk_indices_flat`` is its storage; the ``[nc, 2]`` view is taken on first use); ``cp_cu_first`` /
    ``cp_cu_last``: int32 ``[0, L]`` of the first / last sequence, the two-entry ``cu_seqlens`` fla's CP
    pre-process kernels index; ``seq_order[r]``: the sequence the ``fwdh`` / ``dhu`` walks dispatch at CTA rank
    ``r`` (sequences by chunk count descending, ties by index ascending; the order changes no output byte).  All
    of these are int32 views of ONE device buffer filled by one host-to-device copy.  ``internal_rows[r]``
    (int64): internal row of input row ``r``, built and uploaded the first time :meth:`scatter` needs it.

    Layouts are immutable and cached per packing, so the derived counts below are computed once per
    layout rather than on every access from the backward's hot path.
    """

    lengths: tuple[int, ...]
    offsets: tuple[int, ...]
    rows_external: int
    seq_chunk_start_cpu: tuple[int, ...]
    seq_order_cpu: tuple[int, ...]
    chunk_bos: torch.Tensor
    chunk_len: torch.Tensor
    seq_chunk_start: torch.Tensor
    seq_order: torch.Tensor
    chunk_indices_flat: torch.Tensor
    cp_cu_first: torch.Tensor
    cp_cu_last: torch.Tensor

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

    @cached_property
    def chunk_indices(self) -> torch.Tensor:
        """int32 ``[num_chunks, 2]``: fla's chunk indices of this packing (a view of the uploaded tables)."""
        return self.chunk_indices_flat.view(self.num_chunks, 2)

    @cached_property
    def internal_rows(self) -> torch.Tensor:
        """int64 ``[rows_external]``: the internal row of every input row (:meth:`scatter`'s index). Built on first
        use -- one pinned staging buffer, one ``non_blocking`` host-to-device copy -- and kept with the layout."""
        return _row_table(self.lengths, self.offsets, self.rows_external, self.seq_chunk_start_cpu, self.chunk_bos.device)

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


_TABLE_ALIGN = 64  # int32 elements: every table view starts 256-byte aligned inside the shared buffer


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _chunk_tables(lengths: tuple[int, ...], offsets: tuple[int, ...], bos_out, clen_out, ci_out):
    """Fill the per-chunk tables straight into their (pinned) int32 staging views, vectorised: ``bos_out`` /
    ``clen_out`` (``nc`` entries each: first input row, valid tokens) and ``ci_out`` (``[nc, 2]``: sequence
    index, chunk index within it -- the rows of fla's ``prepare_chunk_indices``). Returns ``starts`` (``N + 1``
    Python ints). Per-sequence scalars stay in Python (a pack has a few sequences); only the per-chunk arrays
    go through numpy."""
    n = len(lengths)
    nch = [(length + (CHUNK - 1)) // CHUNK for length in lengths]
    starts = [0] * (n + 1)
    for s in range(n):
        starts[s + 1] = starts[s] + nch[s]
    nc = starts[-1]
    # per chunk: index within its sequence (by repeat), first input row, valid tokens, owning sequence
    i = np.arange(nc, dtype=np.int64) - np.repeat(np.asarray(starts[:-1], dtype=np.int64), nch)
    bos_out[:] = np.repeat(np.asarray(offsets, dtype=np.int64), nch) + i * CHUNK
    clen_out[:] = np.minimum(CHUNK, np.repeat(np.asarray(lengths, dtype=np.int64), nch) - i * CHUNK)
    ci_out[:, 0] = np.repeat(np.arange(n, dtype=np.int64), nch)
    ci_out[:, 1] = i
    return starts


def _longest_first(starts) -> list[int]:
    """``seq_order``: sequence indices by chunk count descending, ties by index ascending (the serial walks
    dispatch the longest sequence first; the order does not change any output byte)."""
    counts = [starts[n + 1] - starts[n] for n in range(len(starts) - 1)]
    return sorted(range(len(counts)), key=lambda n: (-counts[n], n))


def _row_table(lengths: tuple[int, ...], offsets: tuple[int, ...], rows_external: int, starts, device) -> torch.Tensor:
    """int64 ``internal_rows`` of the packing on ``device`` (internal row = row + (first internal row of its
    sequence - its exclusive cumsum)): written into a pinned staging buffer with a single ``np.add(..., out=)``
    when the sequences lie back to back (packed and fixed-length input: offsets = exclusive cumsum of the
    lengths), else with one fancy-index assignment; then one ``non_blocking`` host-to-device copy. On a CPU
    device the buffer is pageable and the copy is a plain copy (tests)."""
    n = len(lengths)
    pinned = device.type == "cuda"
    host = torch.empty(rows_external, dtype=torch.int64, pin_memory=pinned)
    rows_out = host.numpy()
    excl = [0] * n
    for s in range(1, n):
        excl[s] = excl[s - 1] + lengths[s - 1]
    shift = np.asarray([starts[s] * CHUNK - excl[s] for s in range(n)], dtype=np.int64)
    if tuple(excl) == offsets:
        np.add(np.arange(rows_external, dtype=np.int64), np.repeat(shift, lengths), out=rows_out)
    else:
        j = np.arange(rows_external, dtype=np.int64) - np.repeat(np.asarray(excl, dtype=np.int64), lengths)
        rows_out[np.repeat(np.asarray(offsets, dtype=np.int64), lengths) + j] = np.repeat(shift + np.asarray(excl, dtype=np.int64), lengths) + j
    dev = torch.empty(rows_external, dtype=torch.int64, device=device)
    dev.copy_(host, non_blocking=pinned)
    return dev


def _build_layout(lengths, offsets, rows_external: int, device) -> ChunkLayout:
    """Build the layout of ``lengths`` / ``offsets`` on ``device``.

    Host cost is a handful of vectorised numpy ops writing straight into ONE pinned int32 staging buffer
    (``chunk_bos``, ``chunk_len``, ``seq_chunk_start``, ``chunk_indices`` and the two CP ``cu_seqlens`` pairs,
    each a 256-byte-aligned section); the device side is exactly one device allocation and one ``non_blocking``
    host-to-device copy, so building a layout never drains the stream (the U0-1 gate allows up to two such
    copies per new packing; the second one is the per-row ``internal_rows`` table, built on first use). The
    staging buffer comes from torch's caching pinned-memory allocator, which hands it out again once the copy
    has retired. On a CPU device the buffer is pageable and the copy is a plain copy (tests).
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
    o_start = 2 * table
    o_ci = o_start + _round_up(n + 1, _TABLE_ALIGN)
    o_cp = o_ci + _round_up(2 * nc, _TABLE_ALIGN)
    o_so = o_cp + _TABLE_ALIGN  # seq_order section (the CP pairs use 4 of the preceding aligned block)
    host = torch.empty(o_so + _round_up(n, _TABLE_ALIGN), dtype=torch.int32, pin_memory=pinned)
    a = host.numpy()  # numpy views of the staging buffer cost well under a microsecond; a torch view op ~4 us
    starts = _chunk_tables(lengths, offsets, a[:nc], a[table : table + nc], a[o_ci : o_ci + 2 * nc].reshape(nc, 2))
    if nc_pad != nc:  # trailing pad chunk: no rows, zero length
        a[nc] = 0
        a[table + nc] = 0
    a[o_start : o_start + n + 1] = starts
    a[o_cp : o_cp + 4] = (0, lengths[0], 0, lengths[-1])
    order = _longest_first(starts)
    a[o_so : o_so + n] = order

    dev = torch.empty(host.numel(), dtype=torch.int32, device=device)
    dev.copy_(host, non_blocking=pinned)
    # every table view comes out of ONE split call (gap sections may be empty) rather than one slice op per table
    bos, _, clen, _, start, _, ci, _, cp_first, cp_last, _, sorder, _ = dev.split(
        [nc_pad, table - nc_pad, nc_pad, table - nc_pad, n + 1, o_ci - o_start - (n + 1), 2 * nc, o_cp - o_ci - 2 * nc,
         2, 2, _TABLE_ALIGN - 4, n, _round_up(n, _TABLE_ALIGN) - n]
    )
    return ChunkLayout(
        lengths=lengths,
        offsets=offsets,
        rows_external=int(rows_external),
        seq_chunk_start_cpu=tuple(starts),
        seq_order_cpu=tuple(order),
        chunk_bos=bos,
        chunk_len=clen,
        seq_chunk_start=start,
        seq_order=sorder,
        chunk_indices_flat=ci,
        cp_cu_first=cp_first,
        cp_cu_last=cp_last,
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
    num_pairs = rows // PAIR_ROWS
    # persistent: one CTA per SM grid-striding over the num_pairs * hv tiles; u / w are the contiguous bf16
    # [R, HV, 128] TMA-store destinations
    kernel("wy", arch).launch(
        grid=(max(1, min(num_pairs * hv, _sm_count(vb.device))), 1, 1),
        akk_tma=akk,
        vb_tma=vb,
        kb_tma=kb,
        u_tma=u,
        w_tma=w,
        num_heads=hv,
        num_pairs=num_pairs,
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


def _serial_walk_stage(stage, pairs, device, walk="auto"):
    """Pick the two-half (``stage``) or four-slice (``stage + "_slices"``) kernel for the serial chunk walks.

    The slice kernels shorten each (sequence, head) walk by ~1.3x but quadruple the CTA count; they win
    only while every CTA runs in one wave (``pairs * 4 <= SM count``, measured on B200: b1 t8192
    199/236 -> 153/175 us; b4 t2048 dhu 68 -> 100 us and rlpack 8x2688 133/192 -> 157/276 us once a
    second wave appears).  ``walk``: ``"auto"`` = that rule; ``"v1"`` / ``"v2"`` pin the two-half / four-slice
    kernel (the CP host protocol's ``serial_walk`` knob; v1 and v2 are bitwise equal on every row).
    Returns ``(stage name, grid x)``."""
    _check(walk in _WALKS, f"serial_walk must be one of {_WALKS}, got {walk!r}")
    if walk == "auto":
        walk = "v2" if pairs * SLICE_BLOCKS <= _sm_count(device) else "v1"
    if walk == "v2":
        return stage + "_slices", pairs * SLICE_BLOCKS
    return stage, pairs * 2


_STATE_DUMMY: dict = {}


def state_dummy(device) -> torch.Tensor:
    """Cached one-element fp32 zero bound to a flag-off state slot of ``fwdh`` / ``dhu`` (``use_h0 = 0`` /
    ``store_ht = 0`` / ``use_dht = 0`` / ``store_dh0 = 0``): the kernels never dereference the pointer when the
    flag is off, so no per-call allocation and no extra launch."""
    key = _device_key(device)
    hit = _STATE_DUMMY.get(key)
    if hit is None:
        hit = _STATE_DUMMY[key] = torch.zeros(1, dtype=torch.float32, device=device)
    return hit


def _check_state(t: torch.Tensor, *, name: str, num_sequences: int, hv: int) -> torch.Tensor:
    """An fp32 ``[N, HV, 128, 128]`` state carrier (k rows x v cols, contiguous; slot = layout sequence index)."""
    _check(
        t.dtype == torch.float32 and t.is_contiguous() and tuple(t.shape) == (num_sequences, hv, HEAD_DIM, HEAD_DIM),
        f"{name}: expected contiguous fp32 [{num_sequences}, {hv}, 128, 128], got {t.dtype} {tuple(t.shape)}",
    )
    return t


def _state_args(state_in, return_out, *, layout, hv, device, in_name, out_name):
    """``(state_in, use_in, state_out, store_out)`` for the two state slots of a serial-walk launch."""
    n = layout.num_sequences
    if state_in is None:
        state_in, use_in = state_dummy(device), 0
    else:
        state_in, use_in = _check_state(state_in, name=in_name, num_sequences=n, hv=hv), 1
    if return_out:
        state_out, store_out = torch.empty(n, hv, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device), 1
    else:
        state_out, store_out = state_dummy(device), 0
    return state_in, use_in, state_out, store_out


def _use_cp(cp_context) -> bool:
    """An fla CP context with a process group: the CP host protocol applies."""
    return (
        cp_context is not None and getattr(cp_context, "group", None) is not None
    )  # config-access-exempt: fla CP contexts carry a process group only when built for a CP group


def _first_slot(state_c: torch.Tensor, n: int) -> torch.Tensor:
    """Compressed CP state ``[1, HV, 128, 128]`` -> the ``[n, HV, 128, 128]`` carrier with slot 0 filled (fla's
    ``expand_h0``); ``n == 1``: the tensor itself."""
    if n == 1:
        return state_c
    out = state_c.new_zeros(n, *state_c.shape[1:])
    out[0].copy_(state_c[0])
    return out


def _last_slot(state_c: torch.Tensor, n: int) -> torch.Tensor:
    """``dht`` of the single-sequence pre-process call (slot 0) -> slot ``n - 1`` of the ``[n, HV, 128, 128]`` carrier."""
    if n == 1:
        return state_c
    out = state_c.new_zeros(n, *state_c.shape[1:])
    out[n - 1].copy_(state_c[0])
    return out


@functools.cache
def _fla_ops() -> dict:
    """fla's pieces of the CP host protocol (imported on first use: fla is only needed under CP). The pre-process
    wrappers are fla's own CP kernels (``pre_process_*_kernel_merged`` / ``merge_fwd_bwd_kernel`` + one
    ``all_gather`` per call on every rank); the rest is the A1 feed's recompute of fla's forward operands."""
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.common.gate import fused_beta_sigmoid_fwd
    from fla.ops.cp.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu_pre_process,
        chunk_gated_delta_rule_fwd_h_pre_process,
    )
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
    from fla.ops.kda.gate import kda_gate_chunk_cumsum
    from fla.ops.kda.wy_fast import recompute_w_u_fwd
    from fla.ops.utils.constant import RCP_LN2

    return {
        "l2norm_fwd": l2norm_fwd,
        "fused_beta_sigmoid_fwd": fused_beta_sigmoid_fwd,
        "chunk_kda_fwd_intra": chunk_kda_fwd_intra,
        "kda_gate_chunk_cumsum": kda_gate_chunk_cumsum,
        "recompute_w_u_fwd": recompute_w_u_fwd,
        "fwd_h_pre_process": chunk_gated_delta_rule_fwd_h_pre_process,
        "bwd_dhu_pre_process": chunk_gated_delta_rule_bwd_dhu_pre_process,
        "RCP_LN2": RCP_LN2,
    }


def _a1_forward_feed(fl, q, k, v, g_raw, beta_logits, A_log, dt_bias, *, scale, lower_bound):
    """fla's forward operands of one sequence ``[1, L, ., 128]`` (raw q/k/v/g/beta-logit slices): ``g`` (log2 chunk
    cumsum, fp32), ``w`` / ``u`` / ``kg`` (bf16) exactly as fla's own forward computes them for that sequence
    (fla chunk.py l2norm + fused sigmoid, chunk_fwd.py), in fixed-length mode (``B = 1``, no ``cu_seqlens``, no
    chunk indices, no host copies): the forward feed of the CP host protocol."""
    q_n, _ = fl["l2norm_fwd"](q)
    k_n, _ = fl["l2norm_fwd"](k)
    beta = fl["fused_beta_sigmoid_fwd"](beta_logits, 1.0)
    g = fl["kda_gate_chunk_cumsum"](
        g=g_raw, A_log=A_log, dt_bias=dt_bias, scale=fl["RCP_LN2"], chunk_size=CHUNK, cu_seqlens=None,
        chunk_indices=None, lower_bound=lower_bound,
    )
    w, u, _qg, kg, _aqk, _akk = fl["chunk_kda_fwd_intra"](
        q=q_n, k=k_n, v=v, gk=g, beta=beta, scale=scale, cu_seqlens=None, chunk_size=CHUNK, chunk_indices=None,
        safe_gate=True, disable_recompute=True,
    )
    return {"g": g, "w": w, "u": u, "kg": kg}


def _a1_backward_feed(fl, q_norm, k_norm, v, beta_s, Akk, g_raw, A_log, dt_bias, *, lower_bound):
    """fla's backward recompute operands of one sequence ``[1, L, ., 128]`` from the saved set: ``g``, ``w``,
    ``qg``, ``kg`` (fla chunk_bwd.py recompute; ``u`` is discarded): the backward feed of the CP host protocol."""
    g = fl["kda_gate_chunk_cumsum"](
        g=g_raw, A_log=A_log, dt_bias=dt_bias, scale=fl["RCP_LN2"], chunk_size=CHUNK, cu_seqlens=None,
        chunk_indices=None, lower_bound=lower_bound,
    )
    w, _u, qg, kg = fl["recompute_w_u_fwd"](
        k=k_norm, v=v, beta=beta_s, A=Akk, gk=g, q=q_norm, cu_seqlens=None, chunk_indices=None
    )
    return {"g": g, "w": w, "qg": qg, "kg": kg}


def _roles(cp_context) -> dict:
    return {
        k: getattr(cp_context, k, None)  # config-access-exempt: fla CP context fields are selected by role name
        for k in ("is_first_rank", "is_last_rank", "pre_num_ranks", "post_num_ranks", "num_seqs")
    }


#: Record of this process's most recent CP exchange (diagnostics; not consumed by the op):
#: ``h0_compressed`` (fp32 ``[1, HV, 128, 128]`` k rows x v cols, None when no state enters this rank), ``dht``
#: (same layout, None until the backward ran), ``fwdh_variant`` / ``dhu_variant`` (``"v1"`` | ``"v2"``), ``roles``,
#: ``local_lengths``, ``forward_source``.
LAST_CP_STATES: dict = {}


def _launch_fwdh(arch, w, kg, u, gk, *, layout, h0=None, return_ht=False, walk="auto", zero_pad_rows=True):
    """Forward state walk. ``h0`` fp32 ``[N, HV, 128, 128]`` seeds each sequence's state (``None`` = zeros, the
    flag-off path); with ``return_ht`` the fp32 state after each sequence's last chunk is returned third.
    ``zero_pad_rows``: fill the trailing pad chunk's rows of ``v_new`` (never written by the walk) with zeros --
    the backward keeps the fill for its pair-tiled consumers; the forward passes ``False`` because ``fwd_o`` walks
    the real chunks only and never reads those rows."""
    rows, hv, _ = w.shape
    nc = layout.num_chunks
    h_out = torch.empty(nc, hv, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=w.device)
    v_new = torch.empty(rows, hv, HEAD_DIM, dtype=torch.bfloat16, device=w.device)
    if zero_pad_rows and layout.has_pad_chunk:
        v_new[nc * CHUNK :].zero_()
    pairs = layout.num_sequences * hv
    stage, grid = _serial_walk_stage("fwdh", pairs, w.device, walk)
    h0_t, use_h0, ht_t, store_ht = _state_args(
        h0, return_ht, layout=layout, hv=hv, device=w.device, in_name="h0", out_name="ht"
    )
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
        h0=h0_t,
        use_h0=use_h0,
        ht_out=ht_t,
        store_ht=store_ht,
        seq_order=layout.seq_order,
    )
    if return_ht:
        return h_out, v_new, ht_t
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


def _launch_dhu(arch, kg, qg, w, do, dv1, gk, *, layout, scale, dht=None, return_dh0=False, walk="auto"):
    """Reverse state walk. ``dht`` fp32 ``[N, HV, 128, 128]`` seeds each sequence's adjoint state (``None`` =
    zeros); with ``return_dh0`` the fp32 adjoint state before each sequence's first chunk is returned third."""
    rows, hv, _ = kg.shape
    nc = layout.num_chunks
    dh_out = torch.empty(nc, hv, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=kg.device)
    dv2 = torch.empty(rows, hv, HEAD_DIM, dtype=torch.bfloat16, device=kg.device)
    if layout.has_pad_chunk:
        dv2[nc * CHUNK :].zero_()
    pairs = layout.num_sequences * hv
    stage, grid = _serial_walk_stage("dhu", pairs, kg.device, walk)
    dht_t, use_dht, dh0_t, store_dh0 = _state_args(
        dht, return_dh0, layout=layout, hv=hv, device=kg.device, in_name="dht", out_name="dh0"
    )
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
        dht=dht_t,
        use_dht=use_dht,
        dh0_out=dh0_t,
        store_dh0=store_dh0,
        seq_order=layout.seq_order,
    )
    if return_dh0:
        return dh_out, dv2, dh0_t
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
        dk_tma=dk,
        dg_tma=dg,
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
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    return_dh0: bool = False,
    cp_context=None,
    serial_walk: str = "auto",
    feed_beta_from_logits: bool = False,
) -> dict[str, torch.Tensor | None]:
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
    - ``h0`` / ``dht``: optional fp32 ``[N, HV, 128, 128]`` state carriers (k rows x v cols, one slot per
      sequence of the layout): the state entering each sequence's first chunk (the forward's
      ``initial_state``) and the adjoint state entering its last chunk (the gradient of the forward's final
      state). ``None`` = zeros (the flag-off path: the kernels read a cached one-element zero and skip the
      load). ``return_dh0`` returns the adjoint state before each sequence's first chunk as ``dh0`` (the
      gradient of ``h0``); otherwise ``dh0`` is ``None``.
    - ``cp_context``: fla ``FLACPContext`` with a process group = context parallelism, the host protocol of
      the deterministic kernels (fla's state exchange around the Cake stages) mirrored stage for stage: ``h0`` is
      then the **compressed** state
      entering this rank's first local sequence (fp32 ``[1, HV, 128, 128]`` k rows x v cols, the forward's
      ``h0_compressed``; ``None`` on a rank that receives none) and ``dht`` / ``return_dh0`` must be ``None`` /
      ``False`` (fla discards ``dh0`` under CP). Between dAv and the adjoint recurrence the backward runs fla's
      CP backward pre-process on the first local sequence (the A1 feed: fla's own recompute of ``g`` / ``w`` /
      ``qg`` / ``kg`` from the saved set, ``state_v_first=True`` as the fla path tunes those kernels, one
      ``all_gather``) and seeds the last local sequence's adjoint state with the exchanged ``dht``.
    - ``serial_walk``: ``"auto"`` (the launchers' dispatch rule) | ``"v1"`` | ``"v2"`` pins the fwdh / dhu kernel.
    - ``feed_beta_from_logits``: the A1 feed's beta is fla's ``fused_beta_sigmoid_fwd`` of ``beta_logits`` on the
      boundary sequence (Cake forward: the saved ``beta`` is Cake's sigmoid) instead of the saved ``beta`` slice.

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
    use_cp = _use_cp(cp_context)
    _check(
        not use_cp or (dht is None and not return_dh0),
        "dht / dh0 are not part of the CP contract (fla discards dh0 under context parallelism)",
    )

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
    # forward state recurrence from the carried state (CP: the compressed h0 of the first local sequence in slot 0;
    # the first rank of its sequence receives none -- fla carries zeros there, the kernel's flag-off path is the
    # same values without the 64 KiB/head load)
    h0_c = None
    if use_cp:
        if h0 is not None:
            h0_c = _check_state(h0, name="h0 (compressed)", num_sequences=1, hv=hv)
        h0 = None if (h0_c is None or cp_context.is_first_rank) else _first_slot(h0_c, layout.num_sequences)
    h_state, v_new = _launch_fwdh(arch, w, pre["kg"], u, pre["gk"], layout=layout, h0=h0, walk=serial_walk)
    dAqk, dv1 = _launch_dav(arch, pre["do"], v_new, pre["aqk_tril"], scale=scale)
    # CP hook (fla chunk_bwd.py, between dAv and dhu): fla's backward pre-process on the FIRST local sequence
    # (internal rows [0, L1): prep gathers the sequences in order from row 0) -> the adjoint state entering the
    # LAST local sequence. state_v_first=True as the Miles fla path tunes fla's kernels (Triton's autotune cache keys
    # ignore the constexpr; a False call after a True one reuses the V-first config and overflows shared memory on
    # GB300); the compressed state is transposed once on the host.
    dht_c = None
    if use_cp:
        fl = _fla_ops()
        first = layout.lengths[0]
        if feed_beta_from_logits:
            _check(beta_logits is not None, "feed_beta_from_logits needs beta_logits")
            beta_feed = fl["fused_beta_sigmoid_fwd"](beta_raw[:first][None], 1.0)
        else:
            beta_feed = beta_s[:first][None]
        feed = _a1_backward_feed(
            fl, qn[:first][None], kn[:first][None], v_r[:first][None], beta_feed, akk[:first][None], g_r[:first][None],
            A_log, dt_bias, lower_bound=lower_bound,
        )
        dht_vf, _none = fl["bwd_dhu_pre_process"](
            q=feed["qg"], k=feed["kg"], w=feed["w"], do=pre["do"][:first][None], dv=dv1[:first][None], gk=feed["g"],
            scale=scale, state_v_first=True, cu_seqlens=layout.cp_cu_first, dht=None, initial_state=None,
            context=cp_context, chunk_size=CHUNK,
        )
        del feed
        dht_c = dht_vf.transpose(-1, -2).contiguous()
        dht = None if cp_context.is_last_rank else _last_slot(dht_c, layout.num_sequences)
    dhu_out = _launch_dhu(
        arch, pre["kg"], pre["qg"], w, pre["do"], dv1, pre["gk"], layout=layout, scale=scale, dht=dht,
        return_dh0=return_dh0, walk=serial_walk,
    )
    dh_state, dv2 = dhu_out[0], dhu_out[1]
    dh0 = dhu_out[2] if return_dh0 else None
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
    if use_cp:
        walk = _serial_walk_stage("fwdh", layout.num_sequences * hv, q_norm.device, serial_walk)[0]
        LAST_CP_STATES.update({
            "h0_compressed": h0_c, "dht": dht_c, "fwdh_variant": "v2" if walk.endswith("_slices") else "v1",
            "dhu_variant": "v2" if walk.endswith("_slices") else "v1", "roles": _roles(cp_context),
            "local_lengths": list(layout.lengths),
        })
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
        "dh0": dh0,
    }


__all__ = [
    "ChunkLayout",
    "LAST_CP_STATES",
    "chunk_kda_backward",
    "chunk_layout",
    "host_boundaries",
    "state_dummy",
    "CHUNK",
    "HEAD_DIM",
]
