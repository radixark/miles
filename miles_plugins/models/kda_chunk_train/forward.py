"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Cake forward of the chunked KDA training step (``--kda-backend deterministic_cake_fwd``).

Five device launches in the stage order of the Cake composition this host mirrors
(the Cake compiler's composed forward: same kernels, same operands)::

    fwd_prep   q/k L2 norm + rstd, lower-bound safe gate cumsum ``gk``, WY inputs vb/kb/kg/qg, the layer's beta
               sigmoid, fused gather of the token rows into the chunk-aligned internal layout
    fwd_intra  Aqk / Akk (fla's storage convention, zeros above the diagonal) in token rows (+ internal copies)
    wy         u = Akk . vb,  w = Akk . kb
    fwdh       h snapshots + v_new (serial chunk walk; ``h0`` in / ``ht`` out = the fp32 state carriers)
    fwd_o      o = bf16(scale * qg . h + tril(Aqk) . v_new), scattered to the token rows

It takes the layer's **beta logits** (bf16; the sigmoid happens inside ``fwd_prep``) and returns ``o`` plus exactly
the saved set :func:`.backward.chunk_kda_backward` consumes -- ``q_norm`` / ``k_norm`` bf16 ``[B, T, H, 128]``,
``q_rstd`` / ``k_rstd`` fp32 ``[B, T, H]``, ``Aqk`` / ``Akk`` bf16 ``[B, T, HV, 64]`` -- and the fp32 ``beta``
``[B, T, HV]`` the kernels used. The sequence layout (fixed ``[B, T]`` or packed ``cu_seqlens`` + ``cu_seqlens_cpu``)
is the backward's :class:`.backward.ChunkLayout`, built once per packing and shared by both directions.
Repeated calls on identical inputs are bit-identical (no atomics, fixed-order reductions).

Host cost per call is kept to what the five launches need: the token-row operands are passed to the kernels in the
caller's own ``[B, T, ...]`` shape (the kernels take base pointers over the token rows; no per-call reshape of the
inputs) and the token-row outputs are allocated directly in that shape (no per-call view of the outputs). Only the
chunk-aligned internal buffers carry the ``[rows, HV, 128]`` layout. The forward never reads the pad rows of its
internal buffers (``fwd_o`` reads real chunks only), so it does not zero them.

Under context parallelism (``cp_context`` with a process group) the forward follows the deterministic kernels'
CP host protocol (Cake-forward mode): before the five launches it recomputes fla's forward operands of the **last local
sequence** (the A1 feed: fla's l2norm, fused beta sigmoid, gate cumsum and ``chunk_kda_fwd_intra`` in
fixed-length mode), runs fla's CP forward pre-process on them (``state_v_first=True``, one ``all_gather``)
and seeds the first local sequence's state with the exchanged compressed ``h0`` (k rows x v cols, saved as
``h0_compressed`` for the backward); the first rank of its sequence receives none.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ._jit import device_arch, kernel
from .backward import (
    CHUNK,
    HEAD_DIM,
    LAST_CP_STATES,
    _a1_forward_feed,
    _check,
    _first_slot,
    _fla_ops,
    _launch_fwdh,
    _launch_wy,
    _roles,
    _sm_count,
    _use_cp,
    chunk_layout,
)

SAVED_SET = ("q_norm", "k_norm", "q_rstd", "k_rstd", "Aqk", "Akk")
_DUMMY_BF16: dict = {}


def _dummy_bf16(device: torch.device) -> torch.Tensor:
    """Cached one-element bf16 tensor for the flag-off ``qg_out`` slot of ``fwd_prep`` (never dereferenced)."""
    hit = _DUMMY_BF16.get(device.index)
    if hit is None:
        hit = _DUMMY_BF16[device.index] = torch.zeros(1, dtype=torch.bfloat16, device=device)
    return hit


def _token_rows(name: str, t: torch.Tensor, rows: int, tail: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """A token-row operand as the kernels take it: ``dtype``, trailing dims ``tail``, exactly ``rows`` token rows in
    the caller's own leading shape (``[T, ...]`` or ``[B, T, ...]``) and contiguous (the kernels take the base
    pointer). Returns ``t`` itself when it already qualifies, else a contiguous copy."""
    if t.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {t.dtype}")
    n = len(tail)
    _check(
        t.dim() > n and tuple(t.shape[-n:]) == tail and t.numel() == rows * math.prod(tail),
        f"{name}: expected [..., {rows} token rows, {', '.join(map(str, tail))}], got {tuple(t.shape)}",
    )
    return t if t.is_contiguous() else t.contiguous()


FWD_BULK_MIN_CTAS = 256  # fwd_prep dispatch: the bulk kernel at >= 256 (chunk, head) CTAs, else the split kernel


def _launch_fwd_prep(arch, q, k, v, g_raw, beta_logits, A_log, dt_bias, *, layout, lead, lower_bound, write_qg=True):
    """Token rows ``T = layout.rows_external`` (validated by the caller): bf16 ``q``/``k`` ``[.., H, 128]`` (raw), bf16
    ``v``/``g_raw`` ``[.., HV, 128]``, bf16 ``beta_logits`` ``[.., HV]``, fp32 ``A_log`` ``[HV]`` / ``dt_bias``
    ``[HV * 128]``. Returns the saved-set tensors in token rows with the leading shape ``lead`` (``q_norm``/``k_norm``
    bf16, ``q_rstd``/``k_rstd`` fp32, ``beta`` fp32) and the internal-layout stage inputs ``gk`` fp32, ``vb``/``kb``/``kg``
    (+ ``qg``) bf16 ``[R, HV, 128]``."""
    h, hv = q.shape[-2], v.shape[-2]
    _check(
        hv % h == 0 and A_log.shape == (hv,) and dt_bias.shape == (hv * HEAD_DIM,),
        "fwd_prep: expected HV a multiple of H, [HV] A_log, [HV*128] dt_bias",
    )
    _check(hv * HEAD_DIM * CHUNK < 2**31, "fwd_prep: HV * 128 * 64 must fit a 32-bit row offset")
    dev, bf, f32 = v.device, torch.bfloat16, torch.float32
    rows = layout.rows
    out = {
        "q_norm": torch.empty(*lead, h, HEAD_DIM, dtype=bf, device=dev),
        "k_norm": torch.empty(*lead, h, HEAD_DIM, dtype=bf, device=dev),
        "q_rstd": torch.empty(*lead, h, dtype=f32, device=dev),
        "k_rstd": torch.empty(*lead, h, dtype=f32, device=dev),
        "beta": torch.empty(*lead, hv, dtype=f32, device=dev),
        "gk": torch.empty(rows, hv, HEAD_DIM, dtype=f32, device=dev),
    }
    # separate allocations (not one carved block): the composition frees vb / kb right after ``wy`` and kg after
    # ``fwdh`` so the stream-ordered caching allocator can hand their blocks to v_new / o (U1-M peak memory)
    for name in ("vb", "kb", "kg"):
        out[name] = torch.empty(rows, hv, HEAD_DIM, dtype=bf, device=dev)
    if write_qg:
        out["qg"] = torch.empty(rows, hv, HEAD_DIM, dtype=bf, device=dev)
        qg_ptr = out["qg"]
    else:
        qg_ptr = _dummy_bf16(dev)
    # dispatch: the bulk kernel at >= FWD_BULK_MIN_CTAS (chunk, head) CTAs, else the split kernel (one CTA per
    # 32-token half-chunk and head); same arguments, same outputs
    if layout.num_chunks_padded * hv >= FWD_BULK_MIN_CTAS:
        stage, grid = "fwd_prep", (layout.num_chunks_padded, hv, 1)
    else:
        stage, grid = "fwd_prep_split", (2 * layout.num_chunks_padded, hv, 1)
    kernel(stage, arch).launch(
        grid=grid,
        q=q,
        k=k,
        v=v,
        g_raw=g_raw,
        beta_logits=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        chunk_bos=layout.chunk_bos,
        chunk_len=layout.chunk_len,
        q_norm_out=out["q_norm"],
        k_norm_out=out["k_norm"],
        q_rstd_out=out["q_rstd"],
        k_rstd_out=out["k_rstd"],
        beta_out=out["beta"],
        gk_out=out["gk"],
        vb_out=out["vb"],
        kb_out=out["kb"],
        kg_out=out["kg"],
        qg_out=qg_ptr,
        num_qk_heads=h,
        num_heads=hv,
        group=hv // h,
        write_qg=1 if write_qg else 0,
        lower_bound=float(lower_bound),
    )
    return out


def _launch_fwd_intra(arch, q_norm, k_norm, gk, beta, *, layout, lead, scale):
    """Token rows: bf16 ``q_norm``/``k_norm`` ``[.., H, 128]``, fp32 ``beta`` ``[.., HV]`` (the ``fwd_prep`` outputs);
    internal fp32 ``gk`` ``[R, HV, 128]``. Returns ``(aqk, akk, aqk_int, akk_int)``: bf16 ``[.., HV, 64]`` saved-set
    tensors in token rows (leading shape ``lead``) and the internal-layout copy of ``akk`` with zero pad rows (``akk``
    itself when the layout is the identity) for ``wy``. ``aqk_int`` is always ``None``: the forward reads ``aqk`` from
    the token rows (``fwd_o``), so the kernel does not produce an internal copy of it."""
    rows, hv, _ = gk.shape
    h = q_norm.shape[-2]
    _check(rows == layout.rows, "fwd_intra: gk does not match the layout's internal rows")
    dev, bf = gk.device, torch.bfloat16
    aqk = torch.empty(*lead, hv, CHUNK, dtype=bf, device=dev)
    akk = torch.empty(*lead, hv, CHUNK, dtype=bf, device=dev)
    if layout.identity:
        akk_int, write_int = akk, 0
    else:
        akk_int = torch.empty(rows, hv, CHUNK, dtype=bf, device=dev)
        write_int = 1
    kernel("fwd_intra", arch).launch(
        grid=(layout.num_chunks_padded, hv, 1),
        q_norm=q_norm,
        k_norm=k_norm,
        gk=gk,
        beta=beta,
        chunk_bos=layout.chunk_bos,
        chunk_len=layout.chunk_len,
        aqk_ext=aqk,
        akk_ext=akk,
        akk_int=akk_int,
        num_heads=hv,
        num_qk_heads=h,
        group=hv // h,
        write_int=write_int,
        scale=float(scale),
    )
    return aqk, akk, None, akk_int


def _launch_fwd_o(arch, qg, h, aqk_ext, v_new, *, layout, lead, scale):
    """``qg``/``v_new``: internal-layout bf16 ``[R, HV, 128]``; ``h``: bf16 ``[num_chunks, HV, 128, 128]`` (the state
    entering each real chunk); ``aqk_ext``: token-row bf16 ``[.., HV, 64]``. Returns ``o`` bf16 ``[.., HV, 128]`` in
    the token rows with the leading shape ``lead`` (pad rows are never written). Reads real chunks only: the pad
    rows of ``qg`` / ``v_new`` are never touched."""
    rows, hv, kd = qg.shape
    nc = layout.num_chunks
    _check(
        kd == HEAD_DIM and rows == layout.rows and v_new.shape == qg.shape and tuple(h.shape) == (nc, hv, HEAD_DIM, HEAD_DIM),
        "fwd_o: expected internal [R,HV,128] qg/v_new and [NC,HV,128,128] h",
    )
    o = torch.empty(*lead, hv, HEAD_DIM, dtype=torch.bfloat16, device=qg.device)
    # persistent: one CTA per SM grid-striding over the nc * hv (chunk, head) tiles
    kernel("fwd_o", arch).launch(
        grid=(max(1, min(nc * hv, _sm_count(qg.device))), 1, 1),
        qg_tma=qg,
        h_tma=h.view(nc * hv * HEAD_DIM, HEAD_DIM),
        aqk_tma=aqk_ext,
        vn_tma=v_new,
        o_out=o,
        chunk_bos=layout.chunk_bos,
        chunk_len=layout.chunk_len,
        num_heads=hv,
        scale=float(scale),
        num_chunks=nc,
    )
    return o


def chunk_kda_forward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | Sequence[int] | None = None,
    h0: torch.Tensor | None = None,
    output_final_state: bool = False,
    cp_context=None,
    serial_walk: str = "auto",
) -> dict[str, torch.Tensor | None]:
    """Composed Cake forward (five launches).

    - ``q``, ``k``: raw (pre-l2norm) bf16 ``[B, T, H, 128]``; ``v``, ``g``: bf16 ``[B, T, HV, 128]`` (``g`` = raw
      gate pre-activations); ``beta_logits``: bf16 ``[B, T, HV]`` (the layer's pre-sigmoid beta); ``A_log`` fp32
      ``[HV]``; ``dt_bias`` fp32 ``[HV * 128]``; ``scale``; ``lower_bound`` (negative). Contiguous operands are passed
      to the kernels as they are (no per-call copy or reshape); a non-contiguous one is copied once.
    - ``cu_seqlens`` / ``cu_seqlens_cpu``: the packed boundaries as :func:`.backward.chunk_kda_backward` takes them
      (``B == 1``; the host copy keeps the op free of device-to-host copies).
    - ``h0``: optional fp32 ``[N, HV, 128, 128]`` initial state per sequence (``None`` = zeros);
      ``output_final_state`` returns the fp32 state after each sequence's last chunk as ``ht``.
    - ``cp_context``: fla ``FLACPContext`` with a process group (context parallelism; ``h0`` /
      ``output_final_state`` are then rejected, as fla does): the CP host protocol of the module docstring runs and
      the exchanged compressed state is returned as ``h0_compressed`` (fp32 ``[1, HV, 128, 128]`` k rows x v cols)
      for :func:`.backward.chunk_kda_backward`. ``serial_walk``: ``"auto"`` | ``"v1"`` | ``"v2"`` (fwdh kernel pin).

    Returns ``o`` bf16 ``[B, T, HV, 128]``, the saved set (``q_norm``, ``k_norm``, ``q_rstd``, ``k_rstd``, ``Aqk``,
    ``Akk``; shapes and dtypes as the backward's docstring, every one a fresh contiguous kernel output allocated in
    that shape), ``beta`` fp32 ``[B, T, HV]``, ``ht`` (or ``None``) and ``h0_compressed`` (``None`` outside CP).
    """
    arch = device_arch(v.device)
    batch0, seq0, h, kd = q.shape
    hv = v.shape[2]
    _check(kd == HEAD_DIM and v.shape[-1] == HEAD_DIM, "K = V = 128 is required")
    _check(hv % h == 0, "HV must be a multiple of H")
    layout = chunk_layout(batch0, seq0, cu_seqlens, q.device, cu_seqlens_cpu=cu_seqlens_cpu)
    rows = layout.rows_external
    lead = (batch0, seq0)
    bf, f32 = torch.bfloat16, torch.float32
    q_r = _token_rows("q", q, rows, (h, HEAD_DIM), bf)
    k_r = _token_rows("k", k, rows, (h, HEAD_DIM), bf)
    v_r = _token_rows("v", v, rows, (hv, HEAD_DIM), bf)
    g_r = _token_rows("g", g, rows, (hv, HEAD_DIM), bf)
    beta_r = _token_rows("beta_logits", beta_logits, rows, (hv,), bf)
    if A_log.dtype != f32 or not A_log.is_contiguous():
        A_log = A_log.to(f32).contiguous()
    if dt_bias.dtype != f32 or not dt_bias.is_contiguous():
        dt_bias = dt_bias.to(f32).contiguous()

    # CP (host protocol, Cake-forward mode): fla's forward pre-process on the LAST local sequence (external rows
    # [s, T), fed with fla's own recompute of its operands) exchanges the compressed state entering the FIRST local
    # sequence; its two-entry cu_seqlens is the layout's (uploaded with the tables); state_v_first=True as the fla path
    # tunes those kernels (Triton autotune keys ignore the constexpr), one host transpose.
    use_cp = _use_cp(cp_context)
    _check(
        not use_cp or (h0 is None and not output_final_state),
        "h0 / output_final_state are not supported under context parallelism",
    )
    h0_c = None
    if use_cp:
        fl = _fla_ops()
        start = layout.offsets[-1]
        feed = _a1_forward_feed(
            fl, q_r[:, start:], k_r[:, start:], v_r[:, start:], g_r[:, start:], beta_r[:, start:],
            A_log, dt_bias, scale=scale, lower_bound=lower_bound,
        )
        h0_vf = fl["fwd_h_pre_process"](
            k=feed["kg"], w=feed["w"], u=feed["u"], gk=feed["g"], cu_seqlens=layout.cp_cu_last, initial_state=None,
            context=cp_context, chunk_size=CHUNK, state_v_first=True,
        )
        del feed
        h0_c = h0_vf.transpose(-1, -2).contiguous()
        h0 = None if cp_context.is_first_rank else _first_slot(h0_c, layout.num_sequences)
        LAST_CP_STATES.clear()
        LAST_CP_STATES.update({"h0_compressed": h0_c, "roles": _roles(cp_context), "forward_source": "cake"})

    pre = _launch_fwd_prep(arch, q_r, k_r, v_r, g_r, beta_r, A_log, dt_bias, layout=layout, lead=lead, lower_bound=lower_bound)
    aqk, akk, _aqk_int, akk_int = _launch_fwd_intra(arch, pre["q_norm"], pre["k_norm"], pre["gk"], pre["beta"], layout=layout, lead=lead, scale=scale)
    u, w = _launch_wy(arch, akk_int, pre["vb"], pre["kb"])
    del pre["vb"], pre["kb"]  # forward transients die with their consumer; the kernels launch on the current stream
    # the forward never reads v_new's pad rows (fwd_o walks real chunks only): no pad fill in this direction
    res = _launch_fwdh(
        arch, w, pre["kg"], u, pre["gk"], layout=layout, h0=h0, return_ht=output_final_state, walk=serial_walk, zero_pad_rows=False
    )
    h_state, v_new = res[0], res[1]
    ht = res[2] if output_final_state else None
    del pre["kg"], pre["gk"], akk_int, u, w  # w / u are recomputed by the backward from the saved set (fla rule)
    o = _launch_fwd_o(arch, pre["qg"], h_state, aqk, v_new, layout=layout, lead=lead, scale=scale)
    # every tensor below was allocated in the caller's [B, T, ...] shape by its stage: no view per call
    return {
        "o": o,
        "q_norm": pre["q_norm"],
        "k_norm": pre["k_norm"],
        "q_rstd": pre["q_rstd"],
        "k_rstd": pre["k_rstd"],
        "Aqk": aqk,
        "Akk": akk,
        "beta": pre["beta"],
        "ht": ht,
        "h0_compressed": h0_c,
    }


__all__ = ["SAVED_SET", "chunk_kda_forward"]
