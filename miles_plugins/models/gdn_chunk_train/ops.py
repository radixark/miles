"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Deterministic ``chunk_gated_delta_rule`` (forward + backward) on the generated kernels (SM100a / SM103a).

Drop-in replacement for ``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` (FLA v0.5.2) as the
Qwen GDN wrappers call it: ``[B, T, H, K]`` inputs, packed varlen through ``cu_seqlens``, in-kernel
q/k L2 normalisation, optional FP32 initial/final state, and context-parallel state passing through
an ``FLACPContext``.  Both passes are bit-deterministic call to call: no atomics, fixed reduction
order, and a launch geometry that is a pure function of the shape and the device.

The kernels mirror FLA's chunk-64 decomposition one to one (prep = L2 norm + gate cumsum, WY solve,
state scan, output; backward: dv_local, adjoint state scan, dq/dk/dw/dg, WY pullback, finalize) and
recompute the transient WY/state tensors exactly in the backward, as FLA does.  ``K = V = 128``,
``HV % H == 0`` (grouped value heads without materialised ``repeat_interleave``), BF16 q/k/v.

Context parallelism reuses FLA v0.5.2's ``fla.ops.cp`` pre-processing exactly
(``chunk_gated_delta_rule_fwd_h_pre_process``, ``compress_h0``, ``expand_h0``,
``chunk_gated_delta_rule_bwd_dhu_pre_process``), so ``build_gdn_cp_context`` semantics are
unchanged.  Those helpers are only imported when a ``cp_context`` is supplied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch

from ._jit import MODULES, device_arch, kernel

__all__ = ["ChunkGatedDeltaRuleFunction", "ChunkMeta", "chunk_gated_delta_rule", "gdn_chunk_backward", "gdn_chunk_forward"]

CHUNK = 64
HEAD_DIM = 128
# Register-scan configurations (value rows per CTA, K split) in preference order; every warp owns
# 16 value rows and K / ksplit keys.  Must match the exported ``fwd_h_r*``/``dhu_r*`` stages.
REG_SCAN_CONFIGS = ((32, 2), (32, 1), (64, 1))
# Tests may pin a scan configuration; production always derives it from shape and device.
FORCED_SCAN_CONFIG: tuple[int, int] | None = None


# --------------------------------------------------------------------------- #
# chunk table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChunkMeta:
    """Chunk table for packed varlen sequences (all int32 CUDA tensors).

    ``num_chunks`` is the launch extent; when the table is built on the device from ``cu_seqlens``
    it is an upper bound (``T // 64 + num_seqs``) and the trailing entries are padding chunks with
    ``chunk_len == 0`` that every chunk kernel skips.
    """

    chunk_start: torch.Tensor
    chunk_len: torch.Tensor
    seq_chunk_start: torch.Tensor
    num_chunks: int
    num_seqs: int
    seq_lens: tuple[int, ...] | None


def build_chunk_meta(seq_lens, device) -> ChunkMeta:
    starts: list[int] = []
    lens: list[int] = []
    seq_chunk_start = [0]
    token = 0
    for seq_len in seq_lens:
        seq_len = int(seq_len)
        for chunk_start in range(0, seq_len, CHUNK):
            starts.append(token + chunk_start)
            lens.append(min(CHUNK, seq_len - chunk_start))
        seq_chunk_start.append(len(starts))
        token += seq_len

    def to_dev(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    return ChunkMeta(
        chunk_start=to_dev(starts),
        chunk_len=to_dev(lens),
        seq_chunk_start=to_dev(seq_chunk_start),
        num_chunks=len(starts),
        num_seqs=len(seq_lens),
        seq_lens=tuple(int(x) for x in seq_lens),
    )


def build_chunk_meta_from_cu_seqlens(cu_seqlens: torch.Tensor, total_tokens: int, arch: str) -> ChunkMeta:
    """Chunk table from a device ``cu_seqlens`` tensor without synchronising the host."""
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a 1-D tensor with at least two entries")
    num_seqs = int(cu_seqlens.numel()) - 1
    num_chunks_max = total_tokens // CHUNK + num_seqs
    cu32 = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    cu32 = cu32.contiguous()
    device = cu_seqlens.device
    chunk_start = torch.empty(num_chunks_max, dtype=torch.int32, device=device)
    chunk_len = torch.empty(num_chunks_max, dtype=torch.int32, device=device)
    seq_chunk_start = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
    kernel("meta", arch).launch(
        grid=(1, 1, 1),
        cu_seqlens=cu32,
        chunk_start=chunk_start,
        chunk_len=chunk_len,
        seq_chunk_start=seq_chunk_start,
        num_seqs=num_seqs,
        num_chunks_max=num_chunks_max,
        total_tokens=int(total_tokens),
    )
    return ChunkMeta(
        chunk_start=chunk_start,
        chunk_len=chunk_len,
        seq_chunk_start=seq_chunk_start,
        num_chunks=num_chunks_max,
        num_seqs=num_seqs,
        seq_lens=None,
    )


# --------------------------------------------------------------------------- #
# scan configuration (a pure function of shape and device)
# --------------------------------------------------------------------------- #
@cache
def _device_props(device_index: int) -> tuple[int, int, int]:
    props = torch.cuda.get_device_properties(device_index)
    return int(props.multi_processor_count), int(props.shared_memory_per_block_optin), int(props.major)


def _scan_record(kind: str, vb: int, ks: int, arch: str) -> dict:
    return MODULES[f"{kind}_r{vb}k{ks}"][arch]


def choose_register_scan_config(kind: str, num_seqs: int, num_v_heads: int, device_index: int, arch: str) -> tuple[int, int]:
    """Register-scan (value rows per CTA, K split) whose grid is fully co-resident on the device.

    The preferred configuration (32 rows, K split in two -> four warps) has the shortest per-chunk
    chain and fills all four SM sub-partitions; wider blocks are used only when the preferred grid
    would need more than one wave.  The choice depends only on shape and device, never on data.
    """
    if FORCED_SCAN_CONFIG is not None:
        return FORCED_SCAN_CONFIG
    sm_count, smem_optin, _major = _device_props(device_index)
    for vb, ks in REG_SCAN_CONFIGS:
        ctas = num_seqs * num_v_heads * (HEAD_DIM // vb)
        threads = 32 * (vb // 16) * ks
        smem = int(_scan_record(kind, vb, ks, arch)["scan"]["select_smem_bytes"])
        resident = min(2048 // threads, 65536 // (256 * threads), smem_optin // smem, 4)
        if resident >= 1 and ctas <= sm_count * resident:
            return vb, ks
    return 64, 1


def _scan_stage(kind: str, num_seqs: int, num_v_heads: int, device: torch.device, arch: str) -> tuple[str, int]:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    vb, ks = choose_register_scan_config(kind, num_seqs, num_v_heads, device_index, arch)
    _sm_count, smem_optin, major = _device_props(device_index)
    record = _scan_record(kind, vb, ks, arch)
    if major < 9 or int(record["scan"]["select_smem_bytes"]) > smem_optin:
        raise NotImplementedError(f"the register-resident {kind} scan does not fit device {device_index} (only Blackwell is supported)")
    return f"{kind}_r{vb}k{ks}", vb


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
def _launch_wy(arch, kn, v, g_cs, beta32, A, meta: ChunkMeta, num_heads: int, num_v_heads: int, *, recompute: bool):
    w = torch.empty(kn.shape[0], num_v_heads, HEAD_DIM, dtype=kn.dtype, device=kn.device)
    u = torch.empty_like(v)
    kernel("wy", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        kn=kn,
        v=v,
        g_cs=g_cs,
        beta=beta32,
        A_out=A,
        w_out=w,
        u_out=u,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        recompute=1 if recompute else 0,
    )
    return w, u


def _launch_fwd_h(arch, kn, w, u, g_cs, initial_state, meta: ChunkMeta, num_heads: int, num_v_heads: int, *, store_final_state: bool):
    device = kn.device
    h = torch.empty(meta.num_chunks, num_v_heads, HEAD_DIM, HEAD_DIM, dtype=kn.dtype, device=device)
    v_new = torch.empty_like(u)
    # The kernel writes every element of ``final_state`` when requested; no fill kernel is needed.
    final_state = torch.empty(meta.num_seqs if store_final_state else 0, num_v_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device)
    h0 = initial_state.contiguous().float() if initial_state is not None else final_state
    stage, vb = _scan_stage("fwd_h", meta.num_seqs, num_v_heads, device, arch)
    kernel(stage, arch).launch(
        grid=(meta.num_seqs, num_v_heads, HEAD_DIM // vb),
        kn=kn,
        w=w,
        u=u,
        g_cs=g_cs,
        h0=h0,
        h_out=h,
        v_new=v_new,
        final_state=final_state,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        seq_chunk_start=meta.seq_chunk_start,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        use_initial_state=1 if initial_state is not None else 0,
        store_final_state=1 if store_final_state else 0,
    )
    return h, v_new, final_state


def gdn_chunk_forward_stage1(q, k, v, g, beta, *, meta: ChunkMeta, scale: float | None = None, normalize_qk: bool = True) -> dict[str, Any]:
    """Prep (l2norm, gate cumsum) and WY (A, w, u); no state is touched yet."""
    total_tokens, num_heads, key_dim = q.shape
    num_v_heads, value_dim = v.shape[1], v.shape[2]
    if key_dim != HEAD_DIM or value_dim != HEAD_DIM:
        raise ValueError("gdn_chunk_forward requires K = V = 128")
    if num_v_heads % num_heads != 0:
        raise ValueError("num_v_heads must be a multiple of num_heads")
    if scale is None:
        scale = 1.0 / math.sqrt(key_dim)
    device = q.device
    arch = device_arch(device)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g32 = g.contiguous().float()
    beta = beta.contiguous()
    # BF16 beta is converted inside the prep kernel (no separate cast kernel); FP32 passes through.
    beta_bf16 = beta.dtype == torch.bfloat16
    beta32 = torch.empty(total_tokens, num_v_heads, dtype=torch.float32, device=device) if beta_bf16 else beta.float()

    qn = torch.empty_like(q)
    kn = torch.empty_like(k)
    rstd_q = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=device)
    rstd_k = torch.empty_like(rstd_q)
    g_cs = torch.empty(total_tokens, num_v_heads, dtype=torch.float32, device=device)
    kernel("prep_bf16beta" if beta_bf16 else "prep", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        q=q,
        k=k,
        g=g32,
        beta=beta,
        beta32_out=beta32,
        qn=qn,
        kn=kn,
        rstd_q=rstd_q,
        rstd_k=rstd_k,
        g_cs=g_cs,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        normalize_qk=1 if normalize_qk else 0,
    )
    A = torch.empty(total_tokens, num_v_heads, CHUNK, dtype=q.dtype, device=device)  # fully written by the WY kernel
    w, u = _launch_wy(arch, kn, v, g_cs, beta32, A, meta, num_heads, num_v_heads, recompute=False)
    return {
        "arch": arch,
        "normalize_qk": normalize_qk,
        "num_heads": num_heads,
        "num_v_heads": num_v_heads,
        "qn": qn,
        "kn": kn,
        "rstd_q": rstd_q,
        "rstd_k": rstd_k,
        "g_cs": g_cs,
        "A": A,
        "w": w,
        "u": u,
        "v": v,
        "beta32": beta32,
        "meta": meta,
        "scale": float(scale),
    }


def gdn_chunk_forward_stage2(stage1: dict[str, Any], *, initial_state=None, output_final_state: bool = False) -> dict[str, Any]:
    """State scan and output; extends the stage-1 dict in place."""
    arch = stage1["arch"]
    meta: ChunkMeta = stage1["meta"]
    kn, qn, w, u, g_cs, v = (stage1[name] for name in ("kn", "qn", "w", "u", "g_cs", "v"))
    num_heads, num_v_heads = stage1["num_heads"], stage1["num_v_heads"]
    h, v_new, final_state = _launch_fwd_h(arch, kn, w, u, g_cs, initial_state, meta, num_heads, num_v_heads, store_final_state=output_final_state)
    o = torch.empty_like(v)
    kernel("fwd_o", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        qn=qn,
        kn=kn,
        v_new=v_new,
        h=h,
        g_cs=g_cs,
        o_out=o,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        scale=float(stage1["scale"]),
    )
    stage1.update({"output": o, "final_state": final_state if output_final_state else None, "initial_state": initial_state, "h": h, "v_new": v_new})
    return stage1


def gdn_chunk_forward(q, k, v, g, beta, *, meta: ChunkMeta, scale: float | None = None, initial_state=None, output_final_state: bool = False, normalize_qk: bool = True) -> dict[str, Any]:
    """Run the deterministic chunked forward; returns outputs and the tape ``gdn_chunk_backward`` needs."""
    stage1 = gdn_chunk_forward_stage1(q, k, v, g, beta, meta=meta, scale=scale, normalize_qk=normalize_qk)
    return gdn_chunk_forward_stage2(stage1, initial_state=initial_state, output_final_state=output_final_state)


# --------------------------------------------------------------------------- #
# backward
# --------------------------------------------------------------------------- #
def gdn_chunk_backward(forward: dict[str, Any], do, dht=None, *, initial_state=None, dht_from_dv_local=None, dbeta_dtype=None) -> dict[str, Any]:
    """Run the deterministic chunked backward from the (possibly pruned) forward tape.

    ``forward`` must carry ``arch``, ``qn``, ``kn``, ``rstd_q``, ``rstd_k``, ``v``, ``g_cs``, ``beta32``,
    ``A``, ``meta``, ``scale`` and ``normalize_qk``.  ``w``/``u``/``h``/``v_new`` are recomputed exactly
    (same kernels, same order) when the tape does not carry them.  ``initial_state`` is the FP32 state
    the forward started from (``None`` for a zero state); its gradient is returned only when it is
    given.  ``dht_from_dv_local(w, dv_local)`` optionally supplies the final-state gradient after
    ``dv_local`` exists (context-parallel state passing).
    """
    arch = forward["arch"]
    meta: ChunkMeta = forward["meta"]
    qn, kn = forward["qn"], forward["kn"]
    A = forward["A"]
    g_cs = forward["g_cs"]
    scale = float(forward["scale"])
    v = forward["v"]
    beta32 = forward["beta32"]
    total_tokens, num_heads, _ = qn.shape
    num_v_heads = v.shape[1]
    device = qn.device
    do = do.contiguous()
    if initial_state is None:
        initial_state = forward.get("initial_state")
    has_initial_state = initial_state is not None
    if forward.get("w") is None or forward.get("u") is None:
        w, u = _launch_wy(arch, kn, v, g_cs, beta32, A, meta, num_heads, num_v_heads, recompute=True)
    else:
        w, u = forward["w"], forward["u"]
    if forward.get("h") is None or forward.get("v_new") is None:
        h, v_new, _ = _launch_fwd_h(arch, kn, w, u, g_cs, initial_state, meta, num_heads, num_v_heads, store_final_state=False)
    else:
        h, v_new = forward["h"], forward["v_new"]

    dv_local = torch.empty_like(v)
    kernel("dv_local", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        qn=qn,
        kn=kn,
        do=do,
        g_cs=g_cs,
        dv_local=dv_local,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        scale=scale,
    )

    if dht_from_dv_local is not None:
        dht = dht_from_dv_local(w, dv_local)
    dh = torch.empty_like(h)
    dv2 = torch.empty_like(v)
    # Written in full by the scan when requested; no fill kernel.
    dh0 = torch.empty(meta.num_seqs if has_initial_state else 0, num_v_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device)
    dht32 = dht.contiguous().float() if dht is not None else dh0
    scan_stage, vb = _scan_stage("dhu", meta.num_seqs, num_v_heads, device, arch)
    kernel(scan_stage, arch).launch(
        grid=(meta.num_seqs, num_v_heads, HEAD_DIM // vb),
        qn=qn,
        kn=kn,
        w=w,
        do=do,
        dv_local=dv_local,
        g_cs=g_cs,
        dht=dht32,
        dh_out=dh,
        dv2=dv2,
        dh0=dh0,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        seq_chunk_start=meta.seq_chunk_start,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        use_final_state_grad=1 if dht is not None else 0,
        store_initial_state_grad=1 if has_initial_state else 0,
        scale=scale,
    )

    dq_hv = torch.empty(total_tokens, num_v_heads, HEAD_DIM, dtype=qn.dtype, device=device)
    dk_hv = torch.empty_like(dq_hv)
    dw = torch.empty_like(w)
    dg1 = torch.empty(total_tokens, num_v_heads, dtype=torch.float32, device=device)
    kernel("dqkwg", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        qn=qn,
        kn=kn,
        v_new=v_new,
        do=do,
        dv2=dv2,
        h=h,
        dh=dh,
        g_cs=g_cs,
        dq_out=dq_hv,
        dk_out=dk_hv,
        dw_out=dw,
        dg_out=dg1,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        scale=scale,
    )

    dk2 = torch.empty_like(dq_hv)
    dv = torch.empty_like(v)
    # ``dbeta`` is produced directly in the caller's dtype (BF16 or FP32); no cast kernel follows.
    dbeta_bf16 = dbeta_dtype == torch.bfloat16
    dbeta = torch.empty(total_tokens, num_v_heads, dtype=torch.bfloat16 if dbeta_bf16 else torch.float32, device=device)
    dg2 = torch.empty(total_tokens, num_v_heads, dtype=torch.float32, device=device)
    kernel("wy_bwd_bf16" if dbeta_bf16 else "wy_bwd", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        kn=kn,
        v=v,
        A=A,
        dw=dw,
        du=dv2,
        g_cs=g_cs,
        beta=beta32,
        dk2_out=dk2,
        dv_out=dv,
        dbeta_out=dbeta,
        dg2_out=dg2,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
    )

    dq = torch.empty_like(qn)
    dk = torch.empty_like(kn)
    dg = torch.empty_like(dg2)
    kernel("finalize", arch).launch(
        grid=(meta.num_chunks, num_v_heads, 1),
        qn=qn,
        kn=kn,
        rstd_q=forward["rstd_q"],
        rstd_k=forward["rstd_k"],
        dq_hv=dq_hv,
        dk_hv=dk_hv,
        dk2=dk2,
        dg1=dg1,
        dg2=dg2,
        dq_out=dq,
        dk_out=dk,
        dg_out=dg,
        chunk_start=meta.chunk_start,
        chunk_len=meta.chunk_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        normalize_qk=1 if forward.get("normalize_qk", True) else 0,
    )
    return {"dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta, "dinitial_state": dh0 if has_initial_state else None}


# --------------------------------------------------------------------------- #
# autograd wrapper with FLA's signature
# --------------------------------------------------------------------------- #
def _chunk_meta(cu_seqlens, cu_seqlens_cpu, batch: int, seq_len: int, device, arch: str) -> ChunkMeta:
    """Chunk table without a device-to-host synchronisation.

    Equal-length batches and a caller-supplied host copy of ``cu_seqlens`` build the exact table from
    host integers; a device-only ``cu_seqlens`` builds it on the GPU (padded launch extent).
    """
    if cu_seqlens is None:
        return build_chunk_meta([seq_len] * batch, device)
    if batch != 1:
        raise ValueError("cu_seqlens requires a leading batch dimension of 1")
    if cu_seqlens_cpu is not None:
        values = [int(x) for x in cu_seqlens_cpu.tolist()]
        return build_chunk_meta([b - a for a, b in zip(values[:-1], values[1:], strict=True)], device)
    return build_chunk_meta_from_cu_seqlens(cu_seqlens, batch * seq_len, arch)


def _cp_modules():
    from fla.ops.cp.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu_pre_process,
        chunk_gated_delta_rule_fwd_h_pre_process,
        compress_h0,
        expand_h0,
    )

    return (
        chunk_gated_delta_rule_fwd_h_pre_process,
        chunk_gated_delta_rule_bwd_dhu_pre_process,
        compress_h0,
        expand_h0,
    )


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """Deterministic chunked gated delta rule with FLA's autograd contract."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None,
        cu_seqlens_cpu: torch.Tensor | None,
        use_qk_l2norm_in_kernel: bool,
        cp_context: Any,
    ):
        batch, seq_len, num_heads, key_dim = q.shape
        num_v_heads, value_dim = v.shape[2], v.shape[3]
        arch = device_arch(q.device)

        def flat(t, heads, dim):
            return t.reshape(batch * seq_len, heads, dim)

        q_flat, k_flat = flat(q, num_heads, key_dim), flat(k, num_heads, key_dim)
        v_flat = flat(v, num_v_heads, value_dim)
        g_flat = g.reshape(batch * seq_len, num_v_heads)
        beta_flat = beta.reshape(batch * seq_len, num_v_heads)
        meta = _chunk_meta(cu_seqlens, cu_seqlens_cpu, batch, seq_len, q.device, arch)

        use_cp = cp_context is not None and getattr(cp_context, "group", None) is not None
        if use_cp and initial_state is not None:
            raise ValueError("chunk_gated_delta_rule: initial_state must be None under context parallelism")

        # prep + WY first; CP derives this rank's incoming state from w/u
        fwd: dict[str, Any]
        if use_cp:
            fwd_h_pre, _bwd_pre, compress_h0, _expand_h0 = _cp_modules()
            fwd = gdn_chunk_forward_stage1(q_flat, k_flat, v_flat, g_flat, beta_flat, meta=meta, scale=scale, normalize_qk=use_qk_l2norm_in_kernel)
            initial_state = fwd_h_pre(
                k=fwd["kn"].unsqueeze(0),
                w=fwd["w"].unsqueeze(0),
                u=fwd["u"].unsqueeze(0),
                g=fwd["g_cs"].unsqueeze(0),
                cu_seqlens=cu_seqlens,
                initial_state=None,
                context=cp_context,
                state_v_first=False,
                chunk_size=CHUNK,
            )
            fwd = gdn_chunk_forward_stage2(fwd, initial_state=initial_state, output_final_state=output_final_state)
            saved_state = compress_h0(initial_state, context=cp_context)
        else:
            fwd = gdn_chunk_forward(
                q_flat,
                k_flat,
                v_flat,
                g_flat,
                beta_flat,
                meta=meta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                normalize_qk=use_qk_l2norm_in_kernel,
            )
            saved_state = initial_state

        ctx.save_for_backward(fwd["qn"], fwd["kn"], fwd["rstd_q"], fwd["rstd_k"], fwd["v"], fwd["g_cs"], fwd["beta32"], fwd["A"], saved_state, cu_seqlens)
        ctx.meta = meta
        ctx.arch = arch
        ctx.scale = float(scale)
        ctx.shapes = (batch, seq_len, num_heads, num_v_heads, key_dim, value_dim)
        ctx.normalize_qk = bool(use_qk_l2norm_in_kernel)
        ctx.cp_context = cp_context if use_cp else None
        ctx.has_initial_state_input = initial_state is not None and not use_cp
        ctx.dtypes = (g.dtype, beta.dtype)
        output = fwd["output"].reshape(batch, seq_len, num_v_heads, value_dim)
        final_state = fwd["final_state"] if output_final_state else None
        return output.to(q.dtype), final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor | None):
        qn, kn, rstd_q, rstd_k, v_flat, g_cs, beta32, A, saved_state, cu_seqlens = ctx.saved_tensors
        batch, seq_len, num_heads, num_v_heads, key_dim, value_dim = ctx.shapes
        g_dtype, beta_dtype = ctx.dtypes
        do_flat = do.reshape(batch * seq_len, num_v_heads, value_dim).contiguous()
        tape = {
            "arch": ctx.arch,
            "qn": qn,
            "kn": kn,
            "rstd_q": rstd_q,
            "rstd_k": rstd_k,
            "v": v_flat,
            "g_cs": g_cs,
            "beta32": beta32,
            "A": A,
            "meta": ctx.meta,
            "scale": ctx.scale,
            "normalize_qk": ctx.normalize_qk,
        }
        cp_context = ctx.cp_context
        initial_state = saved_state
        dht_hook = None
        if cp_context is not None:
            _fwd_pre, bwd_dhu_pre, _compress_h0, expand_h0 = _cp_modules()
            initial_state = expand_h0(saved_state, context=cp_context)
            if dht is not None:
                raise ValueError("chunk_gated_delta_rule: final-state gradients are not part of the CP contract")

            def dht_hook(w, dv_local):
                dht_local, _ = bwd_dhu_pre(
                    q=qn.unsqueeze(0),
                    k=kn.unsqueeze(0),
                    w=w.unsqueeze(0),
                    do=do_flat.unsqueeze(0),
                    dv=dv_local.unsqueeze(0),
                    g=g_cs.unsqueeze(0),
                    scale=ctx.scale,
                    cu_seqlens=cu_seqlens,
                    dht=None,
                    initial_state=initial_state,
                    context=cp_context,
                    state_v_first=False,
                    chunk_size=CHUNK,
                )
                return dht_local

        grads = gdn_chunk_backward(tape, do_flat, dht, initial_state=initial_state, dht_from_dv_local=dht_hook, dbeta_dtype=beta_dtype)
        dq = grads["dq"].reshape(batch, seq_len, num_heads, key_dim)
        dk = grads["dk"].reshape(batch, seq_len, num_heads, key_dim)
        dv = grads["dv"].reshape(batch, seq_len, num_v_heads, value_dim)
        dg = grads["dg"].reshape(batch, seq_len, num_v_heads).to(g_dtype)  # no-op for the FP32 gate
        dbeta = grads["dbeta"].reshape(batch, seq_len, num_v_heads).to(beta_dtype)  # no-op: produced in beta's dtype
        dh0 = grads["dinitial_state"] if ctx.has_initial_state_input else None
        return dq, dk, dv, dg, dbeta, None, dh0, None, None, None, None, None


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cp_context: Any = None,
    chunk_size: int = 64,
    **unsupported: Any,
):
    """FLA v0.5.2 ``chunk_gated_delta_rule`` signature on the deterministic kernels.

    Supported: ``[B, T, H, K]`` q/k and ``[B, T, HV, V]`` v with ``K = V = 128`` and ``HV % H == 0``,
    BF16 q/k/v, packed varlen via ``cu_seqlens`` (``B = 1``), FP32 ``initial_state`` /
    ``output_final_state`` (V-last ``[N, HV, K, V]``), ``use_qk_l2norm_in_kernel``, and ``cp_context``
    state passing.  Gate fusion (``use_gate_in_kernel``), beta sigmoid fusion, ``state_v_first``,
    ``head_first``, and non-64 chunk sizes are rejected explicitly.
    """
    for name, value in unsupported.items():
        if value not in (None, False):
            raise NotImplementedError(f"chunk_gated_delta_rule: {name}={value!r} is not supported by the deterministic kernels")
    if chunk_size != CHUNK:
        raise NotImplementedError(f"chunk_gated_delta_rule: chunk_size must be {CHUNK}")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise NotImplementedError("chunk_gated_delta_rule: q/k/v must be bfloat16")
    if q.shape[-1] != HEAD_DIM or v.shape[-1] != HEAD_DIM:
        raise NotImplementedError("chunk_gated_delta_rule: head dims must be 128")
    if g.dtype != torch.float32:
        g = g.float()
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if cu_seqlens is not None and cu_seqlens_cpu is None and cp_context is not None:
        cu_seqlens_cpu = getattr(cp_context, "cu_seqlens_cpu", None)
    return ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
        use_qk_l2norm_in_kernel,
        cp_context,
    )
