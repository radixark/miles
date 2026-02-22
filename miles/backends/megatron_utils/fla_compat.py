"""
Patch installed FLA's GDN kernels to match SGLang's implementations.

Problem: The installed FLA v0.4.0 and SGLang's forked FLA use different Triton kernel
implementations for the Gated Delta Net (GDN) chunk computation. Key differences:
  1. solve_tril: loop start i=2 (FLA) vs i=1 (SGLang) — CRITICAL algorithmic difference
     that causes the 16x16 triangular inverse to be computed incorrectly for the first
     off-diagonal element, producing max diffs of ~65536 in the A matrix.
  2. chunk_scaled_dot_kkt: safe_exp vs exp, different boundary masking.
  3. chunk_fwd_o: boundary masking differences.

These differences cause train_rollout_logprob_abs_diff ≈ 7.5 when comparing Megatron's
GDN forward pass against SGLang's inference engine.

Fix: Replace FLA's forward-path kernel functions with SGLang's implementations in the
chunk module namespace. The backward pass (from installed FLA) is unaffected because:
  - It does NOT call solve_tril, chunk_scaled_dot_kkt_fwd, or chunk_fwd_o
  - It uses the saved A matrix from forward and calls only recompute_w_u_fwd,
    chunk_bwd_dv_local, chunk_gated_delta_rule_bwd_dhu, chunk_bwd_dqkwg, and
    prepare_wy_repr_bwd — all of which remain unpatched
"""

import logging
import sys

logger = logging.getLogger(__name__)

_patched = False


def patch_fla_for_sglang_compat():
    """Replace FLA's forward-path GDN kernels with SGLang's versions.

    Patches the following functions in fla.ops.gated_delta_rule.chunk:
    - solve_tril: identical API, direct replacement
    - chunk_scaled_dot_kkt_fwd: wrapper to adapt parameter names (g → g_cumsum)
    - chunk_fwd_o: wrapper to drop unused g_gamma parameter

    Safe to call multiple times (idempotent). No-op if SGLang or FLA is not available.
    """
    global _patched
    if _patched:
        return True

    try:
        import fla.ops.gated_delta_rule.chunk as fla_chunk
    except ImportError:
        logger.debug("FLA not installed, skipping SGLang compatibility patch")
        return False

    try:
        sglang_path = "/sgl-workspace/sglang/python"
        if sglang_path not in sys.path:
            sys.path.insert(0, sglang_path)

        from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o as sglang_chunk_fwd_o
        from sglang.srt.layers.attention.fla.chunk_scaled_dot_kkt import (
            chunk_scaled_dot_kkt_fwd as sglang_chunk_scaled_dot_kkt_fwd,
        )
        from sglang.srt.layers.attention.fla.solve_tril import solve_tril as sglang_solve_tril
    except ImportError:
        logger.debug("SGLang FLA not available, skipping compatibility patch")
        return False

    # 1. Patch solve_tril — identical API, direct replacement
    fla_chunk.solve_tril = sglang_solve_tril

    # 2. Patch chunk_scaled_dot_kkt_fwd — adapt parameter names
    #    FLA calls:  chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta, ...)
    #    SGLang has:  chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum=None, ...)
    _sglang_kkt = sglang_chunk_scaled_dot_kkt_fwd

    import torch as _torch

    def patched_chunk_scaled_dot_kkt_fwd(
        k, g=None, beta=None, cu_seqlens=None, chunk_size=64, output_dtype=_torch.float32
    ):
        return _sglang_kkt(
            k=k,
            beta=beta,
            g_cumsum=g,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            output_dtype=output_dtype,
        )

    fla_chunk.chunk_scaled_dot_kkt_fwd = patched_chunk_scaled_dot_kkt_fwd

    # 3. Patch chunk_fwd_o — drop g_gamma parameter not in SGLang
    #    FLA calls:  chunk_fwd_o(q, k, v, h, g=g, scale=scale, cu_seqlens=cu_seqlens)
    #    SGLang has:  chunk_fwd_o(q, k, v, h, g=None, scale=None, cu_seqlens=None, ...)
    _sglang_fwd_o = sglang_chunk_fwd_o

    def patched_chunk_fwd_o(q, k, v, h, g=None, g_gamma=None, scale=None, cu_seqlens=None, chunk_size=64):
        return _sglang_fwd_o(
            q=q,
            k=k,
            v=v,
            h=h,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )

    fla_chunk.chunk_fwd_o = patched_chunk_fwd_o

    _patched = True
    logger.info(
        "Patched FLA GDN kernels (solve_tril, chunk_scaled_dot_kkt_fwd, chunk_fwd_o) "
        "with SGLang-compatible versions"
    )
    return True
