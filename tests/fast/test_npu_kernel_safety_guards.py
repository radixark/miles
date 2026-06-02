"""Source-level safety-guard tests for the NPU tilelang kernels.

Verifies that defensive guards added in response to reviewer feedback on
miles#1246 are present and shaped correctly:

  * 5 negative-sentinel guards (`if cur_idx >= 0:`) — prevent NPU AICore
    OOB / hardware-trap on `idx == -1` masked top-k positions.
  * NPU atomic-add intrinsic spelling — `T.npuir_atomic_addx4`, NOT
    `T.atomic_addx4` (the latter was a typo in lighting_indexer_bwd that
    silently broke the scatter on NPU).
  * `_MAGIC_THRESHOLD` in `indexer.py` is AMP-safe (>= 1e10) so that
    legitimate loss-scaled gradients survive the defensive R-KA-15 filter.

These tests do source-level checks, not runtime checks: the actual sentinel
behaviour requires an Ascend NPU + tilelang JIT compile, which can't run in
the fast-test suite. The source-level check is enough to prevent regression
of these specific reviewer-flagged issues — if someone deletes a guard or
flips the threshold, this test fails immediately in CI.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_NPU_DIR = (
    Path(__file__).resolve().parents[2]
    / "miles_plugins"
    / "models"
    / "glm5"
    / "ops"
    / "_npu"
)


def _read(name: str) -> str:
    p = _NPU_DIR / name
    assert p.is_file(), f"NPU kernel source missing: {p}"
    return p.read_text(encoding="utf-8")


# ---- 1. Sentinel guards ----------------------------------------------------


def _count_sentinel_guards(src: str) -> int:
    """Count `if cur_idx >= 0:` occurrences."""
    return len(re.findall(r"\bif\s+cur_idx\s*>=\s*0\s*:", src))


def test_lighting_indexer_bwd_has_two_sentinel_guards():
    """Reviewer #1246 HIGH-1 + HIGH-2: gather AND scatter must guard cur_idx."""
    src = _read("_lighting_indexer_bwd_kernel.py")
    n = _count_sentinel_guards(src)
    assert n >= 2, (
        f"expected >= 2 `if cur_idx >= 0:` guards in "
        f"_lighting_indexer_bwd_kernel.py (one for IndexK gather, one for "
        f"dIndexK atomic-add); found {n}"
    )


def test_sparse_mla_bwd_has_two_sentinel_guards():
    """Reviewer #1246 HIGH-3 + HIGH-4: gather AND scatter must guard cur_idx."""
    src = _read("_sparse_mla_bwd_kernel.py")
    n = _count_sentinel_guards(src)
    assert n >= 2, (
        f"expected >= 2 `if cur_idx >= 0:` guards in "
        f"_sparse_mla_bwd_kernel.py (one for KV gather, one for dKV "
        f"atomic-add); found {n}"
    )


def test_sparse_mla_fwd_has_one_sentinel_guard():
    """Reviewer #1246 HIGH-5: fwd gather must guard cur_idx."""
    src = _read("_sparse_mla_fwd_kernel.py")
    n = _count_sentinel_guards(src)
    assert n >= 1, (
        f"expected >= 1 `if cur_idx >= 0:` guard in "
        f"_sparse_mla_fwd_kernel.py (KV gather); found {n}"
    )


# ---- 2. NPU atomic-add intrinsic spelling ----------------------------------


def test_lighting_indexer_bwd_uses_npuir_atomic_addx4():
    """Reviewer #1246 HIGH-2: must be `T.npuir_atomic_addx4`, not `T.atomic_addx4`.

    `T.atomic_addx4` is not a real intrinsic exposed by the NPU backend; using
    it silently breaks the scatter (the lighting_indexer_bwd scatter writes to
    `dIndexK` via this intrinsic, so getting the name wrong means `dIndexK`
    accumulator stays zero on NPU). The correct backend symbol is
    `T.npuir_atomic_addx4`, and the sparse_mla_bwd kernel already uses it.
    """
    src = _read("_lighting_indexer_bwd_kernel.py")
    bad = re.findall(r"\bT\.atomic_addx4\s*\(", src)
    good = re.findall(r"\bT\.npuir_atomic_addx4\s*\(", src)
    assert not bad, (
        f"lighting_indexer_bwd still uses T.atomic_addx4 (wrong intrinsic, "
        f"breaks NPU scatter); found {len(bad)} occurrences"
    )
    assert good, (
        "lighting_indexer_bwd does not call T.npuir_atomic_addx4 — the "
        "dIndexK scatter is missing"
    )


def test_sparse_mla_bwd_uses_npuir_atomic_addx4():
    """sparse_mla_bwd was already using the correct intrinsic — keep it."""
    src = _read("_sparse_mla_bwd_kernel.py")
    bad = re.findall(r"\bT\.atomic_addx4\s*\(", src)
    good = re.findall(r"\bT\.npuir_atomic_addx4\s*\(", src)
    assert not bad
    assert good


# ---- 3. AMP-safe _MAGIC_THRESHOLD -----------------------------------------


def _extract_magic_threshold(src: str) -> float:
    m = re.search(r"_MAGIC_THRESHOLD\s*=\s*([0-9eE.+\-]+)", src)
    assert m, "_MAGIC_THRESHOLD assignment not found in indexer.py"
    return float(m.group(1))


def test_magic_threshold_is_amp_safe():
    """Reviewer #1246 HIGH-6: threshold must not zero legitimate AMP gradients.

    With AMP loss-scale up to 2^16 (65536), a normal post-scale gradient of
    0.1 is 6553.6 — far above the original 1e3, which would silently zero
    it and break training. Threshold must be >= 1e10 to stay above any
    realistic scaled gradient while still catching the R-KA-15 sentinel
    (~6e37) and its bf16-cast remnants (~1.6e29).
    """
    src = _read("indexer.py")
    thresh = _extract_magic_threshold(src)
    assert thresh >= 1e10, (
        f"_MAGIC_THRESHOLD = {thresh} is unsafe under AMP "
        f"(loss-scale * legitimate-grad can exceed 1e7 routinely); "
        f"must be >= 1e10"
    )
    assert thresh < 1e38, (
        f"_MAGIC_THRESHOLD = {thresh} exceeds FP32 max (~3.4e38); the R-KA-15 "
        f"sentinel value (6.04e37) would then survive the filter"
    )


# ---- 4. Cross-reference helper -------------------------------------------


@pytest.mark.parametrize(
    "fname",
    [
        "_lighting_indexer_bwd_kernel.py",
        "_sparse_mla_bwd_kernel.py",
        "_sparse_mla_fwd_kernel.py",
    ],
)
def test_kernel_source_mentions_reviewer_finding(fname: str):
    """Comment trail must reference '#1246' so future readers find the why."""
    src = _read(fname)
    assert "#1246" in src, (
        f"{fname} no longer references reviewer #1246 finding — the rationale "
        f"comment was likely deleted by accident"
    )
