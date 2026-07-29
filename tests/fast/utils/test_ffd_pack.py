"""CPU tests for first-fit-decreasing packing and its bin-count gain over first-fit."""

from __future__ import annotations

import random

from miles.utils.seqlen_balancing import first_fit_decreasing_pack, first_fit_pack


def _check_valid(bins, lengths, cap):
    seen = sorted(i for b in bins for i in b)
    assert seen == list(range(len(lengths))), "every sample packed exactly once"
    for b in bins:
        total = sum(lengths[i] for i in b)
        assert total <= cap or len(b) == 1, f"bin over cap with >1 sample: {total}"


def test_ffd_same_contract_as_first_fit():
    rng = random.Random(3)
    for _ in range(50):
        n = rng.randint(1, 64)
        cap = rng.randint(50, 400)
        lengths = [rng.randint(1, cap * 2) for _ in range(n)]  # includes oversized
        _check_valid(first_fit_decreasing_pack(lengths, cap), lengths, cap)


def test_ffd_never_more_bins_than_first_fit():
    rng = random.Random(11)
    for _ in range(200):
        n = rng.randint(2, 96)
        cap = rng.randint(100, 1000)
        lengths = [rng.randint(1, cap) for _ in range(n)]
        ff = len(first_fit_pack(lengths, cap))
        ffd = len(first_fit_decreasing_pack(lengths, cap))
        assert ffd <= ff, f"FFD produced more bins ({ffd}) than FF ({ff})"


def test_ffd_gain_on_mid_length_distribution():
    """FFD's gain concentrates where item sizes are a sizable fraction of the
    cap (long-response RL with roomy token budgets): measured ~3-7% fewer bins
    for mean-length ~cap/4..cap/2 regimes, 0% for short-response regimes where
    both packers are already optimal. Assert the mid regime keeps a >=4% edge."""
    rng = random.Random(42)
    cap = 9216
    total_ff = total_ffd = 0
    for _ in range(500):
        lengths = [rng.randint(2000, 6000) for _ in range(32)]
        total_ff += len(first_fit_pack(lengths, cap))
        total_ffd += len(first_fit_decreasing_pack(lengths, cap))
    assert total_ffd <= total_ff * 0.96, f"expected >=4% bin savings, got FF={total_ff} FFD={total_ffd}"
