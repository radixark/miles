"""Delegate TOP trainer projections to serving's batch-invariant GEMMs."""

from sglang.srt.batch_invariant_ops import matmul_persistent
from sglang.srt.true_on_policy.qwen_moe_router import shared_gate_linear

from miles_plugins.top.kernels.surrogate import linear_surrogate
from miles_plugins.top.ops import delegated_matched_tp1_linear


def _column_linear(x, weight):
    if weight.shape[0] == 1:
        return shared_gate_linear(x, weight)
    return matmul_persistent(x.contiguous(), weight.t())


def delegated_column_linear(x, weight):
    """x [T, K] @ weight.T with serving's GEMM; the gradient is the ordinary linear one."""
    return linear_surrogate(_column_linear, x, weight, name="column_linear")


def delegated_row_linear(x_2d, weight):
    """Row projection on the full-K GEMM serving shares with the TP1 trainer."""
    return delegated_matched_tp1_linear(x_2d, weight)
