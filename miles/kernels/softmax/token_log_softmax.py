"""Log-softmax of target tokens at selected rows: per-row statistics and the logits gradient.

The streaming kernels read one vocab shard of the selected rows in the logits' dtype and compute in
fp32 on ``d = (x - m) / T``, measured from the row max ``m``. Columns from ``n_unpadded_cols`` on
are vocab padding, with probability and gradient zero. The gradient is written in two passes: a
dense one that needs no targets, then a per-row pass that writes the exact value at each target.
A store-only kernel zeroes the rows the op did not score. The streaming passes are memory-bound,
so the launch shape is the only per-GPU setting; ``kernel_configs`` picks it by GPU family.
"""

import functools
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class LaunchConfig:
    """Vocab elements per block and warps per program for one kernel."""

    block_v: int
    num_warps: int


@dataclass(frozen=True)
class KernelConfigs:
    stats: LaunchConfig
    grad: LaunchConfig
    zero: LaunchConfig


# measured at [65536, 129280] bf16 by tests/manual/bench_fused_log_probs.py, which prints a new row
_MEASURED_CONFIGS = {
    "sm90": KernelConfigs(stats=LaunchConfig(2048, 1), grad=LaunchConfig(4096, 1), zero=LaunchConfig(2048, 16)),
    "sm100": KernelConfigs(stats=LaunchConfig(2048, 1), grad=LaunchConfig(8192, 2), zero=LaunchConfig(2048, 16)),
    "sm103": KernelConfigs(stats=LaunchConfig(2048, 1), grad=LaunchConfig(8192, 2), zero=LaunchConfig(1024, 16)),
}
# an unmeasured family (e.g. ROCm) runs the B200 shape; every shape gives the same results
_FALLBACK_CONFIGS = _MEASURED_CONFIGS["sm100"]


def gpu_family(device: torch.device) -> str:
    """``sm<major><minor>`` on NVIDIA (sm90, sm100, sm103), the gfx architecture on ROCm."""
    props = torch.cuda.get_device_properties(device)
    if torch.version.hip is not None:
        return props.gcnArchName.split(":")[0]
    return f"sm{props.major}{props.minor}"


@functools.cache
def kernel_configs(device: torch.device) -> KernelConfigs:
    return _MEASURED_CONFIGS.get(gpu_family(device), _FALLBACK_CONFIGS)


@triton.jit
def _row_stats_kernel(
    logits_ptr,
    rows_ptr,
    max_ptr,
    sum_ptr,
    dsum_ptr,
    stride_row,
    n_unpadded_cols,
    inv_temperature,
    WITH_ENTROPY: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.load(rows_ptr + pid).to(tl.int64)
    row_ptr = logits_ptr + row * stride_row
    lanes = tl.arange(0, BLOCK_V)
    log2_scale = inv_temperature * 1.4426950408889634  # exp(d) = 2^((x - m) * log2_scale)

    # subtracting the running max m before scaling keeps d exact near m, where a confident token lives
    run_max = tl.full([], float("-inf"), tl.float32)
    run_sum = tl.zeros([], tl.float32)
    run_sum_error = tl.zeros([], tl.float32)  # Kahan compensation of run_sum
    run_dsum = tl.zeros([], tl.float32)
    for start in range(0, n_unpadded_cols, BLOCK_V):
        cols = start + lanes
        in_vocab = cols < n_unpadded_cols
        x = tl.load(row_ptr + cols, mask=in_vocab, other=float("-inf")).to(tl.float32)
        new_max = tl.maximum(run_max, tl.max(x, axis=0))
        e = tl.exp2((x - new_max) * log2_scale)
        rescale = tl.exp2((run_max - new_max) * log2_scale)
        if WITH_ENTROPY:
            # re-measuring the running sums from the new max shifts every earlier d by the same amount
            shift = tl.where(run_max == float("-inf"), 0.0, (run_max - new_max) * inv_temperature)
            d = (x - new_max) * inv_temperature
            run_dsum = (run_dsum + shift * run_sum) * rescale + tl.sum(tl.where(in_vocab, e * d, 0.0), axis=0)
        # Kahan: for a confident token run_sum sits near 1, and later blocks' ~1e-7 sums would round away
        scaled_sum = run_sum * rescale
        addend = tl.sum(e, axis=0) - run_sum_error * rescale
        new_sum = scaled_sum + addend
        run_sum_error = (new_sum - scaled_sum) - addend
        run_sum = new_sum
        run_max = new_max

    tl.store(max_ptr + pid, run_max)
    tl.store(sum_ptr + pid, run_sum)
    if WITH_ENTROPY:
        tl.store(dsum_ptr + pid, run_dsum)


@triton.jit
def _logits_grad_kernel(
    logits_ptr,
    grad_ptr,
    rows_ptr,
    max_ptr,
    log_sum_ptr,
    mean_ptr,
    target_grad_sum_ptr,
    grad_entropy_ptr,
    stride_row,
    grad_stride_row,
    n_vocab,
    n_unpadded_cols,
    inv_temperature,
    HAS_GRAD_LOG_PROBS: tl.constexpr,
    HAS_GRAD_ENTROPY: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_block = tl.program_id(1)
    row = tl.load(rows_ptr + pid_row).to(tl.int64)
    cols = pid_block * BLOCK_V + tl.arange(0, BLOCK_V)
    in_vocab = cols < n_vocab

    x = tl.load(logits_ptr + row * stride_row + cols, mask=in_vocab, other=0.0).to(tl.float32)
    d = (x - tl.load(max_ptr + pid_row)) * inv_temperature
    p = tl.exp2((d - tl.load(log_sum_ptr + pid_row)) * 1.4426950408889634)
    p = tl.where(cols < n_unpadded_cols, p, 0.0)  # padding columns: zero probability, so a zero gradient
    dd = tl.zeros([BLOCK_V], tl.float32)
    if HAS_GRAD_LOG_PROBS:
        # every column's share of the target terms; the target columns are rewritten exactly later
        dd = -tl.load(target_grad_sum_ptr + pid_row) * p
    if HAS_GRAD_ENTROPY:
        c = tl.load(grad_entropy_ptr + pid_row)
        dd = dd - c * p * (d - tl.load(mean_ptr + pid_row))
    grad = dd * inv_temperature
    tl.store(grad_ptr + row * grad_stride_row + cols, grad.to(grad_ptr.dtype.element_ty), mask=in_vocab)


@triton.jit
def _zero_unscored_rows_kernel(grad_ptr, scored_ptr, grad_stride_row, n_vocab, BLOCK_V: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    if tl.load(scored_ptr + row) == 0:
        row_ptr = grad_ptr + row * grad_stride_row
        zeros = tl.zeros([BLOCK_V], grad_ptr.dtype.element_ty)
        for start in range(0, n_vocab, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            tl.store(row_ptr + cols, zeros, mask=cols < n_vocab)


@triton.jit
def _target_grads_kernel(
    grad_ptr,
    rows_ptr,
    targets_ptr,
    log_probs_ptr,
    one_minus_p_ptr,
    grad_log_probs_ptr,
    target_grad_sum_ptr,
    log_sum_ptr,
    mean_ptr,
    grad_entropy_ptr,
    grad_stride_row,
    n_targets,
    n_unpadded_cols,
    vocab_start,
    inv_temperature,
    HAS_GRAD_LOG_PROBS: tl.constexpr,
    HAS_GRAD_ENTROPY: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.load(rows_ptr + pid).to(tl.int64)
    lanes = tl.arange(0, BLOCK_K)
    in_row = lanes < n_targets
    offs = pid * n_targets + lanes
    target = tl.load(targets_ptr + offs, mask=in_row, other=-1)
    col = target - vocab_start
    owned = in_row & (target >= 0) & (col >= 0) & (col < n_unpadded_cols)

    log_p = tl.load(log_probs_ptr + offs, mask=owned, other=0.0)
    p = tl.exp(log_p)
    dd = tl.zeros([BLOCK_K], tl.float32)
    if HAS_GRAD_LOG_PROBS:
        # a token listed twice in a row gets the sum of its gradients, so every copy writes the same value
        g_token = tl.zeros([BLOCK_K], tl.float32)
        for j in range(0, n_targets):
            same = target == tl.load(targets_ptr + pid * n_targets + j)
            g_token += tl.where(same, tl.load(grad_log_probs_ptr + pid * n_targets + j), 0.0)
        g_row = tl.load(target_grad_sum_ptr + pid)
        # 1 - p comes precomputed with expm1: 1 - exp(log p) would round it away for a confident token
        dd = g_token * tl.load(one_minus_p_ptr + offs, mask=owned, other=0.0) - (g_row - g_token) * p
    if HAS_GRAD_ENTROPY:
        d = log_p + tl.load(log_sum_ptr + pid)
        dd = dd - tl.load(grad_entropy_ptr + pid) * p * (d - tl.load(mean_ptr + pid))
    tl.store(grad_ptr + row * grad_stride_row + col, (dd * inv_temperature).to(grad_ptr.dtype.element_ty), mask=owned)


def row_statistics(
    logits,
    rows,
    *,
    n_unpadded_cols: int,
    temperature: float,
    with_entropy: bool,
    launch: LaunchConfig | None = None,
):
    """This shard's ``(max, sum exp(d), sum exp(d) * d or None)`` per row, ``d = (x - max) / T``.

    Only the first ``n_unpadded_cols`` columns count; a shard with none reports ``(-inf, 0, 0)``.
    """
    launch = launch or kernel_configs(logits.device).stats
    n_rows = rows.numel()
    stats = torch.empty((3, n_rows), dtype=torch.float32, device=logits.device)
    if n_rows:
        _row_stats_kernel[(n_rows,)](
            logits,
            rows,
            stats[0],
            stats[1],
            stats[2],
            logits.stride(0),
            n_unpadded_cols,
            1.0 / temperature,
            WITH_ENTROPY=with_entropy,
            BLOCK_V=launch.block_v,
            num_warps=launch.num_warps,
        )
    row_max, row_sum, row_dsum = stats
    return row_max, row_sum, (row_dsum if with_entropy else None)


def write_logits_grad(
    grad,
    logits,
    rows,
    targets,
    log_probs,
    one_minus_p,
    row_max,
    log_sum,
    mean,
    grad_log_probs,
    grad_entropy,
    *,
    vocab_start: int,
    n_unpadded_cols: int,
    temperature: float,
    launch: LaunchConfig | None = None,
):
    """Write the gradient of the selected rows into ``grad``; ``grad`` may be ``logits`` itself.

    ``targets``, ``log_probs``, ``one_minus_p`` and ``grad_log_probs`` are ``[R, K]``; a ``-1``
    target is padding. Every column of a selected row is written, the vocab padding from
    ``n_unpadded_cols`` on with zero.
    """
    launch = launch or kernel_configs(logits.device).grad
    n_rows, n_targets = targets.shape
    if not n_rows:
        return
    target_grad_sum = grad_log_probs.sum(dim=1) if grad_log_probs is not None else None
    placeholder = log_sum  # a pointer for a tensor the kernel's flags say it never reads
    _logits_grad_kernel[(n_rows, triton.cdiv(logits.size(1), launch.block_v))](
        logits,
        grad,
        rows,
        row_max,
        log_sum,
        mean if mean is not None else placeholder,
        target_grad_sum if target_grad_sum is not None else placeholder,
        grad_entropy if grad_entropy is not None else placeholder,
        logits.stride(0),
        grad.stride(0),
        logits.size(1),
        n_unpadded_cols,
        1.0 / temperature,
        HAS_GRAD_LOG_PROBS=grad_log_probs is not None,
        HAS_GRAD_ENTROPY=grad_entropy is not None,
        BLOCK_V=launch.block_v,
        num_warps=launch.num_warps,
    )
    # stream order puts this after the dense pass, so these values replace its target columns
    _target_grads_kernel[(n_rows,)](
        grad,
        rows,
        targets,
        log_probs,
        one_minus_p,
        grad_log_probs if grad_log_probs is not None else placeholder,
        target_grad_sum if target_grad_sum is not None else placeholder,
        log_sum,
        mean if mean is not None else placeholder,
        grad_entropy if grad_entropy is not None else placeholder,
        grad.stride(0),
        n_targets,
        n_unpadded_cols,
        vocab_start,
        1.0 / temperature,
        HAS_GRAD_LOG_PROBS=grad_log_probs is not None,
        HAS_GRAD_ENTROPY=grad_entropy is not None,
        BLOCK_K=max(triton.next_power_of_2(n_targets), 16),
        num_warps=1,
    )


def zero_unscored_rows(grad, rows, *, launch: LaunchConfig | None = None):
    """Zero the rows of ``grad`` that are not in ``rows``, writing only those rows."""
    launch = launch or kernel_configs(grad.device).zero
    n_rows = grad.size(0)
    scored = torch.zeros(n_rows, dtype=torch.uint8, device=grad.device)
    scored[rows] = 1
    _zero_unscored_rows_kernel[(n_rows,)](
        grad, scored, grad.stride(0), grad.size(1), BLOCK_V=launch.block_v, num_warps=launch.num_warps
    )
