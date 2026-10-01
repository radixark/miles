"""Triton kernels for ``fused_log_probs``: per-row softmax statistics and the logits gradient.

Both kernels read one vocab shard of selected logits rows, in the logits' own dtype, and do all
arithmetic in fp32 on ``d = (x - m) / T``, measured from the row max ``m`` before scaling. To keep
the exponential off the critical path they use one ``exp2`` per element, multiply by the reciprocal
temperature instead of dividing, and the statistics kernel rescales its running sums once per
block, not per element.

A third, store-only kernel zeroes the gradient rows the op did not score, so the backward never
reads or rewrites the scored rows a second time.

All three are memory-bound streaming passes, so their launch shape (vocab block and warps) is the
only thing to tune per GPU; ``kernel_configs`` picks it by GPU family.
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


# Measured at [65536, 129280] bf16 with tests/manual/bench_fused_log_probs.py, which prints the row
# for the GPU it runs on. Bandwidth as a share of a device copy on the same GPU:
#   sm90 (H100, H200), on H200: statistics 101%, gradient 93%, zeroing 103%
#   sm100 (B200, GB200), on B200: statistics 89%, gradient 89%, zeroing 108%
#   sm103 (B300, GB300), on GB300: statistics 89%, gradient 90%, zeroing 107%
_MEASURED_CONFIGS = {
    "sm90": KernelConfigs(stats=LaunchConfig(2048, 1), grad=LaunchConfig(4096, 1), zero=LaunchConfig(2048, 16)),
    "sm100": KernelConfigs(stats=LaunchConfig(2048, 1), grad=LaunchConfig(8192, 2), zero=LaunchConfig(2048, 16)),
    "sm103": KernelConfigs(stats=LaunchConfig(2048, 1), grad=LaunchConfig(8192, 2), zero=LaunchConfig(1024, 16)),
}
# A family nobody has measured yet (gfx942: MI300X; gfx950: MI350X, MI355X) runs the B200 shape:
# every shape gives the same results, only the speed differs.
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
    targets_ptr,
    max_ptr,
    sum_ptr,
    dsum_ptr,
    target_ptr,
    stride_row,
    n_vocab,
    vocab_start,
    inv_temperature,
    WITH_ENTROPY: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.load(rows_ptr + pid).to(tl.int64)
    row_ptr = logits_ptr + row * stride_row
    lanes = tl.arange(0, BLOCK_V)
    log2_scale = inv_temperature * 1.4426950408889634  # exp(d) = 2^((x - m) * log2_scale)

    # with d = (x - m) / T for the running max m: sum exp(d) and sum exp(d) * d. Subtracting the max
    # before scaling keeps d exact near the max, where a confident token's log-prob lives.
    run_max = tl.full([], float("-inf"), tl.float32)
    run_sum = tl.zeros([], tl.float32)
    run_sum_error = tl.zeros([], tl.float32)  # Kahan compensation of run_sum
    run_dsum = tl.zeros([], tl.float32)
    for start in range(0, n_vocab, BLOCK_V):
        cols = start + lanes
        in_vocab = cols < n_vocab
        x = tl.load(row_ptr + cols, mask=in_vocab, other=float("-inf")).to(tl.float32)
        new_max = tl.maximum(run_max, tl.max(x, axis=0))
        e = tl.exp2((x - new_max) * log2_scale)
        rescale = tl.exp2((run_max - new_max) * log2_scale)
        if WITH_ENTROPY:
            # re-measuring the running sums from the new max shifts every earlier d by the same amount
            shift = tl.where(run_max == float("-inf"), 0.0, (run_max - new_max) * inv_temperature)
            d = (x - new_max) * inv_temperature
            run_dsum = (run_dsum + shift * run_sum) * rescale + tl.sum(tl.where(in_vocab, e * d, 0.0), axis=0)
        # For a confident token run_sum sits near 1 while each later block adds ~1e-7; compensated
        # addition keeps those small sums instead of rounding each one away at 1's ulp.
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

    target = tl.load(targets_ptr + pid) - vocab_start
    in_shard = (target >= 0) & (target < n_vocab)
    target_x = tl.load(row_ptr + target, mask=in_shard, other=0.0).to(tl.float32)
    tl.store(target_ptr + pid, tl.where(in_shard, (target_x - run_max) * inv_temperature, 0.0))


@triton.jit
def _logits_grad_kernel(
    logits_ptr,
    grad_ptr,
    rows_ptr,
    targets_ptr,
    max_ptr,
    log_sum_ptr,
    mean_ptr,
    one_minus_p_ptr,
    grad_log_probs_ptr,
    grad_entropy_ptr,
    stride_row,
    grad_stride_row,
    n_vocab,
    vocab_start,
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
    dd = tl.zeros([BLOCK_V], tl.float32)
    if HAS_GRAD_LOG_PROBS:
        g = tl.load(grad_log_probs_ptr + pid_row)
        target = tl.load(targets_ptr + pid_row) - vocab_start
        # 1 - p at the target comes precomputed with expm1: exp2 would round it away for a confident token
        dd = tl.where(cols == target, g * tl.load(one_minus_p_ptr + pid_row), -g * p)
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


def row_statistics(
    logits,
    rows,
    targets,
    *,
    vocab_start: int,
    temperature: float,
    with_entropy: bool,
    launch: LaunchConfig | None = None,
):
    """This shard's ``(max, sum exp(d), sum exp(d) * d or None, d_y or 0)`` per row, ``d = (x - max) / T``."""
    launch = launch or kernel_configs(logits.device).stats
    n_rows = rows.numel()
    stats = torch.empty((4, n_rows), dtype=torch.float32, device=logits.device)
    if n_rows:
        _row_stats_kernel[(n_rows,)](
            logits,
            rows,
            targets,
            stats[0],
            stats[1],
            stats[2],
            stats[3],
            logits.stride(0),
            logits.size(1),
            vocab_start,
            1.0 / temperature,
            WITH_ENTROPY=with_entropy,
            BLOCK_V=launch.block_v,
            num_warps=launch.num_warps,
        )
    row_max, row_sum, row_dsum, target = stats
    return row_max, row_sum, (row_dsum if with_entropy else None), target


def write_logits_grad(
    grad,
    logits,
    rows,
    targets,
    row_max,
    log_sum,
    mean,
    one_minus_p,
    grad_log_probs,
    grad_entropy,
    *,
    vocab_start: int,
    temperature: float,
    launch: LaunchConfig | None = None,
):
    """Write the gradient of the selected rows into ``grad``; ``grad`` may be ``logits`` itself."""
    launch = launch or kernel_configs(logits.device).grad
    n_rows = rows.numel()
    if not n_rows:
        return
    grid = (n_rows, triton.cdiv(logits.size(1), launch.block_v))
    _logits_grad_kernel[grid](
        logits,
        grad,
        rows,
        targets,
        row_max,
        log_sum,
        mean if mean is not None else log_sum,
        one_minus_p,
        grad_log_probs if grad_log_probs is not None else log_sum,
        grad_entropy if grad_entropy is not None else log_sum,
        logits.stride(0),
        grad.stride(0),
        logits.size(1),
        vocab_start,
        1.0 / temperature,
        HAS_GRAD_LOG_PROBS=grad_log_probs is not None,
        HAS_GRAD_ENTROPY=grad_entropy is not None,
        BLOCK_V=launch.block_v,
        num_warps=launch.num_warps,
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
