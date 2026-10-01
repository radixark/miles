"""Sweep the launch shape of the ``--log-probs-backend fused`` kernels on this GPU.

Each kernel is a memory-bound streaming pass, so it is reported as bandwidth next to a device copy
of the same logits. The best shape per kernel is printed as the ``_MEASURED_CONFIGS`` row to add
for this GPU family in ``miles/backends/training_utils/loss_hub/fused_log_probs_triton.py``.

    python tests/manual/bench_fused_log_probs.py [--rows 65536] [--vocab 129280]
"""

import argparse
import itertools

import torch

from miles.backends.training_utils.loss_hub import fused_log_probs_triton as kernels
from miles.backends.training_utils.loss_hub.fused_log_probs_triton import KernelConfigs, LaunchConfig

BLOCKS = (1024, 2048, 4096, 8192)
WARPS = (1, 2, 4, 8, 16)
TEMPERATURE = 0.8


def _median_ms(fn, iters: int = 7) -> float:
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def _best_launch(name: str, run, moved_bytes: int, copy_tbps: float) -> LaunchConfig:
    """Time ``run(launch)`` for every shape; print the three fastest and return the fastest."""
    timings = []
    for block_v, num_warps in itertools.product(BLOCKS, WARPS):
        if block_v >= 32 * num_warps:  # at least one element per thread
            launch = LaunchConfig(block_v, num_warps)
            timings.append((_median_ms(lambda launch=launch: run(launch)), launch))
    timings.sort(key=lambda timing: timing[0])
    for ms, launch in timings[:3]:
        tbps = moved_bytes / ms / 1e9
        print(
            f"  {name:<5} block_v={launch.block_v:<5} num_warps={launch.num_warps:<3}"
            f"{ms:7.2f} ms {tbps:5.2f} TB/s ({tbps / copy_tbps:.0%} of a copy)"
        )
    return timings[0][1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--vocab", type=int, default=129_280)
    args = parser.parse_args()

    device = torch.device("cuda", torch.cuda.current_device())
    logits = (torch.randn(args.rows, args.vocab, device=device) * 3).to(torch.bfloat16)
    rows = torch.arange(args.rows, device=device)
    targets = torch.randint(0, args.vocab, (args.rows,), device=device)
    logits_bytes = logits.numel() * logits.element_size()

    copy_tbps = 2 * logits_bytes / _median_ms(lambda: logits.clone()) / 1e9  # a copy reads and writes
    family = kernels.gpu_family(device)
    print(f"{torch.cuda.get_device_name(device)} ({family}): [{args.rows}, {args.vocab}] bf16 logits")
    print(f"  device copy {copy_tbps:5.2f} TB/s")

    def stats(launch):
        for with_entropy in (False, True):
            kernels.row_statistics(
                logits, rows, targets, vocab_start=0, temperature=TEMPERATURE, with_entropy=with_entropy, launch=launch
            )

    row_max, row_sum, row_dsum, target = kernels.row_statistics(
        logits, rows, targets, vocab_start=0, temperature=TEMPERATURE, with_entropy=True
    )
    log_sum, mean = torch.log(row_sum), row_dsum / row_sum
    one_minus_p = -torch.expm1(target - log_sum)
    grad_log_probs = torch.randn(args.rows, device=device)
    grad_entropy = torch.randn(args.rows, device=device)
    grad = torch.empty_like(logits)

    def write_grad(launch):
        kernels.write_logits_grad(
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
            vocab_start=0,
            temperature=TEMPERATURE,
            launch=launch,
        )

    no_rows = torch.empty(0, dtype=torch.long, device=device)

    def zero(launch):
        kernels.zero_unscored_rows(grad, no_rows, launch=launch)

    best = KernelConfigs(
        stats=_best_launch("stats", stats, 2 * logits_bytes, copy_tbps),  # two reads: with and without entropy
        grad=_best_launch("grad", write_grad, 2 * logits_bytes, copy_tbps),  # a read and a write
        zero=_best_launch("zero", zero, logits_bytes, copy_tbps),  # writes every row
    )
    print("\nadd to _MEASURED_CONFIGS:")
    print(
        f'    "{family}": KernelConfigs(stats=LaunchConfig({best.stats.block_v}, {best.stats.num_warps}), '
        f"grad=LaunchConfig({best.grad.block_v}, {best.grad.num_warps}), "
        f"zero=LaunchConfig({best.zero.block_v}, {best.zero.num_warps})),"
    )


if __name__ == "__main__":
    main()
