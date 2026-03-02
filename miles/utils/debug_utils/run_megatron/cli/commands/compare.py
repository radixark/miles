"""``compare`` CLI command."""

import subprocess
import sys

import typer

from miles.utils.debug_utils.run_megatron.cli.commands.args import CompareArgs
from miles.utils.debug_utils.run_megatron.logprob_comparator import compare_logprobs
from miles.utils.misc import exec_command
from miles.utils.typer_utils import dataclass_cli


def register(app: typer.Typer) -> None:
    """Register the ``compare`` command on *app*."""
    app.command()(compare)


def compare_impl(args: CompareArgs) -> None:
    """Core compare logic, called by both ``compare`` command and ``run_and_compare``."""
    activation_passed = _run_activation_comparison(args)

    logprob_passed = True
    if args.baseline_logprob_dir is not None and args.target_logprob_dir is not None:
        logprob_passed = compare_logprobs(
            baseline_dir=args.baseline_logprob_dir,
            target_dir=args.target_logprob_dir,
            threshold=args.logprob_threshold if args.logprob_threshold is not None else 1e-3,
        )

    if not activation_passed or not logprob_passed:
        failures: list[str] = []
        if not activation_passed:
            failures.append("activation comparison")
        if not logprob_passed:
            failures.append("logprob comparison")
        print(f"[cli] FAILED: {', '.join(failures)}", flush=True)
        sys.exit(1)

    print("[cli] Compare completed.", flush=True)


def _run_activation_comparison(args: CompareArgs) -> bool:
    cmd_parts: list[str] = [
        sys.executable,
        "-m",
        "sglang.srt.debug_utils.comparator",
        "--baseline-path",
        str(args.baseline_dir),
        "--target-path",
        str(args.target_dir),
        "--output-format",
        args.output_format,
        "--grouping",
        args.grouping,
    ]

    optional_args: dict[str, object | None] = {
        "--override-baseline-dims": args.override_baseline_dims,
        "--override-target-dims": args.override_target_dims,
        "--patch-config": args.patch_config,
        "--diff-threshold": args.diff_threshold,
    }
    for flag, value in optional_args.items():
        if value is not None:
            cmd_parts.extend([flag, str(value)])

    try:
        exec_command(" ".join(cmd_parts))
        return True
    except subprocess.CalledProcessError:
        print("[cli] Activation comparison failed", flush=True)
        return False


@dataclass_cli(env_var_prefix="")
def compare(args: CompareArgs) -> None:
    """Run comparator on existing dump directories."""
    compare_impl(args)
