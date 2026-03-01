"""``compare`` CLI command."""

import sys

import typer

from miles.utils.debug_utils.run_megatron.cli.commands.args import CompareArgs
from miles.utils.misc import exec_command
from miles.utils.typer_utils import dataclass_cli


def register(app: typer.Typer) -> None:
    """Register the ``compare`` command on *app*."""
    app.command()(compare)


def compare_impl(args: CompareArgs) -> None:
    """Core compare logic, called by both ``compare`` command and ``run_and_compare``."""
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

    exec_command(" ".join(cmd_parts))
    print("[cli] Compare completed.", flush=True)


@dataclass_cli(env_var_prefix="")
def compare(args: CompareArgs) -> None:
    """Run comparator on existing dump directories."""
    compare_impl(args)
