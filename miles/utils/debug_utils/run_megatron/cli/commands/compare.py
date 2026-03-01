"""``compare`` CLI command."""

import sys
from pathlib import Path
from typing import Annotated

import typer

from miles.utils.misc import exec_command


def register(app: typer.Typer) -> None:
    """Register the ``compare`` command on *app*."""
    app.command()(compare)


def compare(
    baseline_dir: Annotated[Path, typer.Option(help="Baseline dump directory")],
    target_dir: Annotated[Path, typer.Option(help="Target dump directory")],
    output_format: Annotated[str, typer.Option(help="Output format: text / json")] = "text",
    grouping: Annotated[str, typer.Option(help="Grouping: logical / raw")] = "logical",
    override_baseline_dims: Annotated[str | None, typer.Option(help="Override baseline dims")] = None,
    override_target_dims: Annotated[str | None, typer.Option(help="Override target dims")] = None,
    patch_config: Annotated[Path | None, typer.Option(help="Patch config YAML path")] = None,
    diff_threshold: Annotated[float | None, typer.Option(help="Pass/fail threshold")] = None,
) -> None:
    """Run comparator on existing dump directories."""
    cmd_parts: list[str] = [
        sys.executable,
        "-m",
        "sglang.srt.debug_utils.comparator",
        "--baseline-path",
        str(baseline_dir),
        "--target-path",
        str(target_dir),
        "--output-format",
        output_format,
        "--grouping",
        grouping,
    ]
    if override_baseline_dims is not None:
        cmd_parts.extend(["--override-baseline-dims", override_baseline_dims])
    if override_target_dims is not None:
        cmd_parts.extend(["--override-target-dims", override_target_dims])
    if patch_config is not None:
        cmd_parts.extend(["--patch-config", str(patch_config)])
    if diff_threshold is not None:
        cmd_parts.extend(["--diff-threshold", str(diff_threshold)])

    exec_command(" ".join(cmd_parts))
    print("[cli] Compare completed.", flush=True)
