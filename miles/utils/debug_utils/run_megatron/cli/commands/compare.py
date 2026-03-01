"""``compare`` CLI command."""

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from miles.utils.debug_utils.run_megatron.cli.comparator_utils import assert_all_passed, print_json_summary


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
    strict: Annotated[bool, typer.Option(help="Assert all passed (exit 1 on failure)")] = True,
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

    print(f"EXEC: {' '.join(cmd_parts)}", flush=True)
    stdout_text, stderr_text, returncode = _run_streaming(cmd_parts)

    if stderr_text.strip():
        print(f"[comparator stderr]\n{stderr_text}", flush=True)
    if returncode != 0:
        print(f"[comparator] exited with code {returncode}", flush=True)
    if output_format == "json":
        print_json_summary(stdout_text)

    if strict:
        if returncode != 0:
            raise typer.Exit(code=1)
        if output_format == "json":
            assert_all_passed(stdout_text)

    print("[cli] Compare completed.", flush=True)


def _run_streaming(cmd_parts: list[str]) -> tuple[str, str, int]:
    """Run subprocess with real-time stdout streaming, returning captured output."""
    proc: subprocess.Popen[str] = subprocess.Popen(
        cmd_parts,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        stdout_lines.append(line)

    assert proc.stderr is not None
    stderr_text: str = proc.stderr.read()
    proc.wait()

    return "".join(stdout_lines), stderr_text, proc.returncode
