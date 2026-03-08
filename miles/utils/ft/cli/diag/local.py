from __future__ import annotations

import asyncio
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import typer

from miles.utils.ft.agents.diagnostics.runner import NodeExecutorRunner
from miles.utils.ft.cli.diag.output import exit_with_results, print_results
from miles.utils.ft.platform.node_agent_factory import build_all_diagnostics

_LOCAL_EXCLUDED = {"nccl_pairwise"}


def local(
    checks: Annotated[list[str] | None, typer.Argument(help="Checks to run (default: all)")] = None,
    timeout: Annotated[int, typer.Option(help="Per-check timeout in seconds")] = 120,
    disk_mounts: Annotated[str, typer.Option(help="Comma-separated mount paths")] = "/",
    num_gpus: Annotated[int, typer.Option(help="Number of GPUs on this node")] = 8,
    xid_since_minutes: Annotated[int, typer.Option(help="Look back N minutes for XID errors")] = 5,
    json_output: Annotated[bool, typer.Option("--json", help="Output results as JSON")] = False,
) -> None:
    """Run diagnostic checks on the local node."""
    mount_paths = [Path(m.strip()) for m in disk_mounts.split(",")]
    xid_since = datetime.now(timezone.utc) - timedelta(minutes=xid_since_minutes)
    diagnostics = build_all_diagnostics(
        num_gpus=num_gpus,
        disk_mounts=mount_paths,
        xid_since=xid_since,
    )

    node_id = socket.gethostname()
    runner = NodeExecutorRunner(node_id=node_id, diagnostics=diagnostics)

    local_defaults = [t for t in runner.available_types if t not in _LOCAL_EXCLUDED]
    selected = checks or local_defaults
    unknown = set(selected) - set(runner.available_types)
    if unknown:
        typer.echo(f"Unknown checks: {', '.join(sorted(unknown))}", err=True)
        raise typer.Exit(code=1)

    results = asyncio.run(runner.run_selected(selected, timeout_seconds=timeout))
    print_results(results, json_output=json_output, node_id=node_id)
    exit_with_results(results)
