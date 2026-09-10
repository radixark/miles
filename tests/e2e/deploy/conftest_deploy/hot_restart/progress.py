import sys
from pathlib import Path

import typer
from pydantic import TypeAdapter
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import RunProgress, read_run_progress
from tests.utils.soak.action import run_command

_adapter = TypeAdapter(RunProgress)
app = typer.Typer()


async def observe_run_progress(*, checkpoint_dir: Path, events_dir: Path, timeout_seconds: float) -> RunProgress:
    result = await run_command(
        [
            sys.executable,
            "-m",
            "tests.e2e.deploy.conftest_deploy.hot_restart.progress",
            str(checkpoint_dir),
            str(events_dir),
        ],
        timeout_seconds=timeout_seconds,
    )
    return _adapter.validate_json(result.stdout)


@app.command()
def main(checkpoint_dir: Path, events_dir: Path) -> None:
    progress = read_run_progress(checkpoint_dir=checkpoint_dir, events_dir=events_dir)
    sys.stdout.write(_adapter.dump_json(progress).decode())


if __name__ == "__main__":
    app()
