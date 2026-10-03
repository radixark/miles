import os
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from tests.ci.ci_utils import CI_GATE_RECORD_DIR_ENV, TestFile, reaping_is_isolated, run_unittest_files

app = typer.Typer(add_completion=False)


@app.command()
def main(
    test_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    timeout_seconds: Annotated[int, typer.Option(min=1)],
) -> None:
    if not os.environ.get(CI_GATE_RECORD_DIR_ENV):
        os.environ[CI_GATE_RECORD_DIR_ENV] = tempfile.mkdtemp(prefix="miles-ci-gate-")

    raise typer.Exit(
        code=run_unittest_files(
            [TestFile(name=str(test_file), estimated_time=0)],
            timeout_per_file=timeout_seconds,
            enable_retry=False,
            continue_on_error=False,
            gate_store=None,
            gate_write_baseline=False,
            reap_leftovers=reaping_is_isolated(),
        )
    )


if __name__ == "__main__":
    app()
