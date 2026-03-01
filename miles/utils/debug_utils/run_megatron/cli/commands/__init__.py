"""CLI commands for run_megatron.

Usage:
    python -m miles.utils.debug_utils.run_megatron run ...
    python -m miles.utils.debug_utils.run_megatron compare ...
    python -m miles.utils.debug_utils.run_megatron run-and-compare ...
    python -m miles.utils.debug_utils.run_megatron show-model-args ...
"""

import typer

from miles.utils.debug_utils.run_megatron.cli.commands import (
    command_compare,
    command_run,
    command_run_and_compare,
)

app: typer.Typer = typer.Typer(pretty_exceptions_enable=False)

command_run.register(app)
command_compare.register(app)
command_run_and_compare.register(app)
