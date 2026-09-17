from typing import Annotated

import typer


NumRolloutOption = Annotated[int, typer.Option(help="Number of rollouts")]

SeedOption = Annotated[int, typer.Option(help="Random seed for fault injection")]
