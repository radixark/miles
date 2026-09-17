# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

from typing import Annotated

import typer

ModeOption = Annotated[str, typer.Option(help="Test mode variant")]
OptionalModeOption = Annotated[
    str | None, typer.Option(help="Test mode variant; a suite whose scenario fixes its topology takes none")
]
PhaseOption = Annotated[str, typer.Option(help="Phase name (multi-phase tests)")]
DumpDirOption = Annotated[str | None, typer.Option(help="Dump base directory")]
EnableDumperOption = Annotated[bool, typer.Option(help="Enable dumper output")]
NumStepsOption = Annotated[int, typer.Option(help="Number of train() calls")]
FullyAsyncOption = Annotated[bool, typer.Option(help="Train through train_async.py with --fully-async")]
TrainerCrashIntervalSecondsOption = Annotated[float, typer.Option(help="Mean seconds between trainer cell injections")]
RolloutCrashIntervalSecondsOption = Annotated[
    float, typer.Option(help="Mean seconds between rollout engine injections")
]
