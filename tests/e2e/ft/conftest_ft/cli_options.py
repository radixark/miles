# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

from typing import Annotated

import typer
from tests.utils.soak.ft.types import FaultTrigger

ModeOption = Annotated[str, typer.Option(help="Test mode variant")]
OptionalModeOption = Annotated[
    str | None, typer.Option(help="Test mode variant; a suite whose scenario fixes its topology takes none")
]
SeedOption = Annotated[int, typer.Option(help="Random seed for fault injection")]
PhaseOption = Annotated[str, typer.Option(help="Phase name (multi-phase tests)")]
DumpDirOption = Annotated[str | None, typer.Option(help="Dump base directory")]
EnableDumperOption = Annotated[bool, typer.Option(help="Enable dumper output")]
NumStepsOption = Annotated[int, typer.Option(help="Number of train() calls")]
NumRolloutOption = Annotated[int, typer.Option(help="Number of rollouts")]
MetricThresholdOption = Annotated[float, typer.Option(help="eval/gsm8k accuracy threshold")]
FullyAsyncOption = Annotated[bool, typer.Option(help="Train through train_async.py with --fully-async")]
FaultTriggersOption = Annotated[
    list[FaultTrigger] | None,
    typer.Option(
        "--fault-triggers",
        help="Fault triggers to draw from: timer lands a fault at a random wall-clock instant, hook when a worker "
        "reaches a named fault hook after a random delay (default: timer)",
    ),
]
TrainerCrashIntervalSecondsOption = Annotated[float, typer.Option(help="Mean seconds between trainer cell injections")]
RolloutCrashIntervalSecondsOption = Annotated[
    float, typer.Option(help="Mean seconds between rollout engine injections")
]
