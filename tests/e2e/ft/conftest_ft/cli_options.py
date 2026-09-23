# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

from enum import StrEnum
from typing import Annotated

import typer


class PreciseHook(StrEnum):
    NONE = "none"
    ALL_GATHER = "all_gather"
    P2P = "p2p"


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
PreciseOption = Annotated[
    PreciseHook,
    typer.Option(help="Set every fault at this trainer hook: the weight-update all-gather, or the P2P weight send"),
]
MixOption = Annotated[
    bool, typer.Option(help="Draw wall-clock faults alongside the hook faults, with randomly delayed hooks")
]
TrainerCrashIntervalSecondsOption = Annotated[float, typer.Option(help="Mean seconds between trainer cell injections")]
RolloutCrashIntervalSecondsOption = Annotated[
    float, typer.Option(help="Mean seconds between rollout engine injections")
]
