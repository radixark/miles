"""``run-and-compare`` CLI command."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import typer

from miles.utils.debug_utils.run_megatron.cli.commands.args import (
    CommonRunArgs,
    CompareArgs,
    RunAndCompareArgs,
    RunArgs,
)
from miles.utils.debug_utils.run_megatron.cli.commands.compare import compare_impl
from miles.utils.debug_utils.run_megatron.cli.commands.run import run_impl
from miles.utils.debug_utils.run_megatron.cli.parallel_utils import ParallelConfig, parse_parallel_args
from miles.utils.typer_utils import dataclass_cli


def register(app: typer.Typer) -> None:
    """Register the ``run-and-compare`` command on *app*."""
    app.command(name="run-and-compare")(run_and_compare)


@dataclass_cli(env_var_prefix="")
def run_and_compare(args: RunAndCompareArgs) -> None:
    """Run baseline + target configs, then compare dumps."""
    baseline_config: ParallelConfig = ParallelConfig.from_parsed_args(parse_parallel_args(args.baseline))
    target_config: ParallelConfig = ParallelConfig.from_parsed_args(parse_parallel_args(args.target))

    baseline_output: Path = args.output_base_dir / baseline_config.dir_name()
    target_output: Path = args.output_base_dir / target_config.dir_name()

    common_fields: dict[str, object] = {f.name: getattr(args, f.name) for f in dataclasses.fields(CommonRunArgs)}

    replay_dir: Path | None = args.output_base_dir / "routing_replay" if args.routing_replay else None

    _run_baseline_and_target(
        baseline_config=baseline_config,
        target_config=target_config,
        baseline_output=baseline_output,
        target_output=target_output,
        replay_dir=replay_dir,
        common_fields=common_fields,
    )

    print("[cli] Comparing baseline vs target", flush=True)
    compare_impl(
        CompareArgs(
            baseline_dir=baseline_output / "standalone",
            target_dir=target_output / "standalone",
            output_format="json",
            grouping="logical",
        )
    )


def _run_baseline_and_target(
    *,
    baseline_config: ParallelConfig,
    target_config: ParallelConfig,
    baseline_output: Path,
    target_output: Path,
    replay_dir: Path | None,
    common_fields: dict[str, object],
) -> None:
    if replay_dir is not None:
        if baseline_config.nproc != 1:
            raise ValueError(f"Routing replay requires single-rank baseline (nproc=1), got {baseline_config}")
        print("[cli] Routing replay enabled", flush=True)

    print("[cli] Step 1/2: Baseline run", flush=True)
    run_impl(
        RunArgs(
            **common_fields,
            **dataclasses.asdict(baseline_config),
            output_dir=baseline_output,
            routing_replay_dump_path=replay_dir,
            routing_replay_load_path=None,
        )
    )

    print("[cli] Step 2/2: Target run", flush=True)
    run_impl(
        RunArgs(
            **common_fields,
            **dataclasses.asdict(target_config),
            output_dir=target_output,
            routing_replay_dump_path=None,
            routing_replay_load_path=replay_dir,
        )
    )
