# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from tests.e2e.ft.conftest_ft.execution import get_common_train_args, prepare, run_training
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode


BuildArgsFn = Callable[[FTTestMode, str, bool], str]


def resolve_dump_dir(test_name: str) -> str:
    # TODO make it configurable, but on local disk instead of remote disk
    output_dir = "/node_public"
    dump_dir = str(Path(output_dir) / "dumps" / test_name)
    os.makedirs(dump_dir, exist_ok=True)
    return dump_dir


def _dump_subdir(side: str, phase: str) -> str:
    return f"{side}/{phase}" if phase else side


def run_pipeline(
    *,
    test_name: str,
    build_baseline_args: BuildArgsFn,
    build_target_args: BuildArgsFn,
    compare_fn: Callable[[str, FTTestMode], None],
    phases: list[str] | None,
    mode: str,
    enable_dumper: bool = True,
) -> None:
    """Full pipeline (prepare + every phase's baseline/target + compare) for one mode."""
    effective_phases: list[str] = phases or [""]
    ft_mode: FTTestMode = resolve_mode(mode)
    dump_dir: str = resolve_dump_dir(test_name)
    print(f"Dump directory: {dump_dir}")

    prepare(ft_mode)

    for phase in effective_phases:
        baseline_dump = f"{dump_dir}/{_dump_subdir('baseline', phase)}"
        run_training(
            train_args=build_baseline_args(ft_mode, baseline_dump, enable_dumper),
            mode=ft_mode,
            dump_dir=baseline_dump,
        )

        target_dump = f"{dump_dir}/{_dump_subdir('target', phase)}"
        run_training(
            train_args=build_target_args(ft_mode, target_dump, enable_dumper),
            mode=ft_mode,
            dump_dir=target_dump,
        )

    if enable_dumper:
        compare_fn(dump_dir, ft_mode)


def create_comparison_app_and_run_ci(
    *,
    test_name: str,
    build_baseline_args: BuildArgsFn,
    build_target_args: BuildArgsFn,
    compare_fn: Callable[[str, FTTestMode], None],
    phases: list[str] | None = None,
) -> tuple[typer.Typer, Callable[[str], None]]:
    """Build, from one wiring, the manual typer app and a run_ci(mode) one-shot runner.

    Returns ``(app, run_ci)``: ``app`` exposes run/baseline/target/compare for manual use;
    ``run_ci(mode)`` runs the full pipeline for a single mode (used by the per-mode CI entry
    files), writing dumps under a per-mode test name so concurrent CI modes don't collide.

    For simple (no-phase) tests, leave phases empty.
    For multi-phase tests (e.g. with_failure), provide phase names like ["phase_a", "phase_b"].
    """
    app: typer.Typer = typer.Typer()

    def _run_side(
        side: str,
        build_fn: BuildArgsFn,
        mode: str,
        dump_dir: str | None,
        phase: str,
        *,
        enable_dumper: bool = True,
    ) -> None:
        ft_mode = resolve_mode(mode)
        if dump_dir is None:
            dump_dir = resolve_dump_dir(test_name)
        sub = _dump_subdir(side, phase)
        full_dump_dir = f"{dump_dir}/{sub}"
        args = build_fn(ft_mode, full_dump_dir, enable_dumper)
        prepare(ft_mode)
        run_training(train_args=args, mode=ft_mode, dump_dir=full_dump_dir)

    @app.command()
    def baseline(
        mode: Annotated[str, typer.Option(help="Test mode variant")],
        dump_dir: Annotated[str | None, typer.Option(help="Dump base directory")] = None,
        phase: Annotated[str, typer.Option(help="Phase name (multi-phase tests)")] = "",
        enable_dumper: Annotated[bool, typer.Option(help="Enable dumper output")] = True,
    ) -> None:
        """Run baseline (normal DP) training."""
        _run_side("baseline", build_baseline_args, mode, dump_dir, phase, enable_dumper=enable_dumper)

    @app.command()
    def target(
        mode: Annotated[str, typer.Option(help="Test mode variant")],
        dump_dir: Annotated[str | None, typer.Option(help="Dump base directory")] = None,
        phase: Annotated[str, typer.Option(help="Phase name (multi-phase tests)")] = "",
        enable_dumper: Annotated[bool, typer.Option(help="Enable dumper output")] = True,
    ) -> None:
        """Run target (indep_dp) training."""
        _run_side("target", build_target_args, mode, dump_dir, phase, enable_dumper=enable_dumper)

    @app.command()
    def compare(
        mode: Annotated[str, typer.Option(help="Test mode variant")],
        dump_dir: Annotated[str, typer.Option(help="Dump base directory")],
    ) -> None:
        """Compare baseline and target dumps."""
        ft_mode = resolve_mode(mode)
        compare_fn(dump_dir, ft_mode)

    @app.command()
    def run(
        mode: Annotated[str, typer.Option(help="Test mode variant")],
        enable_dumper: Annotated[bool, typer.Option(help="Enable dumper output")] = True,
    ) -> None:
        """Full pipeline: prepare + all phases + compare."""
        run_pipeline(
            test_name=test_name,
            build_baseline_args=build_baseline_args,
            build_target_args=build_target_args,
            compare_fn=compare_fn,
            phases=phases,
            mode=mode,
            enable_dumper=enable_dumper,
        )

    @app.command()
    def generate_data(
        mode: Annotated[str, typer.Option(help="Test mode variant (must have real rollout)")],
        num_steps: Annotated[int, typer.Option(help="Number of rollout steps to generate")] = 12,
        output_dir: Annotated[
            str, typer.Option(help="Output directory for rollout data")
        ] = "/tmp/generated_rollout_data",
    ) -> None:
        """Generate debug rollout data using real rollout (no dumper)."""
        ft_mode = resolve_mode(mode)
        assert ft_mode.has_real_rollout, f"Mode {mode} does not have real rollout engines"
        prepare(ft_mode)
        args = get_common_train_args(ft_mode, dump_dir=output_dir, num_steps=num_steps, enable_dumper=False)
        run_training(train_args=args, mode=ft_mode)

    def run_ci(mode: str) -> None:
        """Run one mode's full pipeline (entry point for the per-mode CI files)."""
        run_pipeline(
            test_name=f"{test_name}_{mode}",
            build_baseline_args=build_baseline_args,
            build_target_args=build_target_args,
            compare_fn=compare_fn,
            phases=phases,
            mode=mode,
        )

    return app, run_ci


def create_non_comparison_app(
    *,
    test_name: str,
    build_args: Callable[[FTTestMode, str], str],
    verify_fn: Callable[[str, FTTestMode], None] | None = None,
) -> typer.Typer:
    """Generate a typer app with a single 'run' command for non-comparison tests."""
    app: typer.Typer = typer.Typer()

    @app.command()
    def run(
        mode: Annotated[str, typer.Option(help="Test mode variant")],
    ) -> None:
        """Full pipeline: prepare + execute + verify."""
        ft_mode = resolve_mode(mode)
        dump_dir: str = resolve_dump_dir(test_name)
        print(f"Dump directory: {dump_dir}")

        prepare(ft_mode)
        args = build_args(ft_mode, dump_dir)
        run_training(train_args=args, mode=ft_mode)

        if verify_fn is not None:
            verify_fn(dump_dir, ft_mode)

    return app
