# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import typer

from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE, create_comparison_app_and_run_ci
from tests.e2e.ft.conftest_ft.comparisons import compare_deterministic_sides
from tests.e2e.ft.conftest_ft.execution import get_deterministic_p2p_train_args
from tests.e2e.ft.conftest_ft.fault_hook_events import assert_fault_hooks_fired
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.utils.soak.core.utils import create_soak_config
from tests.utils.soak.ft.checkers.reconfigure import ReconfigureInfo

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils import command_utils
from miles.utils.test_utils.fault_injector.models import FaultHookRequest
from miles.utils.test_utils.fault_injector.static_source import compute_fault_hooks_arg

MIN_TRAINED_ROLLOUTS: int = 2

BuildFaultHooksFn = Callable[[FTTestMode, command_utils.ExecuteTrainConfig], list[FaultHookRequest]]
TargetReconfiguresFn = Callable[[FTTestMode], list[ReconfigureInfo]]
AssertTargetEventsFn = Callable[[Path, FTTestMode], None]


def create_fault_hook_comparison_app(
    *,
    test_name: str,
    num_rollouts: int,
    ft_components: tuple[str, ...],
    extra_train_args: str,
    build_fault_hooks: BuildFaultHooksFn,
    expected_target_reconfigures: TargetReconfiguresFn,
    assert_target_events: AssertTargetEventsFn,
) -> tuple[typer.Typer, Callable[[str | None], None]]:
    def build_args(
        mode: FTTestMode,
        dump_dir: str,
        enable_dumper: bool,
        config: command_utils.ExecuteTrainConfig,
        *,
        is_target: bool,
    ) -> str:
        assert tuple(mode.ft_components) == ft_components, (
            f"{test_name} declares faults for {ft_components}, so the mode must enable ft on exactly those, "
            f"got ft_components={mode.ft_components}"
        )
        args = get_deterministic_p2p_train_args(
            mode, dump_dir=dump_dir, num_steps=num_rollouts, enable_dumper=enable_dumper, test_name=test_name
        )
        args += extra_train_args
        if is_target:
            args += compute_fault_hooks_arg(build_fault_hooks(mode, config))
        return args

    def compare(dump_dir: str, mode: FTTestMode) -> None:
        target_dir = f"{dump_dir}/{TARGET_SIDE}"
        compare_deterministic_sides(
            baseline_dir=f"{dump_dir}/{BASELINE_SIDE}",
            target_dir=target_dir,
            min_trained_rollouts=MIN_TRAINED_ROLLOUTS,
            expected_target_reconfigures=expected_target_reconfigures(mode),
        )

        events_dir = Path(target_dir) / EVENTS_DIRNAME
        request_ids = [request.request_id for request in build_fault_hooks(mode, command_utils.default_config())]
        assert_fault_hooks_fired(events_dir, request_ids=request_ids)
        assert_target_events(events_dir, mode)

        print(f"{test_name} comparison test PASSED")

    return create_comparison_app_and_run_ci(
        test_name=test_name,
        build_baseline_args=partial(build_args, is_target=False),
        build_target_args=partial(build_args, is_target=True),
        compare_fn=compare,
        config_for_side=lambda side, config: create_soak_config(config),
    )
