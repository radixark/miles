import argparse
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.event_log import EventLog

from miles.utils.arguments import (
    _resolve_ft_components,
    _resolve_mini_ft_controller_enable,
    get_miles_extra_args_provider,
    supports_partial_target_weight_update,
)
from miles.utils.external_utils.command_utils.base_backend import (
    BaseCommandBackend,
    ExecuteTrainConfig,
    ExecuteTrainRequest,
    LaunchGuard,
)

SCENARIO_RUN_ID: str = "260101-000000-000"


@dataclass(frozen=True)
class CapturedLaunch:
    request: ExecuteTrainRequest
    config: ExecuteTrainConfig
    guard: LaunchGuard | None

    @property
    def argv(self) -> list[str]:
        return shlex.split(self.request.train_args)

    def value_of(self, flag: str) -> str:
        argv = self.argv
        assert argv.count(flag) == 1, f"{flag} appears {argv.count(flag)} times in {argv}"
        return argv[argv.index(flag) + 1]


@dataclass
class FakeSoakRunner:
    event_log: EventLog
    forms: dict[str, Any]
    config: SoakRunnerConfig


@dataclass
class ScenarioHarness:
    dumps_root: Path
    launch_error: BaseException | None = None
    launches: list[CapturedLaunch] = field(default_factory=list)
    soaks: list[dict[str, Any]] = field(default_factory=list)
    prepared: list[dict[str, Any]] = field(default_factory=list)
    checks: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    async def run_soak(self, **kwargs: Any) -> FakeSoakRunner:
        self.soaks.append(kwargs)
        await kwargs["sut_run"]
        return FakeSoakRunner(event_log=kwargs["event_log"], forms=kwargs["forms"], config=kwargs["runner_config"])

    def execute_train_inner(
        self, *, request: ExecuteTrainRequest, config: ExecuteTrainConfig, guard: LaunchGuard | None
    ) -> None:
        self.launches.append(CapturedLaunch(request=request, config=config, guard=guard))
        if self.launch_error is not None:
            raise self.launch_error

    def record_prepare(self, *args: Any, **kwargs: Any) -> None:
        self.prepared.append({"args": args, **kwargs})

    def record_backend_prepare(self, backend: BaseCommandBackend) -> None:
        self.prepared.append({"config": backend.config})

    def recorder(self, name: str) -> Callable[..., None]:
        def record(*args: Any, **kwargs: Any) -> None:
            self.checks.append((name, args, kwargs))

        return record

    def spy(self, name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def record(*args: Any, **kwargs: Any) -> Any:
            self.checks.append((name, args, kwargs))
            return function(*args, **kwargs)

        return record

    def calls_of(self, name: str) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        return [(args, kwargs) for called, args, kwargs in self.checks if called == name]

    @property
    def checker_names(self) -> list[str]:
        return [name for name, _, _ in self.checks]


@dataclass(frozen=True)
class ParsedFaultToleranceArgs:
    namespace: argparse.Namespace
    ft_components: list[str]
    mini_ft_controller_enable: bool
    partial_target_weight_update: bool


def parse_fault_tolerance_args(train_args: str) -> ParsedFaultToleranceArgs:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    get_miles_extra_args_provider()(parser)
    namespace, _ = parser.parse_known_args(shlex.split(train_args))
    ft_components = _resolve_ft_components(namespace)
    namespace.ft_components = ft_components
    return ParsedFaultToleranceArgs(
        namespace=namespace,
        ft_components=ft_components,
        mini_ft_controller_enable=_resolve_mini_ft_controller_enable(namespace),
        partial_target_weight_update=supports_partial_target_weight_update(namespace),
    )
