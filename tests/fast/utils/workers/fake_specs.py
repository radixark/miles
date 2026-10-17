from collections.abc import Callable
from typing import Any, ClassVar, Self

from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.runtime_base import BaseLeafConfig
from miles.utils.workers.worker_spec import (
    BaseCommandSpec,
    BaseServeSpec,
    LaunchCommandContext,
    SchedulingSpec,
    WorkerCtorContext,
    WorkerLaunchContext,
)


def _no_env_vars(_ctx: WorkerLaunchContext) -> dict[str, str]:
    return {}


def _no_ctor_kwargs(_ctx: WorkerCtorContext) -> dict[str, Any]:
    return {}


class FakeCommandSpec(BaseCommandSpec):
    args: Any = None
    fixed_scheduling: SchedulingSpec
    env_vars: Callable[[WorkerLaunchContext], dict[str, str]] = _no_env_vars
    command: Callable[[LaunchCommandContext], str]

    @classmethod
    def create(cls, config: Any) -> Self:
        raise NotImplementedError(f"{cls.__name__} is built directly by the test that uses it")

    def scheduling(self, scaling: ScalingConfig) -> SchedulingSpec:
        return self.fixed_scheduling

    def env_var(self, ctx: WorkerLaunchContext) -> dict[str, str]:
        return self.env_vars(ctx)

    def launch_command(self, ctx: LaunchCommandContext) -> str:
        return self.command(ctx)


class FakeServeSpec(BaseServeSpec):
    worker_type: ClassVar[str] = "fake"
    config_class: ClassVar[type[BaseLeafConfig]] = BaseLeafConfig
    args: Any = None
    fixed_scheduling: SchedulingSpec
    env_vars: Callable[[WorkerLaunchContext], dict[str, str]] = _no_env_vars
    make_ctor_kwargs: Callable[[WorkerCtorContext], dict[str, Any]] = _no_ctor_kwargs

    @classmethod
    def create(cls, config: Any) -> Self:
        raise NotImplementedError(f"{cls.__name__} is built directly by the test that uses it")

    def scheduling(self, scaling: ScalingConfig) -> SchedulingSpec:
        return self.fixed_scheduling

    def env_var(self, ctx: WorkerLaunchContext) -> dict[str, str]:
        return self.env_vars(ctx)

    def ctor_kwargs(self, ctx: WorkerCtorContext) -> dict[str, Any]:
        return self.make_ctor_kwargs(ctx)
