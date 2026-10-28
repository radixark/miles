import asyncio
import sys
from types import ModuleType

import pytest

from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.args.schema import A, Arg, BaseConfig


class AbortConfig(BaseConfig):
    abort_marker: A[int, Arg()] = 11


calls: list[ImmutableNamespace] = []


async def run_agent(args: ImmutableNamespace) -> None:
    pass


run_agent.config_class = AbortConfig


async def abort(args: ImmutableNamespace) -> None:
    await asyncio.sleep(0)
    calls.append(args)


@pytest.fixture
def abort_plugin(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "calls", [])
    return module
