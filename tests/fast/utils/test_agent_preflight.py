"""An agent function's ``preflight`` hook is found like ``abort``, but its error fails the run."""

import pytest
from tests.fast.fixtures.hook_consumer_fixtures import _hook

from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.function_registry import function_registry
from miles.utils.misc import call_agent_preflight_hook


async def test_preflight_receives_plugin_fields_and_its_error_fails_the_run() -> None:
    """A setup check that only logged its failure would leave the rollout dropping every sample."""
    args = ImmutableNamespace(
        custom_agent_function_path=_hook("test_preflight.agent.run"), rollout_dataset_agent_functions=[]
    )
    markers: list[int] = []

    async def preflight(view: ImmutableNamespace) -> None:
        markers.append(view.hook_marker)
        raise RuntimeError("the agent's tool is not on PATH")

    with function_registry.temporary("test_preflight.agent.preflight", preflight):
        with pytest.raises(RuntimeError, match="not on PATH"):
            await call_agent_preflight_hook(args)
    assert markers == [41]


async def test_every_agent_the_run_can_call_is_checked_once() -> None:
    """A dataset's own agent runs samples too, while a repeat of the same agent needs no second check."""
    args = ImmutableNamespace(
        custom_agent_function_path=_hook("test_preflight.global.run"),
        rollout_dataset_agent_functions=[_hook("test_preflight.swe.run"), _hook("test_preflight.global.run")],
    )
    checked: list[str] = []

    async def preflight_global(view: ImmutableNamespace) -> None:
        checked.append("global")

    async def preflight_swe(view: ImmutableNamespace) -> None:
        checked.append("swe")

    with function_registry.temporary("test_preflight.global.preflight", preflight_global):
        with function_registry.temporary("test_preflight.swe.preflight", preflight_swe):
            await call_agent_preflight_hook(args)
    assert checked == ["global", "swe"]


@pytest.mark.parametrize("path", [None, "miles.utils.misc.split_evenly", "missing_test_preflight_package.agent.run"])
async def test_an_agent_without_a_preflight_hook_starts_unchecked(path: str | None) -> None:
    """Agent functions written before the hook existed must still start."""
    args = ImmutableNamespace(
        custom_agent_function_path=None if path is None else _hook(path), rollout_dataset_agent_functions=[]
    )
    await call_agent_preflight_hook(args)
