import logging
from typing import Any

import pytest
from tests.fast.fixtures.hook_consumer_fixtures import _hook

from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.function_registry import function_registry, load_function
from miles.utils.misc import call_agent_abort_hook


class TestTypedFunctionPaths:
    def test_typed_paths_resolve_the_registered_callable_without_losing_sync_validation(self) -> None:
        """Typed hook paths share registry lookup and synchronous validation with string paths."""

        async def callback() -> None:
            pass

        hook = _hook()
        with function_registry.temporary(hook.path, callback):
            assert load_function(hook) is callback
            with pytest.raises(ValueError):
                load_function(hook, sync_required=True)


class TestTypedAgentAbort:
    @pytest.mark.parametrize("fails", [False, True])
    async def test_sibling_abort_receives_plugin_fields_and_reports_errors(
        self, caplog: pytest.LogCaptureFixture, fails: bool
    ) -> None:
        """The sibling abort callback receives the same plugin settings and failed teardown remains visible."""
        args = ImmutableNamespace(custom_agent_function_path=_hook("test86.agent.run"))
        seen: list[Any] = []

        async def abort(view: ImmutableNamespace) -> None:
            assert view.hook_marker == 41
            seen.append(view)
            if fails:
                raise RuntimeError("provider shutdown failed")

        with function_registry.temporary("test86.agent.abort", abort), caplog.at_level(logging.WARNING):
            await call_agent_abort_hook(args)

        assert len(seen) == 1
        assert ("provider shutdown failed" in caplog.text) is fails
        assert "hook_marker" not in dict(args)

    @pytest.mark.parametrize("path", [None, "single_component", "missing_test86_package.agent"])
    async def test_absent_agent_or_abort_callback_does_not_require_plugin_state(self, path: str | None) -> None:
        """Optional abort teardown tolerates absent configuration and integrations with no sibling module."""
        args = ImmutableNamespace(custom_agent_function_path=None if path is None else _hook(path))
        await call_agent_abort_hook(args)
