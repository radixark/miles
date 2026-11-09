import logging
from typing import Any

from unittest.mock import AsyncMock

import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config
from tests.fast.fixtures.hook_consumer_fixtures import _hook, _HookConfig

from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.args.runtime import OrchestratorConfig
from miles.utils.function_registry import function_registry, load_function
from miles.utils.hot_restart import init_or_reset_inference_controller
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
        args = ImmutableNamespace(
            custom_agent_function_path=_hook("test86.agent.run"), rollout_dataset_agent_functions=[]
        )
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

    async def test_a_dataset_agent_function_gets_its_own_abort(self) -> None:
        """An agent chosen by a mixed dataset kept issuing requests on abort when only the run-wide agent was aborted."""
        args = ImmutableNamespace(
            custom_agent_function_path=_hook("test86.global.run"),
            rollout_dataset_agent_functions=[_hook("test86.swe.run")],
        )
        aborted: list[str] = []

        async def abort_global(view: ImmutableNamespace) -> None:
            aborted.append("global")

        async def abort_swe(view: ImmutableNamespace) -> None:
            assert view.hook_marker == 41
            aborted.append("swe")

        with function_registry.temporary("test86.global.abort", abort_global):
            with function_registry.temporary("test86.swe.abort", abort_swe):
                await call_agent_abort_hook(args)

        assert aborted == ["global", "swe"]

    async def test_a_removed_mix_file_does_not_stop_dataset_agent_teardown(self, tmp_path) -> None:
        """Teardown uses the agents resolved at startup, never the live YAML."""
        args = ImmutableNamespace(
            custom_agent_function_path=None,
            rollout_data_yaml_config=str(tmp_path / "removed.yaml"),
            rollout_dataset_agent_functions=[_hook("test86.swe.run")],
        )
        aborted: list[int] = []

        async def abort_swe(view: ImmutableNamespace) -> None:
            aborted.append(view.hook_marker)

        with function_registry.temporary("test86.swe.abort", abort_swe):
            await call_agent_abort_hook(args)

        assert aborted == [41]

    async def test_hot_restart_aborts_a_typed_dataset_agent(self, tmp_path) -> None:
        """Hot restart passes the orchestrator config, which must carry the resolved dataset agents and their fields."""
        config = tmp_path / "mix.yaml"
        config.write_text(
            "datasets:\n  a:\n    path: /data/a.jsonl\n    samples_per_step: 1\n    custom_agent_function_path: test86.dsagent.run\n"
        )
        aborted: list[int] = []

        async def agent(**kwargs: Any) -> None:
            pass

        agent.config_class = _HookConfig

        async def abort(view: ImmutableNamespace) -> None:
            aborted.append(view.hook_marker)

        with function_registry.temporary("test86.dsagent.run", agent):
            with function_registry.temporary("test86.dsagent.abort", abort):
                config_all = parse_fsdp_test_config(
                    "--rollout-data-yaml-config",
                    str(config),
                    "--custom-generate-function-path",
                    "miles.rollout.generate_hub.agentic_tool_call.generate",
                )
                args = OrchestratorConfig.slice_from(config_all)
                controller = AsyncMock()
                controller.is_initialized.return_value = True
                await init_or_reset_inference_controller(controller, args=args, trainers=[])

        assert aborted == [41]

    async def test_two_agents_in_one_module_each_get_their_own_abort(self) -> None:
        """Abort was deduplicated by module, so a second agent with a different typed config was never torn down."""
        first, second = _hook("test86.shared.run_a"), _hook("test86.shared.run_b")
        second = second.model_copy(update={"config": second.config.model_copy(update={"hook_marker": 7})})
        args = ImmutableNamespace(custom_agent_function_path=first, rollout_dataset_agent_functions=[second])
        markers: list[int] = []

        async def abort(view: ImmutableNamespace) -> None:
            markers.append(view.hook_marker)

        with function_registry.temporary("test86.shared.abort", abort):
            await call_agent_abort_hook(args)
        assert sorted(markers) == [7, 41]

    @pytest.mark.parametrize("path", [None, "single_component", "missing_test86_package.agent"])
    async def test_absent_agent_or_abort_callback_does_not_require_plugin_state(self, path: str | None) -> None:
        """Optional abort teardown tolerates absent configuration and integrations with no sibling module."""
        args = ImmutableNamespace(
            custom_agent_function_path=None if path is None else _hook(path), rollout_dataset_agent_functions=[]
        )
        await call_agent_abort_hook(args)
