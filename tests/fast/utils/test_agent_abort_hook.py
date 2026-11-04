from types import ModuleType

from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config
from tests.fast.utils.abort_hook_fixtures import abort_plugin

from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.misc import call_agent_abort_hook

__all__ = ["abort_plugin"]


class TestAgentAbortHook:
    async def test_abort_imports_the_sibling_hook_and_awaits_it_with_the_resolved_custom_config(
        self, abort_plugin: ModuleType
    ) -> None:
        """The imported sibling abort receives both parsed runtime values and its plugin schema values."""
        args = parse_fsdp_test_config(
            "--custom-agent-function-path",
            "tests.fast.utils.abort_hook_fixtures.run_agent",
            "--abort-marker",
            "37",
            "--num-rollout",
            "5",
        )

        await call_agent_abort_hook(args)

        [received] = abort_plugin.calls
        assert isinstance(received, ImmutableNamespace)
        assert received.abort_marker == 37
        assert received.num_rollout == 5
        assert received.train_backend == "fsdp"
        assert args.custom_agent_function_path.config.abort_marker == 37
