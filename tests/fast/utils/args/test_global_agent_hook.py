from types import SimpleNamespace

from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config
from tests.fast.utils.args.agent_hook_fixtures import _AgentHookConfig

from miles.utils.args.runtime import RolloutConfig
from miles.utils.function_registry import function_registry


class TestGlobalAgentHook:
    def test_agent_arguments_are_discovered_without_selecting_the_agentic_rollout_plugin(self) -> None:
        """A globally declared agent hook registers and transports its own configuration independently."""
        path = "global-agent-hook-probe"
        with function_registry.temporary(path, SimpleNamespace(config_class=_AgentHookConfig)):
            config = parse_fsdp_test_config("--custom-agent-function-path", path, "--agent-probe-threshold", "0.75")
            rollout = RolloutConfig.slice_from(config)
            restored = RolloutConfig.model_validate_json(rollout.model_dump_json())

        assert restored.custom_agent_function_path.path == path
        assert isinstance(restored.custom_agent_function_path.config, _AgentHookConfig)
        assert restored.custom_agent_function_path.config.agent_probe_threshold == 0.75
        assert "agent_probe_threshold" not in config.model_dump()
        assert "agentic_tool_call" not in restored.rollout_function_path.path
