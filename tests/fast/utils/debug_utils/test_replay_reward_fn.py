from miles.utils.debug_utils.replay_reward_fn import _main_async
from miles.utils.function_registry import function_registry
from miles.utils.types import Sample


class TestMainAsync:
    async def test_main_async_resolves_a_registry_only_reward_function(self, capsys):
        """A registry-only name is not an importable module path, so replay must resolve it via function_registry."""
        scored = []

        async def reward_fn(args, sample):
            scored.append(sample)
            return 0.25

        samples = [Sample(index=0, prompt="prompt", response="response")]

        with function_registry.temporary("test:replay_rm", reward_fn):
            await _main_async(samples=samples, custom_rm_path="test:replay_rm")

        assert scored == samples
        assert "Reward:   0.25" in capsys.readouterr().out
