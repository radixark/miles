from typing import Any

import pytest
from tests.fast.fixtures.hook_consumer_fixtures import _hook

from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.function_registry import function_registry
from miles.utils.types import RewardSpec, Sample


class TestTypedRewardHooks:
    @pytest.mark.parametrize("batched", [False, True])
    async def test_resolved_reward_hook_receives_custom_fields_and_original_sample_identity(
        self, batched: bool
    ) -> None:
        """Single and batched reward dispatch merge typed plugin values without changing launch configuration."""
        hook = _hook()
        args = ImmutableNamespace(custom_rm_path=hook, rm_type=None, rollout_temperature=0.7)
        sample = Sample(prompt="input")
        seen: list[Any] = []

        async def reward(view: ImmutableNamespace, value: Any, **kwargs: Any) -> Any:
            assert view.hook_marker == 41
            assert view.rollout_temperature == 0.7
            seen.append(value)
            return [0.9] if batched else 0.9

        with function_registry.temporary(hook.path, reward):
            result = await batched_async_rm(args, [sample]) if batched else await async_rm(args, sample)

        assert result == ([0.9] if batched else 0.9)
        assert seen[0][0] is sample if batched else seen[0] is sample
        assert "hook_marker" not in dict(args)

    async def test_sample_reward_spec_string_overrides_run_hook_without_reusing_its_custom_fields(self) -> None:
        """A sample-level legacy path selects its own callback instead of leaking the run's resolved config."""
        args = ImmutableNamespace(custom_rm_path=_hook("test86.run"), rm_type=None)
        sample = Sample(prompt="input", reward_spec=RewardSpec(custom_rm_path="test86.sample"))

        async def reward(view: ImmutableNamespace, value: Sample) -> float:
            assert "hook_marker" not in dict(view)
            assert value is sample
            return 0.25

        with function_registry.temporary("test86.sample", reward):
            assert await async_rm(args, sample) == 0.25
        assert args.custom_rm_path.path == "test86.run"
