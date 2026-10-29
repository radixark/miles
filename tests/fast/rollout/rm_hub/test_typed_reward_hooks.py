from typing import Any

import pytest
from tests.fast.fixtures.hook_consumer_fixtures import _hook, _HookConfig

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

    @pytest.mark.parametrize("batched", [False, True])
    async def test_row_naming_the_run_hook_keeps_its_custom_fields(self, batched: bool) -> None:
        """A row naming the configured `--custom-rm-path` must not drop that hook's config."""
        args = ImmutableNamespace(custom_rm_path=_hook("test86.run"), rm_type=None)
        sample = Sample(prompt="input", metadata={"custom_rm_path": "test86.run"})

        async def reward(view: ImmutableNamespace, value: Any) -> Any:
            assert view.hook_marker == 41
            return [0.5] if batched else 0.5

        with function_registry.temporary("test86.run", reward):
            result = await batched_async_rm(args, [sample]) if batched else await async_rm(args, sample)
        assert result == ([0.5] if batched else 0.5)

    async def test_row_naming_another_typed_hook_is_rejected(self) -> None:
        """A row cannot carry a typed hook's config, so naming one fails instead of running it unconfigured."""
        args = ImmutableNamespace(custom_rm_path=_hook("test86.run"), rm_type=None)
        sample = Sample(prompt="input", metadata={"custom_rm_path": "test86.typed"})

        async def typed_reward(view: ImmutableNamespace, value: Sample) -> float:
            pytest.fail("an unconfigured typed hook must not run")

        typed_reward.config_class = _HookConfig
        with function_registry.temporary("test86.typed", typed_reward):
            with pytest.raises(ValueError, match="its own config"):
                await async_rm(args, sample)

    async def test_batched_rejects_a_non_string_row_path_like_the_per_sample_path(self) -> None:
        args = ImmutableNamespace(custom_rm_path=_hook("test86.run"), rm_type=None)
        with pytest.raises(TypeError, match="custom_rm_path"):
            await batched_async_rm(args, [Sample(prompt="input", metadata={"custom_rm_path": {"path": "x"}})])
