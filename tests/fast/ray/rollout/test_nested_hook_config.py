import pytest
from pydantic import ValidationError

from miles.ray.rollout.rollout_executor import _compute_rollout_function_config
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.eval_config import EvalDatasetConfig


def _hook(path: str, **values: object) -> CustomFunctionConfig:
    return CustomFunctionConfig(path=path, config=ImmutableNamespace(**values))


def _args(*, dataset_hook: CustomFunctionConfig | None) -> ImmutableNamespace:
    return ImmutableNamespace(
        custom_generate_function_path=_hook("generate", generate_limit=2),
        custom_agent_function_path=_hook("agent", agent_limit=3),
        custom_rm_path=_hook("reward", reward_limit=4),
        eval_datasets=[EvalDatasetConfig(name="eval", path="eval.jsonl", custom_generate_function_path=dataset_hook)],
    )


class TestNestedHookConfig:
    def test_rollout_boundary_merges_all_nested_owners_without_changing_them(self) -> None:
        """The rollout entry exposes generate, agent, reward and dataset hook fields together."""
        dataset_hook = _hook("dataset", dataset_limit=5)
        args = _args(dataset_hook=dataset_hook)
        before = dict(args)
        function = _hook("rollout", rollout_limit=1)

        view = _compute_rollout_function_config(args, function)

        assert (view.rollout_limit, view.generate_limit, view.agent_limit, view.reward_limit, view.dataset_limit) == (
            1,
            2,
            3,
            4,
            5,
        )
        assert dict(args) == before
        assert dataset_hook.config.dataset_limit == 5
        with pytest.raises(ValidationError, match="frozen"):
            view.dataset_limit = 6

    def test_conflicting_dataset_hook_field_is_rejected_at_the_rollout_boundary(self) -> None:
        """A dataset hook cannot silently override the active rollout hook configuration."""
        args = _args(dataset_hook=_hook("dataset", rollout_limit=9))

        with pytest.raises(ValueError, match="rollout_limit.*conflicting values"):
            _compute_rollout_function_config(args, _hook("rollout", rollout_limit=1))

    def test_dataset_without_a_custom_generator_keeps_other_nested_owners(self) -> None:
        """An absent per-dataset generator does not suppress the shared generation configuration."""
        view = _compute_rollout_function_config(_args(dataset_hook=None), CustomFunctionConfig(path="rollout"))

        assert view.generate_limit == 2
        assert view.agent_limit == 3
        assert view.reward_limit == 4


class TestEvalDatasetHookParsing:
    def test_dataset_generator_paths_are_wrapped_without_mutating_raw_input(self) -> None:
        """Evaluation configuration owns generator paths while absent dataset hooks remain absent."""
        from miles.utils.eval_config import build_eval_dataset_configs

        raw = [
            {"name": "custom", "path": "a.jsonl", "custom_generate_function_path": "custom.generate"},
            {"name": "plain", "path": "b.jsonl"},
        ]

        datasets = build_eval_dataset_configs(args=ImmutableNamespace(), raw_config=raw, defaults={})

        assert datasets[0].custom_generate_function_path == CustomFunctionConfig(path="custom.generate")
        assert datasets[1].custom_generate_function_path is None
        assert raw == [
            {"name": "custom", "path": "a.jsonl", "custom_generate_function_path": "custom.generate"},
            {"name": "plain", "path": "b.jsonl"},
        ]
