"""A train rollout over `--rollout-data-yaml-config` mixes by weight, grades per row and caps per dataset."""

from collections import Counter

import pytest
from tests.fast.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_train

from miles.utils.function_registry import function_registry


async def _row_reward(args, sample, **kwargs) -> float:
    return 0.25


_MIXED_DATASETS = {
    "math": ([{"input": "What is 1+7?", "label": "8"}] * 3, {"samples_per_step": 1}),
    "graded": (
        [{"input": "What is 1+7?", "label": "8", "metadata": {"custom_rm_path": "test3947.row_reward"}}] * 5,
        {"samples_per_step": 3, "max_response_len": 8},
    ),
}

_BATCH_ARGV = ["--rollout-batch-size", "4", "--n-samples-per-prompt", "2", "--global-batch-size", "8"]

_VARIANTS = [
    pytest.param(
        RolloutEnvConfig(
            extra_argv=MODULAR_ROLLOUT_BASE_ARGV + extra_argv_for_variant("single_turn") + _BATCH_ARGV,
            mixed_datasets=_MIXED_DATASETS,
        ),
        id="new_rollout",
    ),
    pytest.param(
        RolloutEnvConfig(
            extra_argv=[
                "--rollout-function-path",
                "miles.rollout.sglang_rollout.generate_rollout",
                "--custom-generate-function-path",
                "miles.rollout.sglang_rollout.generate",
                *_BATCH_ARGV,
            ],
            mixed_datasets=_MIXED_DATASETS,
        ),
        id="old_rollout",
    ),
]


@pytest.mark.parametrize("rollout_env", _VARIANTS, indirect=True)
def test_train_rollout_mixes_grades_and_caps_by_dataset(rollout_env):
    env = rollout_env
    with function_registry.temporary("test3947.row_reward", _row_reward):
        out = load_and_call_train(env.args, env.data_source)

    groups = out.samples
    assert len(groups) == 4 and all(len(group) == 2 for group in groups)
    assert Counter(group[0].metadata["dataset_name"] for group in groups) == {"math": 1, "graded": 3}
    for group in groups:
        name = group[0].metadata["dataset_name"]
        assert {sample.metadata["dataset_name"] for sample in group} == {name}
        assert [sample.reward for sample in group] == ([1, 1] if name == "math" else [0.25, 0.25])

    budgets = sorted(request["sampling_params"]["max_new_tokens"] for request in env.mock_server.request_log)
    assert budgets == [8] * 6 + [16] * 2
