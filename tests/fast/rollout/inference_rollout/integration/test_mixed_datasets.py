"""A train rollout over `--rollout-data-yaml-config` draws its groups from every dataset by weight."""

from collections import Counter

import pytest
from tests.fast.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_train

from miles.rollout.generate_hub import single_turn

_MIXED_DATASETS = {
    "math": ([{"input": "What is 1+7?", "label": "8"}] * 3, {"samples_per_step": 1}),
    "science": ([{"input": "What is 1+7?", "label": "8"}] * 5, {"samples_per_step": 3}),
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
def test_train_rollout_mixes_datasets_by_weight(rollout_env):
    out = load_and_call_train(rollout_env.args, rollout_env.data_source)

    groups = out.samples
    assert len(groups) == 4 and all(len(group) == 2 for group in groups)
    assert Counter(group[0].metadata["dataset_name"] for group in groups) == {"math": 1, "science": 3}
    for group in groups:
        assert len({sample.metadata["dataset_name"] for sample in group}) == 1
        assert [sample.reward for sample in group] == [1, 1]


_GENERATE_CALLS: Counter = Counter()
_THIS_MODULE = "tests.fast.rollout.inference_rollout.integration.test_mixed_datasets"


async def counted_generate(input):
    _GENERATE_CALLS[input.sample.metadata["dataset_name"]] += 1
    return await single_turn.generate(input)


@pytest.mark.parametrize(
    "rollout_env",
    [
        pytest.param(
            RolloutEnvConfig(
                extra_argv=MODULAR_ROLLOUT_BASE_ARGV + extra_argv_for_variant("single_turn") + _BATCH_ARGV,
                mixed_datasets={
                    "math": _MIXED_DATASETS["math"],
                    "science": (
                        _MIXED_DATASETS["science"][0],
                        {"samples_per_step": 3, "generate_function": f"{_THIS_MODULE}.counted_generate"},
                    ),
                },
            ),
            id="new_rollout",
        )
    ],
    indirect=True,
)
def test_a_dataset_generate_function_runs_only_on_its_samples(rollout_env):
    _GENERATE_CALLS.clear()
    load_and_call_train(rollout_env.args, rollout_env.data_source)

    assert _GENERATE_CALLS == {"science": 6}
