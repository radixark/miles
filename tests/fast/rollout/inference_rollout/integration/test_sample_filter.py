from unittest.mock import Mock

import pytest
from tests.fast.rollout.inference_rollout.integration.utils import (
    filter_by_reward,
    integration_env_config,
    load_and_call_train,
)

from miles.utils.misc import function_registry

# Data with only 2 reward=1 samples out of 4.
# This ensures all 4 samples must be generated to collect 2 valid ones.
_FILTER_TEST_DATA_ROWS = [
    {"input": "What is 1+7?", "label": "8"},  # reward=1
    {"input": "What is 1+8?", "label": "wrong"},  # reward=0
    {"input": "What is 1+9?", "label": "wrong"},  # reward=0
    {"input": "What is 1+6?", "label": "7"},  # reward=1
]

_FILTER_HOOK_ARGV = [
    "--rollout-batch-size",
    "2",
    "--over-sampling-batch-size",
    "4",
    "--dynamic-sampling-filter-path",
    "test:filter_by_reward",
    "--rollout-sample-filter-path",
    "test:sample_filter",
    "--rollout-all-samples-process-path",
    "test:all_samples_process",
]

# Appended after the modular defaults; argparse keeps the last occurrence, so
# this routes the same test through the legacy built-in rollout fn, which must
# equally not re-apply the sample filter the manager now owns.
_LEGACY_ROLLOUT_FN_ARGV = [
    "--rollout-function-path",
    "miles.rollout.sglang_rollout.generate_rollout",
    "--custom-generate-function-path",
    "miles.rollout.sglang_rollout.generate",
]


@pytest.mark.parametrize(
    "rollout_env",
    [
        pytest.param(
            integration_env_config(_FILTER_HOOK_ARGV, data_rows=_FILTER_TEST_DATA_ROWS),
            id="sample_filter_vs_all_samples",
        ),
        pytest.param(
            integration_env_config(_FILTER_HOOK_ARGV + _LEGACY_ROLLOUT_FN_ARGV, data_rows=_FILTER_TEST_DATA_ROWS),
            id="sample_filter_vs_all_samples_legacy_rollout_fn",
        ),
    ],
    indirect=True,
)
def test_sample_filter_and_all_samples_process(rollout_env):
    env = rollout_env
    sample_filter_mock = Mock()
    all_samples_process_mock = Mock()

    with (
        function_registry.temporary("test:filter_by_reward", filter_by_reward),
        function_registry.temporary("test:sample_filter", sample_filter_mock),
        function_registry.temporary("test:all_samples_process", all_samples_process_mock),
    ):
        out = load_and_call_train(env.args, env.data_source)

    # --rollout-sample-filter-path is applied generically in the manager
    # (postprocess_rollout_data), NOT inside the rollout fn, so calling the
    # rollout fn directly here must not invoke the sample filter. See
    # tests/fast/ray/rollout/test_rollout_data_conversion.py for the
    # manager-side coverage of the hoisted filter.
    sample_filter_mock.assert_not_called()

    # --rollout-all-samples-process-path stays in the rollout fn (it needs
    # all_samples / data_source the manager lacks), so it is still applied here.
    all_samples_process_mock.assert_called_once()
    _, all_samples, data_source = all_samples_process_mock.call_args[0]
    assert data_source is not None
    # all_samples is the full over-sampled set (4 prompts); the rollout output
    # keeps only the dynamic-filter survivors (the 2 reward==1 prompts), so
    # all_samples must strictly exceed the trained set.
    trained = [s for group in out.samples for s in group]
    assert len(all_samples) > len(trained) > 0
