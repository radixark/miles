from typing import Any

import pytest
from tests.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fixtures.rollout_integration import IntegrationEnvConfig
from tests.rollout.modular_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_rollout

from miles.utils.test_utils.mock_tools import TwoTurnStub
from miles.utils.types import Sample

TWO_TURN_DATA_ROWS = [{"input": [{"role": "user", "content": TwoTurnStub.USER_QUESTION}], "label": "2008"}]

_VARIANT_NAMES = [
    "multi_turn_single_sample",
    "multi_turn_multi_samples",
    "agentic_tool_call_single_sample",
    "agentic_tool_call_multi_samples",
]


def _config_for_variant(variant: str) -> IntegrationEnvConfig:
    return IntegrationEnvConfig(
        extra_argv=MODULAR_ROLLOUT_BASE_ARGV
        + extra_argv_for_variant(variant)
        + ["--rollout-batch-size", "2", "--n-samples-per-prompt", "2", "--apply-chat-template"],
        data_rows=TWO_TURN_DATA_ROWS,
    )


@pytest.mark.parametrize(
    "rollout_integration_env",
    [pytest.param(_config_for_variant(variant), id=variant) for variant in _VARIANT_NAMES],
    indirect=True,
)
@pytest.mark.parametrize("test_type", ["train", "eval"])
def test_rollout(rollout_integration_env, request, test_type):
    env = rollout_integration_env
    variant = request.node.callspec.id

    env.mock_server.process_fn = TwoTurnStub.process_fn

    out = load_and_call_rollout(env.args, env.data_source, mode=test_type)

    if test_type == "train":
        assert len(out.samples) == env.args.rollout_batch_size
        group = out.samples[0]
        _verify_samples(variant, group)
    else:
        assert "toy" in out.data
        samples = out.data["toy"]["samples"]
        _verify_samples(variant, samples)


def _verify_samples(variant: str, samples: list[Any]):
    assert len(samples) == 2, f"n_samples_per_prompt=2, so group should have 2 samples, got {len(samples)}"

    if variant in ("multi_turn_multi_samples", "agentic_tool_call_multi_samples"):
        for group_sample in samples:
            assert isinstance(group_sample, list), "multi_samples variant should return list[Sample] per generate"
            assert len(group_sample) == 2, "multi_samples variant should return 2 samples per generate (one per turn)"
            for i, sample in enumerate(group_sample):
                assert sample.status == Sample.Status.COMPLETED
                assert sample.reward == 1, f"Sample {i} should have reward=1"
            assert "2008" in group_sample[-1].response, "Last sample should contain final answer '2008'"
    else:
        for sample in samples:
            assert isinstance(sample, Sample), "single_sample variant should return Sample, not list"
            assert sample.status == Sample.Status.COMPLETED
            assert sample.reward == 1, "multi_turn_single_sample merges all turns, reward should be 1"
            assert "2008" in sample.response, "Response should contain final answer '2008'"
