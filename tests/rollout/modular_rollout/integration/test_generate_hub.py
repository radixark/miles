import pytest
from tests.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fixtures.rollout_integration import IntegrationEnvConfig
from tests.rollout.modular_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_rollout

from miles.utils.test_utils.mock_tools import TwoTurnStub
from miles.utils.types import Sample

TWO_TURN_DATA_ROWS = [{"input": TwoTurnStub.USER_QUESTION, "label": "2008"}]

_VARIANT_NAMES = [
    "single_turn",
    "multi_turn_single_sample",
    "multi_turn_multi_samples",
    "agentic_tool_call_single_sample",
    "agentic_tool_call_multi_samples",
]


def _config_for_variant(variant: str) -> IntegrationEnvConfig:
    return IntegrationEnvConfig(
        extra_argv=MODULAR_ROLLOUT_BASE_ARGV + extra_argv_for_variant(variant),
        data_rows=TWO_TURN_DATA_ROWS,
    )


def _verify_samples(variant: str, samples: list[Sample], expected_count: int):
    if variant in ("multi_turn_multi_samples", "agentic_tool_call_multi_samples"):
        assert len(samples) == 2
        for sample in samples:
            assert sample.status == Sample.Status.COMPLETED
        assert samples[-1].reward == 1
        assert "2008" in samples[-1].response
    else:
        assert len(samples) == expected_count
        sample = samples[0]
        assert sample.status == Sample.Status.COMPLETED
        if variant == "single_turn":
            assert sample.reward == 0
        else:
            assert sample.reward == 1
            assert "2008" in sample.response


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
        _verify_samples(variant, group, env.args.n_samples_per_prompt)
    else:
        assert "toy" in out.data
        rewards = out.data["toy"]["rewards"]
        samples = out.data["toy"]["samples"]
        _verify_samples(variant, samples, env.args.n_samples_per_eval_prompt)
