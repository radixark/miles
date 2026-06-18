import pytest
from tests.fast.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_rollout

from miles.utils.test_utils.mock_tools import TwoTurnStub
from miles.utils.types import Sample


TWO_TURN_DATA_ROWS = [{"input": [{"role": "user", "content": TwoTurnStub.USER_QUESTION}], "label": "2008"}]

_AGENTIC_VARIANTS = ["agentic_tool_call_single_sample", "agentic_tool_call_multi_samples"]

_METADATA_RM_EXTRA_ARGV = [
    "--rollout-batch-size",
    "2",
    "--n-samples-per-prompt",
    "2",
    "--n-samples-per-eval-prompt",
    "2",
    "--custom-rm-path",
    "tests.fast.rollout.inference_rollout.integration.test_agent_metadata._metadata_reward_function",
]


def _metadata_config_for_variant(variant: str) -> RolloutEnvConfig:
    return RolloutEnvConfig(
        extra_argv=MODULAR_ROLLOUT_BASE_ARGV + extra_argv_for_variant(variant) + _METADATA_RM_EXTRA_ARGV,
        data_rows=TWO_TURN_DATA_ROWS,
    )


@pytest.mark.parametrize(
    "variant,rollout_env",
    [pytest.param(v, _metadata_config_for_variant(v), id=v) for v in _AGENTIC_VARIANTS],
    indirect=["rollout_env"],
)
def test_agent_metadata_reaches_reward_model(rollout_env, variant):
    """Verify that agent metadata is available to --custom-rm-path during reward computation."""
    from miles.utils.test_utils import mock_tools

    mock_tools.AGENTIC_RETURN_METADATA = {"agent_reward": 42.0}
    try:
        env = rollout_env
        env.mock_server.process_fn = TwoTurnStub.process_fn

        out = load_and_call_rollout(env.args, env.data_source, mode="train")

        assert len(out.samples) == env.args.rollout_batch_size
        group = out.samples[0]
        flat = [s for sub in group for s in (sub if isinstance(sub, list) else [sub])]
        for sample in flat:
            assert sample.reward == 42.0, f"RM should have read reward from metadata, got {sample.reward}"
    finally:
        mock_tools.AGENTIC_RETURN_METADATA = None


async def _metadata_reward_function(args, samples: Sample | list[Sample]) -> float | list[float]:
    """Custom RM that reads reward from sample.metadata['agent_reward'] — simulates SWE-bench pattern."""
    if isinstance(samples, list):
        return [_extract_metadata_reward(s) for s in samples]
    return _extract_metadata_reward(samples)


def _extract_metadata_reward(sample: Sample) -> float:
    return float(sample.metadata.get("agent_reward", 0.0))
