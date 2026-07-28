from typing import Any

import pytest
from tests.fast.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_rollout

from miles.utils.test_utils.mock_tools import TwoTurnStub
from miles.utils.types import Sample


TWO_TURN_DATA_ROWS = [{"input": [{"role": "user", "content": TwoTurnStub.USER_QUESTION}], "label": "2008"}]

_VARIANT_NAMES = ["multi_turn", "agentic_tool_call"]

_BASE_EXTRA_ARGV = [
    "--rollout-batch-size",
    "2",
    "--n-samples-per-prompt",
    "2",
    "--n-samples-per-eval-prompt",
    "2",
    "--custom-rm-path",
    "tests.fast.rollout.inference_rollout.integration.test_multi_turn._simple_reward_function",
]


def _config_for_variant(variant: str) -> RolloutEnvConfig:
    return RolloutEnvConfig(
        extra_argv=MODULAR_ROLLOUT_BASE_ARGV + extra_argv_for_variant(variant) + _BASE_EXTRA_ARGV,
        data_rows=TWO_TURN_DATA_ROWS,
    )


@pytest.mark.parametrize(
    "variant,rollout_env",
    [pytest.param(variant, _config_for_variant(variant), id=variant) for variant in _VARIANT_NAMES],
    indirect=["rollout_env"],
)
@pytest.mark.parametrize("test_type", ["train", "eval"])
def test_rollout(rollout_env, variant, test_type):
    env = rollout_env
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
    if variant == "agentic_tool_call":
        if len(samples) > 0 and isinstance(samples[0], list):
            # Train mode: list[list[Sample]] — one singleton list (merged TITO sample) per generate
            assert len(samples) == 2, f"n_samples_per_prompt=2, so group should have 2 samples, got {len(samples)}"
            for group_sample in samples:
                assert isinstance(group_sample, list), "agentic_tool_call returns list[Sample] per generate"
                assert len(group_sample) == 1, "linear trajectory merges into exactly one sample"
                _verify_sample(group_sample[0])
        else:
            # Eval mode: list[Sample], flattened (one merged sample per generate)
            assert len(samples) == 2, f"n_samples_per_eval_prompt=2, so should have 2 samples, got {len(samples)}"
            for sample in samples:
                _verify_sample(sample)
    else:
        assert len(samples) == 2, f"n_samples_per_prompt=2, so group should have 2 samples, got {len(samples)}"
        for sample in samples:
            assert isinstance(sample, Sample), "multi_turn returns a scalar Sample per generate"
            _verify_sample(sample)


def _verify_sample(sample: Sample, expected_reward: float = 1.0, expect_answer: bool = True):
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == expected_reward, f"Sample should have reward={expected_reward}"
    if expect_answer:
        assert "2008" in sample.response, "Response should contain final answer '2008'"


async def _simple_reward_function(args, samples: Sample | list[Sample]) -> float | list[float]:
    if isinstance(samples, list):
        return [_check_reward(sample) for sample in samples]
    return _check_reward(samples)


def _check_reward(sample: Sample) -> float:
    return float(sample.response and (str(sample.label) in sample.response))
