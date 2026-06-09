import pytest
from tests.fast.fixtures.rollout_fixtures import DEFAULT_DATA_ROWS, RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_train

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.misc import function_registry
from miles.utils.types import Sample


async def _multi_sample_generate(input: GenerateFnInput) -> GenerateFnOutput:
    sample = input.sample
    s1 = Sample(
        prompt=sample.prompt,
        response="\\boxed{8}",
        response_length=5,
        tokens=sample.tokens + [59, 79075, 90, 23, 92],
        label=sample.label,
        reward=None,
        status=Sample.Status.COMPLETED,
    )
    s2 = Sample(
        prompt=sample.prompt,
        response="\\boxed{8}",
        response_length=5,
        tokens=sample.tokens + [59, 79075, 90, 23, 92],
        label=sample.label,
        reward=0.5,
        status=Sample.Status.COMPLETED,
    )
    return GenerateFnOutput(samples=[s1, s2])


@pytest.mark.parametrize(
    "rollout_env",
    [
        pytest.param(
            RolloutEnvConfig(
                extra_argv=MODULAR_ROLLOUT_BASE_ARGV
                + [
                    "--custom-generate-function-path",
                    "test:multi_sample_generate",
                    "--rollout-batch-size",
                    "1",
                    "--n-samples-per-prompt",
                    "1",
                ],
                data_rows=DEFAULT_DATA_ROWS,
            ),
            id="multi_sample_output",
        ),
    ],
    indirect=True,
)
def test_multi_sample_output_preserves_existing_reward(rollout_env):
    env = rollout_env
    with function_registry.temporary("test:multi_sample_generate", _multi_sample_generate):
        out = load_and_call_train(env.args, env.data_source)

        assert len(out.samples) == env.args.rollout_batch_size
        group = out.samples[0]
        assert isinstance(group[0], list)
        samples = group[0]
        assert len(samples) == 2
        assert samples[0].reward == 1
        assert samples[1].reward == 0.5
