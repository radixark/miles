import pytest
from tests.fixtures.rollout_integration import IntegrationEnvConfig
from tests.rollout.modular_rollout.integration.utils import (
    _MODULAR_ROLLOUT_ARGV_WITHOUT_GENERATE,
    extra_argv_for_variant,
    load_and_call_train,
)

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput
from miles.rollout.modular_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.utils.test_utils.mock_tools import TwoTurnStub
from miles.utils.types import Sample

TWO_TURN_DATA_ROWS = [{"input": TwoTurnStub.USER_QUESTION, "label": "2008"}]


def _config_for_variant(variant: str) -> IntegrationEnvConfig:
    return IntegrationEnvConfig(
        extra_argv=_MODULAR_ROLLOUT_ARGV_WITHOUT_GENERATE + extra_argv_for_variant(variant),
        data_rows=TWO_TURN_DATA_ROWS,
    )


_VARIANTS = [
    pytest.param(_config_for_variant("single_turn"), id="single_turn"),
    pytest.param(_config_for_variant("multi_turn_single_sample"), id="multi_turn_single_sample"),
    pytest.param(_config_for_variant("multi_turn_multi_samples"), id="multi_turn_multi_samples"),
    pytest.param(_config_for_variant("agentic_tool_call_single_sample"), id="agentic_tool_call_single_sample"),
    pytest.param(_config_for_variant("agentic_tool_call_multi_samples"), id="agentic_tool_call_multi_samples"),
]


@pytest.mark.parametrize("rollout_integration_env", _VARIANTS, indirect=True)
def test_train(rollout_integration_env, request):
    env = rollout_integration_env
    variant = request.node.callspec.id

    env.mock_server.process_fn = TwoTurnStub.process_fn

    out = load_and_call_train(env.args, env.data_source)

    assert len(out.samples) == env.args.rollout_batch_size
    group = out.samples[0]

    if variant in ("multi_turn_multi_samples", "agentic_tool_call_multi_samples"):
        assert len(group) == 2
        for sample in group:
            assert sample.status == Sample.Status.COMPLETED
        assert group[-1].reward == 1
        assert "2008" in group[-1].response
    else:
        assert len(group) == env.args.n_samples_per_prompt
        sample = group[0]
        assert sample.status == Sample.Status.COMPLETED
        if variant == "single_turn":
            assert sample.reward == 0
        else:
            assert sample.reward == 1
            assert "2008" in sample.response


@pytest.mark.parametrize("rollout_integration_env", _VARIANTS, indirect=True)
def test_eval(rollout_integration_env, request):
    env = rollout_integration_env
    variant = request.node.callspec.id

    env.mock_server.process_fn = TwoTurnStub.process_fn

    fn = load_rollout_function(
        RolloutFnConstructorInput(args=env.args, data_source=env.data_source), env.args.eval_function_path
    )
    out = call_rollout_function(fn, RolloutFnEvalInput(rollout_id=0))

    assert "toy" in out.data
    rewards = out.data["toy"]["rewards"]
    samples = out.data["toy"]["samples"]

    if variant in ("multi_turn_multi_samples", "agentic_tool_call_multi_samples"):
        assert len(rewards) == len(samples) == 2
        assert rewards[-1] == 1
        assert "2008" in samples[-1].response
    else:
        assert len(rewards) == len(samples) == env.args.n_samples_per_eval_prompt
        if variant == "single_turn":
            assert rewards[0] == 0
        else:
            assert rewards[0] == 1
            assert "2008" in samples[0].response
