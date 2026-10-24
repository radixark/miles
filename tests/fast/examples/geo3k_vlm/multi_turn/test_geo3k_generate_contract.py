from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from examples.geo3k_vlm.multi_turn import rollout

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample


@pytest.mark.parametrize("budget", [0, 1])
async def test_generation_uses_caller_state_and_closes_environment_on_every_return(
    monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    """Both exhausted and generated responses retain the caller's sample and close its environment."""
    env = SimpleNamespace(reset=Mock(), close=Mock())
    tokenizer = SimpleNamespace(decode=Mock(return_value="answer"))
    state = SimpleNamespace(args=SimpleNamespace(partial_rollout=False), tokenizer=tokenizer)
    sample = Sample(prompt="question", tokens=[10], loss_mask=[], rollout_log_probs=[])
    sampling = {"max_new_tokens": 5}
    prepare = Mock(return_value=(None, [], budget, []))
    infer = AsyncMock(return_value=("answer", [20], [-0.2], "length"))
    monkeypatch.setattr(rollout, "_initialize_resources", lambda args, sample: (env, None, {"max_turns": 1}, "url"))
    monkeypatch.setattr(rollout, "_prepare_start_state", prepare)
    monkeypatch.setattr(rollout, "_run_inference_step", infer)

    output = await rollout.generate(
        GenerateFnInput(state=state, sample=sample, sampling_params=sampling, evaluation=False)
    )

    assert isinstance(output, GenerateFnOutput)
    assert output.samples is sample
    assert prepare.call_args.args[1] is state
    assert prepare.call_args.args[3] is not sampling
    assert sampling == {"max_new_tokens": 5}
    env.reset.assert_called_once_with()
    env.close.assert_called_once_with()
    assert sample.status == Sample.Status.TRUNCATED
    if budget:
        assert infer.call_args.args[-1] is tokenizer
        assert sample.tokens == [10, 20]
        assert sample.loss_mask == [1]
        assert sample.rollout_log_probs == [-0.2]
        assert sample.response == "answer"
        assert sample.response_length == 1
    else:
        infer.assert_not_called()
        assert sample.tokens == [10]
