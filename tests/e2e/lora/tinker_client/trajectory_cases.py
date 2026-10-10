import pytest
from tinker_cookbook.rl.data_processing import trajectory_to_data

from miles.tinker.core.token_trace import TokenTrace
from miles.tinker.client.trajectory import turns_to_trajectory


@pytest.mark.parametrize("next_prompt,expected_count", [([1, 2, 3, 4, 5], 1), ([1, 2, 30, 5], 2), ([9, 10], 2)],
                         ids=["extends", "retokenized", "compacted"])
def test_only_actual_token_prefixes_merge(next_prompt, expected_count):
    trace = TokenTrace()
    trace.record("first", [1, 2], {"tokens": [3, 4], "logprobs": [-0.3, -0.4], "stop_reason": "stop"})
    trace.record("second", next_prompt, {"tokens": [6, 7], "logprobs": [-0.6, -0.7], "stop_reason": "length"})
    datums = trajectory_to_data(turns_to_trajectory(trace.turns), traj_advantage=2.0)
    assert len(datums) == expected_count
    trained_tokens, logprobs = [], []
    for datum in datums:
        mask = datum.loss_fn_inputs["mask"].data
        targets = datum.loss_fn_inputs["target_tokens"].data
        sampled = datum.loss_fn_inputs["logprobs"].data
        advantages = datum.loss_fn_inputs["advantages"].data
        trained_tokens.extend(token for token, weight in zip(targets, mask, strict=True) if weight)
        logprobs.extend(value for value, weight in zip(sampled, mask, strict=True) if weight)
        assert advantages == [2.0 * weight for weight in mask]
    assert trained_tokens == [3, 4, 6, 7]
    assert logprobs == pytest.approx([-0.3, -0.4, -0.6, -0.7])
    if expected_count == 2:
        assert datums[1].model_input.to_ints() == [*next_prompt, 6]


def test_repeated_prompts_and_continuations_after_length_are_retained():
    trace = TokenTrace()
    for index, stop_reason in enumerate(["length", "stop"]):
        trace.record(str(index), [1, 2], {"tokens": [3 + index], "logprobs": [-0.5], "stop_reason": stop_reason})
    trajectory = turns_to_trajectory(trace.turns)
    assert len(trajectory.transitions) == 2
    assert len(trajectory_to_data(trajectory, traj_advantage=1.0)) == 2
