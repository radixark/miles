import pytest
from tinker_agent.trajectory import rows_to_trajectory
from tinker_cookbook.rl.data_processing import trajectory_to_data


def row(prompt, tokens, logprobs, stop_reason):
    return {
        "prompt_tokens": list(prompt),
        "sampled_tokens": list(tokens),
        "logprobs": list(logprobs),
        "metadata": {"status": "ok", "num_samples": 1, "stop_reason": stop_reason},
    }


@pytest.mark.parametrize(
    "next_prompt,expected_count",
    [([1, 2, 3, 4, 5], 1), ([1, 2, 30, 5], 2), ([9, 10], 2)],
    ids=["extends", "retokenized", "compacted"],
)
def test_only_actual_token_prefixes_merge(next_prompt, expected_count):
    turns = [
        row((1, 2), (3, 4), (-0.3, -0.4), "stop"),
        row(tuple(next_prompt), (6, 7), (-0.6, -0.7), "length"),
    ]
    datums = trajectory_to_data(rows_to_trajectory(turns, "completed"), traj_advantage=2.0)
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


def test_a_limit_before_the_first_sample_is_the_cookbooks_empty_terminal_transition():
    trajectory = rows_to_trajectory([], "context_overflow")
    assert trajectory.stop_reason == "context_overflow"
    assert [transition.ac.tokens for transition in trajectory.transitions] == [[]]
    assert trajectory_to_data(trajectory, traj_advantage=1.0) == []


def test_repeated_prompts_and_continuations_after_length_are_retained():
    turns = []
    for index, stop_reason in enumerate(["length", "stop"]):
        turns.append(row((1, 2), (3 + index,), (-0.5,), stop_reason))
    trajectory = rows_to_trajectory(turns, "completed")
    assert [transition.ac.stop_reason for transition in trajectory.transitions] == ["length", "stop"]
    assert trajectory.stop_reason == "completed"
    assert len(trajectory.transitions) == 2
    assert len(trajectory_to_data(trajectory, traj_advantage=1.0)) == 2
