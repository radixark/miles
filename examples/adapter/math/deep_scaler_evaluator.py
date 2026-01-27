from eval_protocol.models import EvaluateResult, EvaluationRow

from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward


def deepscaler_reward_row(row: EvaluationRow, **kwargs) -> EvaluationRow:
    assistant_message = row.last_assistant_message()
    solution = str(assistant_message.content)
    ground_truth = str(row.ground_truth)
    score = get_deepscaler_rule_based_reward(solution, ground_truth)

    row.evaluation_result = EvaluateResult(
        score=float(score),
        is_score_valid=True,
        reason=f"Deepscaler reward: {score}.",
    )
    return row


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-vs"]))
