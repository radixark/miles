# Adapted from https://github.com/eval-protocol/quickstart-gsm8k
import logging
import re

from eval_protocol.models import EvaluateResult, EvaluationRow

logger = logging.getLogger(__name__)


def extract_answer_digits(ground_truth: str) -> str | None:
    if not ground_truth:
        return None

    match = re.search(r"<answer>(.*?)</answer>", ground_truth, flags=re.IGNORECASE | re.DOTALL)
    answer_string = match.group(1) if match else ground_truth
    digits_match = re.search(r"(\d+)", answer_string)
    return digits_match.group(1) if digits_match else None


def gsm8k_reward_row(row: EvaluationRow, **kwargs) -> EvaluationRow:
    # logger.info("I am beginning to execute GSM8k rollout: %s", row.execution_metadata.rollout_id)
    assistant_messages = [message for message in row.messages if message.role == "assistant"]
    last_assistant_content = assistant_messages[-1].content if assistant_messages else ""
    prediction = extract_answer_digits(str(last_assistant_content))
    gt = extract_answer_digits(str(row.ground_truth))

    if prediction is None or gt is None:
        score = 0
        reason = "Missing answer tags in prediction or ground truth."
    elif gt == prediction:
        score = 1
        reason = "Model answer is correct."
    else:
        score = 0
        reason = "Model answer is not correct."

    reason += f" Prediction: {prediction}, Ground Truth: {gt}"

    evaluation_result = EvaluateResult(
        score=score,
        is_score_valid=True,
        reason=reason,
    )
    # logger.info("I am done executing GSM8k rollout: %s", row.execution_metadata.rollout_id)
    row.evaluation_result = evaluation_result
    return row


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-vs"]))
