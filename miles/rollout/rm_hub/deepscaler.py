from miles.rollout.rm_hub.math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy, has_boxed
from miles.utils.reasoning import REASONING_DELIMITERS, extract_after_reasoning


def _grade_boxed_solution(model_solution, label):
    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0
    if label == "":
        return 0

    # Convert single answer to list for uniform processing
    assert isinstance(label, (str, float, int))
    ground_truths = [label]

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if has_boxed(truth):
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return 0

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return 1

    return 0


def get_deepscaler_rule_based_reward(response, label):
    solution = extract_after_reasoning(response)
    if solution is None:
        return 0
    return _grade_boxed_solution(solution, label)


def get_gemma_math_reward(response, label):
    # Gemma-4 closes thinking with <channel|>; grade text after the last one.
    _, delimiter = REASONING_DELIMITERS["gemma4"]
    if delimiter in response:
        response = response.rsplit(delimiter, 1)[-1]
    return _grade_boxed_solution(response, label)
