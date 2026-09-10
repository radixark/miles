from .math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy


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
        if "\\boxed" in truth:
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


# markers that end the reasoning segment; Kimi K-series uses the tagged <|open|>response<|sep|>
_THINKING_CLOSERS = ("<|open|>response<|sep|>", "</think>", "###Response")


def get_deepscaler_rule_based_reward(response, label):
    for closer in _THINKING_CLOSERS:
        if closer in response:
            return _grade_boxed_solution(response.split(closer)[-1], label)
    return 0


def get_gemma_math_reward(response, label):
    # Gemma-4 closes thinking with <channel|>; grade text after the last one.
    if "<channel|>" in response:
        response = response.split("<channel|>")[-1]
    return _grade_boxed_solution(response, label)
