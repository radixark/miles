from examples.experimental.eval.parallel_sft.hle_eval import Args, extract_choice, summarize


def test_hle_default_output_limit_is_128k() -> None:
    assert Args().max_tokens == 131072


def test_extract_choice_uses_explicit_final_answer() -> None:
    assert extract_choice("I considered A and B.\nFinal answer: **D**") == "D"
    assert extract_choice("work\n\\boxed{C}") == "C"
    assert extract_choice("A is tempting, but I will not state a final answer") is None


def test_summarize_emits_wandb_ready_numeric_metrics_and_rewards() -> None:
    summary = summarize(
        [
            {"id": "one", "status_code": 200, "predicted_answer": "A", "correct": 1.0, "completion_tokens": 7},
            {"id": "two", "status_code": 500, "error": "failed"},
        ]
    )

    assert summary["metrics"] == {
        "tasks_total": 2,
        "completed": 1,
        "errors": 1,
        "request_success_rate": 0.5,
        "graded": 1,
        "correct": 1.0,
        "accuracy": 1.0,
        "completion_tokens": 7,
    }
    assert summary["rewards"] == [1.0, None]
