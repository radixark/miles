import asyncio
import json
from pathlib import Path

from aiohttp import web
from pytest import MonkeyPatch

from examples.experimental.eval.parallel_sft import hle_eval
from examples.experimental.eval.parallel_sft.hle_eval import (
    Args,
    RequestStartRateLimiter,
    extract_choice,
    extract_final_answer,
    generation_prompt,
    judge_payload,
    main_async,
    parse_judgment,
    summarize,
)


def test_hle_default_output_limit_is_128k() -> None:
    assert Args().max_tokens == 131072
    assert Args().judge_base_url is None
    assert Args().judge_api_key_env == "HLE_JUDGE_API_KEY"
    assert Args().judge_max_qps == 0.0


def test_extract_choice_uses_explicit_final_answer() -> None:
    assert extract_choice("I considered A and B.\nFinal answer: **D**") == "D"
    assert extract_choice("work\nFinal answer: \\boxed{C}") == "C"
    assert extract_choice("work\n\\boxed{C}") is None
    assert extract_choice("Final answer: C\npostscript") is None
    assert extract_choice("A is tempting, but I will not state a final answer") is None


def test_generation_prompt_requests_only_a_terminal_final_answer() -> None:
    row = {"question": "What is 2 + 2?", "answer_type": "exactMatch"}

    prompt = generation_prompt(row)

    assert "Final answer: ANSWER" in prompt
    assert "Confidence" not in prompt
    assert "confidence" not in prompt
    assert extract_final_answer("reasoning\nFinal answer: 4") == "4"


def test_summarize_emits_wandb_ready_numeric_metrics_and_rewards() -> None:
    summary = summarize(
        [
            {"id": "one", "status_code": 200, "predicted_answer": "A", "correct": 1.0, "completion_tokens": 7},
            {"id": "two", "status_code": 500, "error": "failed"},
        ]
    )

    assert summary["metrics"]["tasks_total"] == 2
    assert summary["metrics"]["problems_total"] == 2
    assert summary["metrics"]["completed"] == 1
    assert summary["metrics"]["errors"] == 1
    assert summary["metrics"]["request_success_rate"] == 0.5
    assert summary["metrics"]["graded"] == 1
    assert summary["metrics"]["correct"] == 1.0
    assert summary["metrics"]["accuracy"] == 1.0
    assert summary["metrics"]["completion_tokens"] == 7
    assert summary["rewards"] == [1.0, None]


def test_judge_payload_targets_external_sglang_model_with_json_schema() -> None:
    args = Args()
    args.judge_model = "hle-grader"
    args.judge_max_tokens = 2048
    row = {
        "id": "problem",
        "question": "What is 2 + 2?",
        "answer": "4",
        "answer_type": "exactMatch",
    }

    payload = judge_payload(args, row, "4")

    assert payload["model"] == "hle-grader"
    assert payload["max_tokens"] == 2048
    assert payload["response_format"]["type"] == "json_schema"
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["correct"]["enum"] == ["yes", "no"]
    assert set(schema["required"]) == {"reasoning", "correct"}
    assert "Candidate final answer:\n4" in payload["messages"][0]["content"]
    assert "Reference answer:\n4" in payload["messages"][0]["content"]
    assert row["question"] not in payload["messages"][0]["content"]


def test_parse_judgment_accepts_fenced_json_and_validates_fields() -> None:
    judgment = {
        "reasoning": "The answers match.",
        "correct": "yes",
    }

    assert parse_judgment(f"```json\n{json.dumps(judgment)}\n```") == judgment


def test_summarize_preserves_four_trials_and_external_judge_metrics() -> None:
    results = [
        {
            "id": "problem",
            "trial_index": trial_index,
            "status_code": 200,
            "completion_tokens": 10,
            "judge_requested": True,
            "judge_status_code": 200,
            "judge_completion_tokens": 5,
            "judgment": {"correct": "yes" if trial_index == 0 else "no"},
            "correct": float(trial_index == 0),
        }
        for trial_index in range(4)
    ]

    summary = summarize(results)

    assert summary["metrics"]["tasks_total"] == 4
    assert summary["metrics"]["problems_total"] == 1
    assert summary["metrics"]["accuracy"] == 0.25
    assert summary["metrics"]["judge_requested"] == 4
    assert summary["metrics"]["judge_completed"] == 4
    assert summary["metrics"]["judge_errors"] == 0
    assert summary["metrics"]["judge_completion_tokens"] == 20
    assert summary["metrics"]["problem_any_correct_rate"] == 1.0
    assert len(summary["per_task"]) == 4


def test_judge_rate_limiter_spaces_request_starts(monkeypatch: MonkeyPatch) -> None:
    async def run_test() -> None:
        now = 0.0
        sleeps: list[float] = []

        def monotonic() -> float:
            return now

        async def sleep(delay: float) -> None:
            nonlocal now
            sleeps.append(delay)
            now += delay

        monkeypatch.setattr(hle_eval.time, "monotonic", monotonic)
        monkeypatch.setattr(hle_eval.asyncio, "sleep", sleep)
        limiter = RequestStartRateLimiter(max_qps=2.0)

        await limiter.wait()
        await limiter.wait()
        await limiter.wait()

        assert sleeps == [0.5, 0.5]

    asyncio.run(run_test())


def test_external_sglang_judge_endpoint_end_to_end(tmp_path: Path) -> None:
    async def run_test() -> None:
        requests: list[dict] = []

        async def chat_completions(request: web.Request) -> web.Response:
            payload = await request.json()
            requests.append(payload)
            if payload["model"] == "checkpoint-model":
                content = "A long private reasoning trace.\nFinal answer: 4"
                completion_tokens = 6
            else:
                content = json.dumps(
                    {
                        "reasoning": "The answers match.",
                        "correct": "yes",
                    }
                )
                completion_tokens = 9
            return web.json_response(
                {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"completion_tokens": completion_tokens},
                }
            )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat_completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]

        input_path = tmp_path / "hle.jsonl"
        output_path = tmp_path / "results.jsonl"
        summary_path = tmp_path / "summary.json"
        input_path.write_text(
            json.dumps(
                {
                    "id": "problem",
                    "question": "What is 2 + 2?",
                    "answer": "4",
                    "answer_type": "exactMatch",
                }
            )
            + "\n"
        )
        args = Args()
        args.input = str(input_path)
        args.base_url = f"http://127.0.0.1:{port}/v1"
        args.model = "checkpoint-model"
        args.output_jsonl = str(output_path)
        args.summary_json = str(summary_path)
        args.n_trials = 2
        args.judge_base_url = f"http://127.0.0.1:{port}/v1"
        args.judge_model = "grader-model"
        args.judge_max_retries = 1

        try:
            await main_async(args)
        finally:
            await runner.cleanup()

        summary = json.loads(summary_path.read_text())
        assert summary["metrics"]["tasks_total"] == 2
        assert summary["metrics"]["graded"] == 2
        assert summary["metrics"]["accuracy"] == 1.0
        assert summary["metrics"]["judge_completed"] == 2
        assert summary["metrics"]["judge_completion_tokens"] == 18
        assert len(output_path.read_text().splitlines()) == 2
        judge_requests = [request for request in requests if request["model"] == "grader-model"]
        assert len(judge_requests) == 2
        assert judge_requests[0]["response_format"]["type"] == "json_schema"
        judge_prompt = judge_requests[0]["messages"][0]["content"]
        assert "Candidate final answer:\n4" in judge_prompt
        assert "long private reasoning trace" not in judge_prompt.lower()
        assert "confidence" not in judge_prompt.lower()

    asyncio.run(run_test())
