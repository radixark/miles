import asyncio
import json
import random
import re
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web
from examples.experimental.eval.parallel_sft import hle_eval
from examples.experimental.eval.parallel_sft.hle_eval import (
    DEFAULT_FILLER_TRAILER,
    FILLER_WORDS,
    Args,
    RequestStartRateLimiter,
    extract_choice,
    extract_final_answer,
    generation_prompt,
    judge_payload,
    main_async,
    parse_judgment,
    prepare_context_budgets,
    render_filler,
    summarize,
)
from pytest import MonkeyPatch, raises


def test_hle_default_output_limit_is_128k() -> None:
    assert Args().max_tokens == 131072
    assert Args().judge_base_url is None
    assert Args().judge_api_key_env == "HLE_JUDGE_API_KEY"
    assert Args().judge_max_qps == 0.0
    assert Args().disable_thinking is False


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
        args.disable_thinking = True
        args.judge_base_url = f"http://127.0.0.1:{port}/v1"
        args.judge_model = "grader-model"
        args.judge_max_retries = 1
        args.incremental = True
        args.generations_jsonl = str(tmp_path / "generations.jsonl")

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
        assert len(Path(args.generations_jsonl).read_text().splitlines()) == 2
        progress = json.loads(summary_path.with_suffix(".progress.json").read_text())
        assert progress["finished_trials"] == progress["planned_trials"] == 2
        checkpoint_requests = [request for request in requests if request["model"] == "checkpoint-model"]
        assert len(checkpoint_requests) == 2
        assert checkpoint_requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
        judge_requests = [request for request in requests if request["model"] == "grader-model"]
        assert len(judge_requests) == 2
        assert judge_requests[0]["response_format"]["type"] == "json_schema"
        judge_prompt = judge_requests[0]["messages"][0]["content"]
        assert "Candidate final answer:\n4" in judge_prompt
        assert "long private reasoning trace" not in judge_prompt.lower()
        assert "confidence" not in judge_prompt.lower()

    asyncio.run(run_test())


def test_luna_grader_uses_reasoning_model_parameters() -> None:
    args = Args()
    args.judge_model = "gpt-5.6-luna"
    args.judge_max_tokens_param = "max_completion_tokens"
    args.judge_reasoning_effort = "medium"
    args.judge_omit_temperature = True
    payload = judge_payload(args, {"answer": "4"}, "4")
    assert payload["max_completion_tokens"] == 16384
    assert "max_tokens" not in payload
    assert "temperature" not in payload
    assert payload["reasoning_effort"] == "medium"
    assert payload["response_format"]["json_schema"]["strict"] is True


def test_context_budget_reserves_templated_prompt(monkeypatch: MonkeyPatch) -> None:
    calls = []

    def template(messages: list[dict], **kwargs: object) -> list[int]:
        calls.append((messages, kwargs))
        return [0] * 100

    monkeypatch.setattr(
        hle_eval.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(apply_chat_template=template),
    )
    args = Args()
    args.max_context_length = 81920
    args.tokenizer_path = "/local/checkpoint-tokenizer"
    row = {"id": "one", "question": "Question", "answer_type": "exactMatch"}
    prepared = prepare_context_budgets(args, [row])[0]
    assert prepared["_prompt_tokens"] == 100
    assert prepared["_max_tokens"] == 81820
    assert "_max_tokens" not in row
    assert calls[0][1]["add_generation_prompt"] is True
    assert calls[0][1]["enable_thinking"] is True
    args.max_context_length = 100
    with raises(ValueError, match="no output budget"):
        prepare_context_budgets(args, [row])


def test_incremental_output_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    output.write_text("preserved\n")
    args = Args()
    args.incremental = True
    args.output_jsonl = str(output)
    with raises(FileExistsError):
        asyncio.run(main_async(args))
    assert output.read_text() == "preserved\n"


class WordTokenizer:
    """Fake fast tokenizer: every whitespace-delimited word is one token, plus 5 template tokens."""

    def apply_chat_template(self, messages: list[dict], **kwargs: object) -> list[int]:
        return [0] * (5 + len(messages[0]["content"].split()))

    def __call__(self, text: str, **kwargs: object) -> dict:
        spans = [match.span() for match in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(spans))), "offset_mapping": spans}


def filler_args(target: int, seed: str = "hle-filler-v1") -> Args:
    args = Args()
    args.max_context_length = 1000
    args.max_tokens = 600
    args.tokenizer_path = "/local/checkpoint-tokenizer"
    args.filler_target_prompt_tokens = target
    args.filler_seed = seed
    return args


def test_filler_pads_prompt_to_target_with_question_first(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(hle_eval.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: WordTokenizer())
    row = {"id": "one", "question": "What is 2 + 2?", "answer_type": "exactMatch"}
    base_tokens = 5 + len(generation_prompt(row).split())

    prepared = prepare_context_budgets(filler_args(400), [row])[0]

    assert prepared["_prompt_tokens"] == 400
    assert prepared["_filler_tokens"] == 400 - base_tokens
    assert prepared["_max_tokens"] == 600
    assert prepared["_message"].startswith(generation_prompt(row) + "\n\n")
    assert prepared["_message"].endswith("\n\n" + DEFAULT_FILLER_TRAILER)
    assert "_message" not in row


def test_filler_is_deterministic_per_seed_and_question(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(hle_eval.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: WordTokenizer())
    row = {"id": "one", "question": "Question", "answer_type": "exactMatch"}
    other = {**row, "id": "two"}

    first = prepare_context_budgets(filler_args(300), [row])[0]["_message"]
    again = prepare_context_budgets(filler_args(300), [row])[0]["_message"]
    reseeded = prepare_context_budgets(filler_args(300, seed="other-seed"), [row])[0]["_message"]
    other_question = prepare_context_budgets(filler_args(300), [other])[0]["_message"]

    assert first == again
    assert first != reseeded
    assert first != other_question


def test_filler_words_are_harmless_dictionary_words() -> None:
    forbidden = {"answer", "final", "question", "yes", "no", "true", "false", "correct", "wrong", "letter", "option"}
    assert all(re.fullmatch(r"[a-z]{3,}", word) for word in FILLER_WORDS)
    assert not forbidden & set(FILLER_WORDS)
    assert len(set(FILLER_WORDS)) == len(FILLER_WORDS)

    text = render_filler(random.Random("seed"), 120)

    assert len(text.split()) == 120
    assert text.count(".") >= 8
    assert "\n\n" in text
    assert set(word.strip(".").lower() for word in text.split()) <= set(FILLER_WORDS)


def test_filler_requires_a_context_budget_below_the_limit(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(hle_eval.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: WordTokenizer())
    row = {"id": "one", "question": "Question", "answer_type": "exactMatch"}
    with raises(ValueError, match="below --max_context_length"):
        prepare_context_budgets(filler_args(1000), [row])
    args = Args()
    args.filler_target_prompt_tokens = 100
    with raises(ValueError, match="requires --max_context_length"):
        asyncio.run(main_async(args))


def test_filler_message_is_sent_and_budget_checked_end_to_end(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(hle_eval.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: WordTokenizer())

    async def run_test() -> None:
        requests: list[dict] = []

        async def chat_completions(request: web.Request) -> web.Response:
            payload = await request.json()
            requests.append(payload)
            prompt_tokens = 5 + len(payload["messages"][0]["content"].split())
            return web.json_response(
                {
                    "choices": [{"message": {"content": "thinking\nFinal answer: 4"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 3},
                }
            )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat_completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        input_path = tmp_path / "hle.jsonl"
        input_path.write_text(
            json.dumps({"id": "p", "question": "What is 2 + 2?", "answer": "4", "answer_type": "exactMatch"}) + "\n"
        )
        args = filler_args(400)
        args.input = str(input_path)
        args.base_url = f"http://127.0.0.1:{port}/v1"
        args.model = "checkpoint-model"
        args.output_jsonl = str(tmp_path / "results.jsonl")
        args.summary_json = str(tmp_path / "summary.json")
        try:
            await main_async(args)
        finally:
            await runner.cleanup()

        sent = requests[0]["messages"][0]["content"]
        assert sent.startswith("What is 2 + 2?")
        assert sent.endswith(DEFAULT_FILLER_TRAILER)
        assert 5 + len(sent.split()) == 400
        assert requests[0]["max_tokens"] == 600
        result = json.loads((tmp_path / "results.jsonl").read_text().splitlines()[0])
        assert result["status_code"] == 200
        assert result["prompt_tokens"] == result["expected_prompt_tokens"] == 400
        assert result["filler_tokens"] > 300
        assert len(result["message_sha256"]) == 64
        assert result["predicted_answer"] == "4"

    asyncio.run(run_test())


def test_resume_reruns_only_missing_trials_and_appends(tmp_path: Path) -> None:
    async def run_test() -> None:
        requests: list[dict] = []

        async def chat_completions(request: web.Request) -> web.Response:
            payload = await request.json()
            requests.append(payload)
            if payload["model"] == "checkpoint-model":
                content = "work\nFinal answer: 4"
            else:
                content = json.dumps({"reasoning": "match", "correct": "yes"})
            return web.json_response(
                {
                    "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 3},
                }
            )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat_completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        input_path = tmp_path / "hle.jsonl"
        input_path.write_text(
            json.dumps({"id": "p", "question": "What is 2 + 2?", "answer": "4", "answer_type": "exactMatch"}) + "\n"
        )
        output = tmp_path / "results.jsonl"
        kept = {"id": "p", "trial_index": 0, "status_code": 200, "correct": 1.0, "completion_tokens": 5}
        failed = {"id": "p", "trial_index": 1, "status_code": 0, "error": "timeout"}
        output.write_text(json.dumps(kept) + "\n" + json.dumps(failed) + "\n")
        generations = tmp_path / "generations.jsonl"
        generations.write_text(json.dumps(kept) + "\n")

        args = Args()
        args.input = str(input_path)
        args.base_url = f"http://127.0.0.1:{port}/v1"
        args.model = "checkpoint-model"
        args.output_jsonl = str(output)
        args.summary_json = str(tmp_path / "summary.json")
        args.generations_jsonl = str(generations)
        args.n_trials = 3
        args.incremental = True
        args.resume = True
        args.disable_thinking = True
        args.judge_base_url = f"http://127.0.0.1:{port}/v1"
        args.judge_model = "grader-model"
        args.judge_max_retries = 1
        try:
            await main_async(args)
        finally:
            await runner.cleanup()

        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert sorted(row["trial_index"] for row in rows) == [
            0,
            1,
            1,
            2,
        ]
        checkpoint_requests = [request for request in requests if request["model"] == "checkpoint-model"]
        assert len(checkpoint_requests) == 2  # trial 0 kept; failed trial 1 and never-run trial 2 regenerated
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["metrics"]["tasks_total"] == 3
        assert summary["metrics"]["completed"] == 3
        progress = json.loads((tmp_path / "summary.progress.json").read_text())
        assert progress["planned_trials"] == progress["finished_trials"] == 3
        assert len(generations.read_text().splitlines()) == 3

    asyncio.run(run_test())


def test_resume_requires_incremental() -> None:
    args = Args()
    args.resume = True
    with raises(ValueError, match="requires --incremental"):
        asyncio.run(main_async(args))
