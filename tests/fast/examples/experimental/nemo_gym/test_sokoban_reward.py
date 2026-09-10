"""Exercise final-answer isolation, actual HTTP serialization, and failure semantics."""

import json
from copy import deepcopy
from enum import Enum
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import sokoban_reward as sr


class Status(Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"


def sample(response: str = "Reasoning.</think><answer>R</answer>") -> SimpleNamespace:
    return SimpleNamespace(
        response=response,
        status=Status.COMPLETED,
        label="R",
        tokens=[1, 2, 3],
        loss_mask=[1, 1],
        metadata={
            "source_dataset": "sokoban",
            "sokoban_question": "Push the box right onto the goal.",
            "gamestr": "+ + + + +\n+ - - - +\n+ * @ X +\n+ - - - +\n+ + + + +",
            "generator_config": {"seed": 42},
        },
    )


def grade(answer: str = "R", reward: float = 1.0) -> dict[str, Any]:
    return {"task_name": "sokoban", "extracted_answer": answer, "score": reward, "reward": reward}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Plan.</think><answer>UDLR</answer>", "UDLR"),
        ("<think>Plan.</think><answer> U D\nL\tR </answer><|im_end|>\n", "UDLR"),
        ("Plan.</think>The solution is:\n<answer>UDLR</answer>\n", "UDLR"),
        # Condensed from a real completion: an unclosed tag used to swallow the final answer.
        ("Candidate <answer>DULULDRLLLD? Revise the moves.</think><answer>DULULLDRRRLLD</answer>", "DULULLDRRRLLD"),
        # A correct reasoning-only plan must not replace an incorrect final answer.
        ("<answer>R</answer> Solved in reasoning.</think><answer>U</answer>", "U"),
    ],
)
def test_extracts_only_final_moves(text: str, expected: str) -> None:
    assert sr.extract_final_answer(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "<answer>R</answer>",
        "<think>Still reasoning <answer>R</answer>",
        "Reasoning.</think>R",
        "Reasoning.</think>\\boxed{R}",
        "Reasoning.</think><answer>R",
        "Reasoning.</think><answer></answer>",
        "Reasoning.</think><answer> \n\t </answer>",
        "Reasoning.</think><answer>R</answer><answer>U</answer>",
        "Reasoning.</think><answer><answer>R</answer></answer>",
        "Reasoning.</think></answer><answer>R",
        "Reasoning.</think><answer>Go RIGHT</answer>",
        "Reasoning.</think><answer>r</answer>",
        "Reasoning.</think><answer>R,U</answer>",
        "Reasoning.</think><answer>R\u200bU</answer>",
        "Reasoning.</think><answer><b>R</b></answer>",
        "Reasoning.</think><answer format='moves'>R</answer>",
        "Reasoning.</think><answer format='moves'><answer>R</answer>",
        "Reasoning.</think><ANSWER>U</ANSWER><answer>R</answer>",
        "Reasoning.</think><answer>R</answer><answer/>",
        "Reasoning.</think><think><answer>R</answer>",
        "Reasoning.</think>More reasoning.</think><answer>R</answer>",
        "Reasoning.</think><answer>R</answer><|im_end|>extra",
        "Reasoning.</think><answer>R</answer><|im_end|><|im_end|>",
    ],
)
def test_rejects_incomplete_ambiguous_or_non_move_answers(text: str) -> None:
    with pytest.raises(sr.InvalidSokobanAnswer):
        sr.extract_final_answer(text)


@pytest.mark.parametrize("value", [None, False, [], {}])
def test_wrong_response_type_is_a_contract_error(value: Any) -> None:
    with pytest.raises(TypeError):
        sr.extract_final_answer(value)


def test_request_isolates_final_answer_without_mutating_training_sample() -> None:
    item = sample("REASONING_ONLY <answer>U</answer></think><answer>R R</answer>")
    item.label = "REFERENCE_ONLY"
    item.metadata.update(sokoban_extracted_answer="STALE_REASONING", sokoban_reward=0.0)
    before = deepcopy(vars(item))
    body = sr.build_verify_request(item)
    assert body["response"]["output"][0]["content"][0]["text"] == "<answer>RR</answer>"
    assert body["answer"] == "REFERENCE_ONLY"
    assert "REFERENCE_ONLY" not in json.dumps(body["responses_create_params"])
    assert "REASONING_ONLY" not in json.dumps(body)
    assert "STALE_REASONING" not in json.dumps(body)
    assert "sokoban_reward" not in body["metadata"]
    body["metadata"]["generator_config"]["seed"] = 99
    assert vars(item) == before


@pytest.mark.parametrize("status", [Status.TRUNCATED, "truncated"])
def test_truncation_cannot_earn_reward_even_with_a_closed_tag(status: Any) -> None:
    item = sample()
    item.status = status
    with pytest.raises(sr.InvalidSokobanAnswer, match="truncated"):
        sr.build_verify_request(item)


@pytest.mark.parametrize("status", ["pending", "aborted", None, False])
def test_unexpected_sample_state_is_not_a_model_failure(status: Any) -> None:
    item = sample()
    item.status = status
    with pytest.raises(ValueError, match="status"):
        sr.build_verify_request(item)


@pytest.mark.parametrize(
    "result",
    [
        None,
        [],
        False,
        {},
        {**grade(), "task_name": "another_task"},
        {**grade(), "extracted_answer": "U"},
        {**grade(), "extracted_answer": None},
        {**grade(), "score": 0.0},
        {**grade(), "reward": True},
        {**grade(), "reward": False},
        {**grade(), "reward": "1"},
        {**grade(), "reward": None},
        {**grade(), "reward": float("nan")},
        {**grade(), "reward": float("inf")},
        {**grade(), "reward": -1},
        {**grade(), "reward": 10**500},
        {**grade(), "reward": 0.5},
        {**grade(), "score": True},
        {**grade(), "mask_sample": True},
        {**grade(), "mask_sample": "false"},
        {**grade(), "mask_sample": 0},
        {**grade(), "failure_reason": "verifier unavailable"},
        {**grade(), "failure_reason": []},
        {**grade(), "failure_reason": False},
    ],
)
def test_invalid_verifier_reply_fails_loudly(result: Any) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        sr._validate_grade(result, "R")


async def test_actual_http_payload_and_training_fields_are_preserved() -> None:
    item = sample("Unclosed <answer>U. More reasoning.</think><answer>R</answer>")
    before = deepcopy(vars(item))
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        assert str(request.url) == "http://gym/verify"
        return httpx.Response(200, json={**grade(), "mask_sample": False, "failure_reason": None})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await sr._score(client, "http://gym/", item) == 1.0
    assert requests[0]["response"]["output"][0]["content"][0]["text"] == "<answer>R</answer>"
    for key in ("response", "tokens", "loss_mask", "label", "status"):
        assert getattr(item, key) == before[key]
    assert item.metadata["sokoban_grading_status"] == "verified"
    assert item.metadata["sokoban_grader_version"] == "final-answer-v1"


async def test_invalid_answer_replaces_stale_grade_without_network() -> None:
    item = sample("<think>Still thinking <answer>R</answer>")
    item.metadata.update(sokoban_reward=1.0, sokoban_extracted_answer="R")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Invalid model output must not reach the verifier")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await sr._score(client, "http://gym", item) == 0.0
    assert item.metadata["sokoban_reward"] == 0.0
    assert item.metadata["sokoban_extracted_answer"] == ""
    assert item.metadata["sokoban_grading_status"] == "missing_or_ambiguous_reasoning_end"


@pytest.mark.parametrize("failure", ["http", "timeout", "json", "schema", "serialization", "metadata"])
async def test_infrastructure_and_contract_failures_do_not_become_zero(failure: str) -> None:
    item = sample()
    if failure == "serialization":
        item.metadata["not_json"] = {1, 2}
    if failure == "metadata":
        item.metadata["gamestr"] = ""
        item.response = "No final answer either"

    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if failure == "http":
            return httpx.Response(503, text="unavailable")
        if failure == "json":
            return httpx.Response(200, text="not JSON")
        return httpx.Response(200, json={**grade(), "reward": "1"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises((httpx.HTTPError, ValueError, TypeError)):
            await sr._score(client, "http://gym", item)
    assert "sokoban_reward" not in item.metadata


async def test_batched_hook_preserves_order_and_skips_invalid_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    base_client = httpx.AsyncClient
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        text = json.loads(request.content)["response"]["output"][0]["content"][0]["text"]
        requests.append(text)
        moves = text.removeprefix("<answer>").removesuffix("</answer>")
        return httpx.Response(200, json=grade(moves, float(moves == "R")))

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        return base_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(sr.httpx, "AsyncClient", client_factory)
    monkeypatch.setenv("NEMO_GYM_SOKOBAN_URL", "http://gym")
    items = [sample(), sample("Plan.</think><answer>U</answer>"), sample("No final answer")]
    assert await sr.reward_func(None, items) == [1.0, 0.0, 0.0]
    assert requests == ["<answer>R</answer>", "<answer>U</answer>"]
    assert await sr.reward_func(None, sample()) == 1.0


async def test_empty_batch_does_not_require_service_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_GYM_SOKOBAN_URL", raising=False)
    assert await sr.reward_func(None, []) == []
