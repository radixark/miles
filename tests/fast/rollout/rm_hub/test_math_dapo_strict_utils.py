"""CPU-only regression coverage for the opt-in final-integer DAPO grader."""

from types import SimpleNamespace

import pytest

from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.rollout.rm_hub.math_dapo_strict_utils import compute_score, extract_final_integer
from miles.rollout.rm_hub.math_dapo_utils import compute_score as legacy_compute_score
from miles.utils.async_utils import run
from miles.utils.types import RewardSpec, Sample


@pytest.mark.parametrize(
    "response",
    [
        "Answer: 45",
        "  answer \t: 45 \n\n",
        "**Answer: 45**",
        "**Answer:** 45",
        "**Answer**: 45",
        "**Answer:** **45**",
        "__Answer: 45__",
        "__Answer__: __45__",
        "*Answer: 45*",
        "_Answer: 45_",
        "`Answer: 45`",
        "Answer: `45`",
        "Answer: **45**",
        "Answer: ***45***",
        "### **Answer: 45**",
        r"Answer: \(45\)",
        r"Answer: \[45\]",
        "Answer: $45$",
        "Answer: $$45$$",
        r"Answer: \boxed{45}",
        r"**Answer: \(\boxed{45}\)**",
        r"**Answer:** $\boxed{45}$",
        "Answer: 45<｜end▁of▁sentence｜>",
        "**Answer: 45**\n<｜end▁of▁sentence｜>\n",
        "Answer: 45<|im_end|><|endoftext|><|eot_id|></s>",
        "We compute 9 * 5 = 45.\n\nAnswer: 45",
        "<think>Try 44.\nAnswer: 44\nRecheck.</think>\nAnswer: 45",
        "Reasoning was prefilled.\nAnswer: 44\n</think>\n**Answer: 45**",
        "<think>Reasoning.</think>Answer: 45",
        "Let me work.\n<think>Reasoning.</think>\nAnswer: 45",
        "Example:\n```text\nAnswer: 46\n```\nAnswer: 45",
        "Work:\n~~~\nAnswer: 46\n~~~~\nAnswer: 45",
    ],
)
def test_complete_single_integer_formats(response):
    assert extract_final_integer(response) == "45"
    assert compute_score(response, "45") == {"score": 1.0, "acc": True, "pred": "45", "valid": True}


@pytest.mark.parametrize(
    "response",
    [
        "",
        "45",
        r"\boxed{45}",
        "The answer is 45",
        "The Answer: 45",
        "notananswer: 45",
        "AnAnswer: 45",
        "> Answer: 45",
        "Answer:\n45",
        "Answer: 45 because 9 * 5 = 45",
        "Answer: 45\nHere is why.",
        "Answer: 45.",
        "Answer: 45 units",
        "Answer: $45$ or $46$",
        "Answer: $45$ or $45$",
        r"Answer: \boxed{45} or \boxed{46}",
        "Answer: 45, 46",
        "Answer: 45/1",
        "Answer: 9 * 5",
        "Answer: x = 45",
        "Answer: 45.0",
        "Answer: 4.5e1",
        "Answer: 4 5",
        "Answer: ４５",
        "Answer: --45",
        "Answer: 45\nAnswer: 46",
        "Answer: 46\nAnswer: 45",
        "Answer: 45\nAnswer: 45",
        "**Answer:** 46\n**Answer: 45**",
        "Answer: maybe 46\nAnswer: 45",
        "Answer: 45\nActually, still checking...",
        "**Answer: 45",
        "**Answer: 45*",
        "Answer: 45**",
        "Answer: **45*",
        "Answer: ***45**",
        "Answer: $45",
        r"Answer: \(45",
        r"Answer: \(45$",
        r"Answer: \boxed{45",
        r"Answer: \boxed{45}{46}",
        "Answer: \"45\"",
        "<think>Still working.\nAnswer: 45",
        "<think>Answer: 45</think>",
        "<think>Answer: 45</think>\nContinuing...",
        "<think>One <think>Two</think>\nAnswer: 45",
        "Answer: 46\n<think>Recheck.</think>\nAnswer: 45",
        "Answer: 45\n<think>Still working.</think>",
        "</think><think>Still thinking.\nAnswer: 45",
        "</think></think>\nAnswer: 45",
        "```text\nAnswer: 45",
        "```text\nAnswer: 45\n```",
        "~~~\nAnswer: 45",
        "```\n~~~\nAnswer: 45",
        "Answer: 45<｜end▁of▁sentence｜> explanation",
        "Answer: 46<｜end▁of▁sentence｜>\nAnswer: 45",
        "Answer: <|im_end|>45",
        "**Answer: 45<｜end▁of▁sentence｜>**",
    ],
)
def test_rejects_ambiguous_nonfinal_or_malformed_responses(response):
    assert extract_final_integer(response) is None
    assert compute_score(response, "45") == {"score": -1.0, "acc": False, "pred": None, "valid": False}


@pytest.mark.parametrize(
    "response,label,pred",
    [
        ("Answer: -45", "-45", "-45"),
        (r"Answer: $\boxed{-45}$", -45, "-45"),
        ("Answer: +00045", "+045.000", "45"),
        ("Answer: -000", 0, "0"),
        ("Answer: 45", 45.0, "45"),
        ("Answer: 45", "45.000", "45"),
        ("Answer: 9007199254740993", "9007199254740993", "9007199254740993"),
        ("Answer: -9007199254740993", -9007199254740993, "-9007199254740993"),
        ("Answer: 9007199254740993", "9007199254740993.000", "9007199254740993"),
        ("Answer: " + "9" * 5000, "9" * 5000, "9" * 5000),
    ],
)
def test_exact_integer_arithmetic(response, label, pred):
    assert compute_score(response, label) == {"score": 1.0, "acc": True, "pred": pred, "valid": True}


@pytest.mark.parametrize("label", ["46", "-45", "9007199254740993"])
def test_valid_wrong_answer_is_distinct_from_invalid_response(label):
    assert compute_score("Answer: 45", label) == {"score": -1.0, "acc": False, "pred": "45", "valid": True}


def test_no_float_rounding_false_positive():
    result = compute_score("Answer: 9007199254740992", "9007199254740993")
    assert result == {"score": -1.0, "acc": False, "pred": "9007199254740992", "valid": True}


@pytest.mark.parametrize(
    "label",
    ["45.9", 45.9, "-45.1", -45.1, "1e3", "nan", float("nan"), float("inf"), float(2**53), True, None, ""],
)
def test_bad_labels_raise_instead_of_creating_math_rewards(label):
    with pytest.raises(ValueError, match="integer label"):
        compute_score("Answer: 45", label)


def test_full_response_is_checked_not_a_300_character_window():
    assert compute_score("Working.\n" * 1000 + "Answer: 45", "45")["score"] == 1.0
    assert extract_final_integer("Answer: 46\n" + "Working.\n" * 1000 + "Answer: 45") is None
    assert extract_final_integer("<think>\n" + "Working.\n" * 1000 + "Answer: 45") is None
    assert extract_final_integer("x" * 1000 + " Answer: 45") is None


def test_explicit_incomplete_text_never_receives_positive_reward():
    assert compute_score("Answer: 45", "45", is_complete=False) == {
        "score": -1.0,
        "acc": False,
        "pred": None,
        "valid": False,
    }


@pytest.mark.parametrize("status", list(Sample.Status))
@pytest.mark.parametrize("response", ["Answer: 45", "**Answer: 45**<｜end▁of▁sentence｜>"])
def test_default_dispatch_requires_completed_status(status, response):
    args = SimpleNamespace(rm_type="dapo_strict", custom_rm_path=None)
    sample = Sample(response=response, label="45", status=status)
    reward = run(async_rm(args, sample))
    is_complete = status == Sample.Status.COMPLETED
    assert reward == {
        "score": 1.0 if is_complete else -1.0,
        "acc": is_complete,
        "pred": "45" if is_complete else None,
        "valid": is_complete,
    }


@pytest.mark.parametrize("config_source", ["args", "metadata", "reward_spec"])
def test_strict_rm_uses_existing_configuration_precedence(config_source):
    args = SimpleNamespace(rm_type="dapo", custom_rm_path=None)
    sample = Sample(response="**Answer: 45**", label="45", status=Sample.Status.COMPLETED)
    if config_source == "args":
        args.rm_type = "dapo_strict"
    elif config_source == "metadata":
        sample.metadata = {"rm_type": "dapo_strict"}
    else:
        sample.reward_spec = RewardSpec(rm_type="dapo_strict")
    assert run(async_rm(args, sample))["score"] == 1.0


def test_group10_format_regression_and_mixed_correctness():
    # Reproduce the reported formatting split, not a replay of unavailable rollout data.
    responses = ["**Answer: 45**", "Answer: 45", "Answer: $45$", r"Answer: \boxed{45}"]
    args = SimpleNamespace(rm_type="dapo", custom_rm_path=None)
    samples = [Sample(response=text, label="45", status=Sample.Status.COMPLETED) for text in responses]
    assert [reward["score"] for reward in run(batched_async_rm(args, samples))] == [-1.0, 1.0, 1.0, 1.0]
    args.rm_type = "dapo_strict"
    rewards = run(batched_async_rm(args, samples))
    assert [reward["score"] for reward in rewards] == [1.0, 1.0, 1.0, 1.0]
    assert [reward["pred"] for reward in rewards] == ["45"] * 4

    samples[1].response = "Answer: 46"
    samples[2].response = "Answer: $45$ or $46$"
    samples[3].status = Sample.Status.TRUNCATED
    run(batched_async_rm(args, samples, inplace_set_reward_field=True))
    assert [sample.reward["score"] for sample in samples] == [1.0, -1.0, -1.0, -1.0]
    assert [sample.reward["valid"] for sample in samples] == [True, True, False, False]


@pytest.mark.parametrize(
    "response,legacy_score,legacy_pred,strict_score,strict_pred",
    [
        ("**Answer: 45**", -1.0, "45**", 1.0, "45"),
        ("**Answer:** 45", -1.0, "**45", 1.0, "45"),
        (r"Answer: \(45\)", -1.0, r"\(45\)", 1.0, "45"),
        ("Answer: $45$ or $46$", 1.0, "45", -1.0, None),
        ("notananswer: 45", 1.0, "45", -1.0, None),
    ],
)
def test_known_legacy_behavior_is_not_silently_changed(response, legacy_score, legacy_pred, strict_score, strict_pred):
    legacy = legacy_compute_score(response, "45")
    strict = compute_score(response, "45")
    assert (legacy["score"], legacy["pred"]) == (legacy_score, legacy_pred)
    assert (strict["score"], strict["pred"]) == (strict_score, strict_pred)
