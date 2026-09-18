# See tests/e2e/short/test_multi_policy_solver_verifier_gsm8k.py for an end-to-end run of this example.
import argparse
import dataclasses
import re
from enum import Enum
from typing import Any

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.single_turn import generate as single_turn_generate
from miles.utils.iter_utils import group_by
from miles.utils.types import Sample

_AGREE_MARKER = "AGREE"
_WRONG_MARKER = "WRONG"
_VERDICT_PREFIX = "VERDICT:"

_VERIFIER_PROMPT_TEMPLATE = (
    "Another model was asked to solve a math problem, and you must check its work.\n\n"
    "Question:\n{question}\n\n"
    "Proposed solution:\n{solver_response}\n\n"
    f"Reason about the proposed solution, then write exactly one verdict line of its own: "
    f"'{_VERDICT_PREFIX} {_AGREE_MARKER}' if the proposed final answer is correct, or "
    f"'{_VERDICT_PREFIX} {_WRONG_MARKER}' if it is not. After a {_WRONG_MARKER} verdict, "
    "write your own final answer on a last line as '#### <answer>'."
)

_VERDICT_PATTERN = re.compile(rf"^{_VERDICT_PREFIX}\s*({_AGREE_MARKER}|{_WRONG_MARKER})\s*$", re.MULTILINE)
_MARKED_ANSWER_PATTERN = re.compile(r"####\s*([^\n]+)")
_NUMBER_PATTERN = re.compile(r"-?\d+(?:[\d,]*\d)?(?:\.\d+)?")


class _Verdict(Enum):
    AGREE = _AGREE_MARKER
    WRONG = _WRONG_MARKER


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    model_ids = resolve_megatron_config(args).model_ids
    assert len(model_ids) == 2, (
        f"examples/multi_policy/solver_verifier.py pairs one solver policy with one verifier policy, but "
        f"--megatron-config names {model_ids}"
    )
    solver_model_id, verifier_model_id = model_ids

    solver_output = await single_turn_generate(input, url=_compute_router_url(args, model_id=solver_model_id))
    solver_sample = solver_output.samples
    assert isinstance(solver_sample, Sample), f"{solver_sample=}"
    assert solver_sample.status != Sample.Status.ABORTED

    verifier_sample = _build_verifier_sample(solver_sample)
    verifier_output = await single_turn_generate(
        dataclasses.replace(input, sample=verifier_sample),
        url=_compute_router_url(args, model_id=verifier_model_id),
    )
    verifier_sample = verifier_output.samples
    assert isinstance(verifier_sample, Sample), f"{verifier_sample=}"

    ground_truth = _extract_answer(solver_sample.label or "")
    solver_correct = _is_correct(solver_sample.response, ground_truth=ground_truth)
    solver_sample.reward = 1.0 if solver_correct else 0.0
    verifier_sample.reward = _compute_verifier_reward(
        solver_correct=solver_correct,
        verdict=_parse_verdict(verifier_sample.response),
        verifier_correct=_is_correct(verifier_sample.response, ground_truth=ground_truth),
    )

    solver_sample.trainer_model_id = solver_model_id
    verifier_sample.trainer_model_id = verifier_model_id
    return GenerateFnOutput(samples=[solver_sample, verifier_sample])


def split_eval_data_by_policy(
    rollout_id: int, args: argparse.Namespace, data: dict[str, dict[str, Any]], extra_metrics: dict[str, Any] | None
) -> bool:
    reward_key = args.eval_reward_key or args.reward_key
    for name in list(data):
        entry = data.pop(name)
        for model_id, samples in group_by(entry["samples"], lambda sample: sample.trainer_model_id).items():
            assert model_id is not None, (
                f"an eval sample of dataset {name!r} carries no trainer_model_id, so its reward cannot be "
                f"attributed to a policy; the generate function must stamp every sample it returns"
            )
            data[f"{name}/{model_id}"] = dict(
                rewards=[sample.reward[reward_key] if sample.reward is not None else None for sample in samples],
                truncated=[sample.status == Sample.Status.TRUNCATED for sample in samples],
                samples=samples,
            )
    return False


def _compute_verifier_reward(*, solver_correct: bool, verdict: _Verdict | None, verifier_correct: bool) -> float:
    if verdict is None:
        return 0.0
    if solver_correct:
        return 1.0 if verdict is _Verdict.AGREE else 0.0
    else:
        if verdict is _Verdict.AGREE:
            return 0.0
        return 1.0 if verifier_correct else 0.5


def _parse_verdict(response: str) -> _Verdict | None:
    if len(found := _VERDICT_PATTERN.findall(response)) != 1:
        return None
    return _Verdict(found[0])


def _compute_router_url(args, *, model_id: str) -> str:
    host, port = args.sglang_model_routers[model_id]
    return f"http://{host}:{port}/generate"


def _build_verifier_sample(solver_sample: Sample) -> Sample:
    prompt = _VERIFIER_PROMPT_TEMPLATE.format(
        question=_extract_question(solver_sample.prompt), solver_response=solver_sample.response
    )
    return Sample(
        group_index=solver_sample.group_index,
        index=solver_sample.index,
        rollout_id=solver_sample.rollout_id,
        prompt=[dict(role="user", content=prompt)],
        label=solver_sample.label,
        metadata=dict(solver_sample.metadata or {}),
        routing_key=solver_sample.routing_key,
    )


def _extract_question(prompt: str | list[dict[str, str]]) -> str:
    assert not isinstance(prompt, str), (
        "examples/multi_policy/solver_verifier.py quotes the question inside the verifier prompt, and a raw "
        "string prompt may already be chat templated with special tokens, so the dataset must use messages"
    )
    [user_content] = [message["content"] for message in prompt if message["role"] == "user"]
    return user_content


def _is_correct(response: str, *, ground_truth: str | None) -> bool:
    if ground_truth is None:
        return False
    return _extract_answer(response) == ground_truth


def _extract_answer(text: str) -> str | None:
    if found := _MARKED_ANSWER_PATTERN.findall(text):
        return _normalize_answer(found[-1])
    if found := _NUMBER_PATTERN.findall(text):
        return _normalize_answer(found[-1])
    return None


def _normalize_answer(answer: str) -> str:
    return answer.strip().rstrip(".").replace(",", "").replace("$", "").strip()
