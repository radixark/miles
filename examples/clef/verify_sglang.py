"""Compare real SGLang outputs with the native checkpoint loader before scoring."""

import json
from pathlib import Path

import httpx
import torch
from jevbench.metrics import brier_score
from tap import Tap

from examples.clef.evaluate_jevbench import load_tasks, request_body
from examples.clef.joint_schema_model import collate_records, encode_record, load_release_model


class Args(Tap):
    model_path: Path
    benchmark: Path
    endpoint: str
    output: Path
    device: str = "cuda:1"
    tolerance: float = 0.02
    full_benchmark: bool = False
    report_only: bool = False


@torch.inference_mode()
def verify(args: Args) -> None:
    tasks = load_tasks(args.benchmark)
    selected = []
    for kind in ("choice", "noul", "score"):
        selected.append(next(task for task in tasks if task["question"]["type"] == kind))
    selected.append(max(tasks, key=lambda task: len(json.dumps(task["state"]))))
    if args.full_benchmark:
        selected = tasks
    model, processor = load_release_model(args.model_path, device=args.device, attn_implementation="sdpa")
    checks = []
    with httpx.Client(timeout=600) as client:
        for task in selected:
            body = request_body(task, "clef")
            encoded = encode_record(processor.tokenizer, body)
            logits = model(collate_records([encoded], processor.tokenizer.pad_token_id, torch.device(args.device)))[0][0]
            reference = dict(zip(encoded.questions[0].option_ids, logits.float().softmax(-1).tolist(), strict=True))
            response = client.post(args.endpoint + "/v1/systemone", json=body)
            response.raise_for_status()
            actual = response.json()["probabilities"]["decision"]
            difference = max(abs(reference[key] - actual[key]) for key in reference)
            check = {"id": task["id"], "type": task["question"]["type"], "input_tokens": len(encoded.input_ids),
                     "reference": reference, "sglang": actual, "maximum_absolute_difference": difference}
            expected = task.get("expected")
            canonical_reference, canonical_actual = reference, actual
            if task["question"]["type"] == "noul":
                canonical_reference = {"yes": reference["true"], "no": reference["false"]}
                canonical_actual = {"yes": actual["true"], "no": actual["false"]}
            if expected is not None:
                check["reference_brier"] = brier_score(canonical_reference, str(expected), task["labels"])
                check["sglang_brier"] = brier_score(canonical_actual, str(expected), task["labels"])
                check["reference_predicted"] = max(sorted(canonical_reference), key=canonical_reference.__getitem__)
                check["sglang_predicted"] = max(sorted(canonical_actual), key=canonical_actual.__getitem__)
                check["reference_correct"] = check["reference_predicted"] == str(expected)
                check["sglang_correct"] = check["sglang_predicted"] == str(expected)
            checks.append(check)
            if not args.full_benchmark:
                print(json.dumps(check), flush=True)
            elif len(checks) % 25 == 0:
                print(f"Compared {len(checks)}/{len(selected)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    passed = all(check["maximum_absolute_difference"] <= args.tolerance for check in checks)
    scorable = [check for check in checks if "reference_brier" in check]
    summary = {"passed": passed, "tolerance": args.tolerance, "rows": len(checks),
               "maximum_absolute_difference": max(check["maximum_absolute_difference"] for check in checks),
               "reference_brier": sum(check["reference_brier"] for check in scorable) / len(scorable),
               "sglang_brier": sum(check["sglang_brier"] for check in scorable) / len(scorable),
               "reference_accuracy": sum(check["reference_correct"] for check in scorable) / len(scorable),
               "sglang_accuracy": sum(check["sglang_correct"] for check in scorable) / len(scorable),
               "changed_argmax": sum(check["reference_predicted"] != check["sglang_predicted"] for check in scorable)}
    args.output.write_text(json.dumps({**summary, "checks": checks}, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if not passed and not args.report_only:
        raise ValueError("SGLang/native differences exceeded tolerance; inspect saved parity report")


if __name__ == "__main__":
    verify(Args(underscores_to_dashes=True).parse_args())
