"""Compare real SGLang outputs with the native checkpoint loader before scoring."""

import json
from pathlib import Path

import httpx
import torch
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


@torch.inference_mode()
def verify(args: Args) -> None:
    tasks = load_tasks(args.benchmark)
    selected = []
    for kind in ("choice", "noul", "score"):
        selected.append(next(task for task in tasks if task["question"]["type"] == kind))
    selected.append(max(tasks, key=lambda task: len(json.dumps(task["state"]))))
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
            checks.append(check)
            print(json.dumps(check), flush=True)
            if difference > args.tolerance:
                raise ValueError(f"SGLang/native parity failed: {difference} > {args.tolerance}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": True, "tolerance": args.tolerance, "checks": checks}, indent=2) + "\n")


if __name__ == "__main__":
    verify(Args(underscores_to_dashes=True).parse_args())
