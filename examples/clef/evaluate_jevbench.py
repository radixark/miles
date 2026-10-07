"""Evaluate deterministic trained Clef distributions on the public JevBench subset."""

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
from jevbench.metrics import brier_score, ece_top_label, latency_summary
from jevbench.scoring import score_task
from tap import Tap


class Args(Tap):
    benchmark: Path
    endpoint: str
    output: Path
    model: str = "clef"


def request_body(task: dict, model: str) -> dict:
    return {"model": model, "id": task["id"], "state": task["state"],
            "questions": {"decision": task["question"]}}


def load_tasks(benchmark: Path) -> list[dict]:
    tasks = []
    for tier in ("easy", "original", "hard"):
        for line in (benchmark / "datasets/public" / (tier + ".jsonl")).read_text().splitlines():
            task = json.loads(line)
            task["public_file"] = tier
            tasks.append(task)
    if len(tasks) != 231 or len({task["id"] for task in tasks}) != 231:
        raise ValueError("Expected the pinned 231-item public JevBench subset")
    return tasks


def summarize(rows: list[dict]) -> dict:
    groups = {"all": rows}
    for field in ("type", "public_file", "family"):
        for value in sorted({row[field] for row in rows}):
            groups[field + ":" + value] = [row for row in rows if row[field] == value]
    report = {}
    for name, group in groups.items():
        valid = [row for row in group if row["valid"]]
        scorable = [row for row in group if row["expected"] is not None]
        calibrated = [row for row in valid if row["expected"] is not None]
        ordinal = [row for row in calibrated if row["type"] == "score"]
        report[name] = {
            "rows": len(group), "scorable_rows": len(scorable), "valid_rows": len(valid),
            "valid_pct": 100 * len(valid) / len(group),
            "accuracy_all_scorable": sum(bool(row["correct"]) for row in scorable) / len(scorable) if scorable else None,
            "brier_valid_only": sum(row["brier"] for row in calibrated) / len(calibrated) if calibrated else None,
            "ece_valid_only": ece_top_label([(max(row["probs"].values()), row["correct"]) for row in calibrated]),
            "collapse_pct_all": 100 * sum(row["collapse"] for row in group) / len(group),
            "near_collapse_99_pct_all": 100 * sum(row["near_collapse_99"] for row in group) / len(group),
            "ordinal_mae_valid_only": sum(abs(row["ordinal_ev"] - float(row["expected"])) for row in ordinal) / len(ordinal) if ordinal else None,
            "latency": latency_summary([row["latency_s"] for row in group]),
        }
    return report


async def evaluate(args: Args) -> None:
    tasks = load_tasks(args.benchmark)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=600) as client:
        warmup = await client.post(args.endpoint + "/v1/systemone", json=request_body(tasks[0], args.model))
        warmup.raise_for_status()
        with args.output.open("w") as stream:
            for task in tasks:
                started = time.perf_counter()
                response = await client.post(args.endpoint + "/v1/systemone", json=request_body(task, args.model))
                response.raise_for_status()
                elapsed = time.perf_counter() - started
                output = response.json()
                probabilities = output["probabilities"]["decision"]
                if task["question"]["type"] == "noul":
                    probabilities = {"yes": probabilities["true"], "no": probabilities["false"]}
                # Explicit canonical labels preserve the benchmark's label and tie conventions.
                probabilities = {label: probabilities[label] for label in task["labels"]}
                scored = score_task(probabilities, SimpleNamespace(**task))
                clean = scored.get("probs")
                expected = task.get("expected")
                row = {"id": task["id"], "family": task["family"], "type": task["question"]["type"],
                       "public_file": task["public_file"], "labels": task["labels"], "expected": expected,
                       "request": request_body(task, args.model), "response": output, "latency_s": elapsed,
                       "brier": brier_score(clean, str(expected), task["labels"]) if clean and expected is not None else None,
                       "collapse": bool(clean and max(clean.values()) == 1),
                       "near_collapse_99": bool(clean and max(clean.values()) >= 0.99), **scored}
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if len(rows) % 25 == 0:
                    print(f"Completed {len(rows)}/{len(tasks)}", flush=True)
    report = {"protocol": {"benchmark_commit": "bb05a335bc809e61b20c0f745d25499a82b326fc",
                           "inference": "deterministic Clef head; no generated response tokens",
                           "latency": "serial requests; warmup excluded"}, "metrics": summarize(rows)}
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["metrics"]["all"], indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(evaluate(Args(underscores_to_dashes=True).parse_args()))
