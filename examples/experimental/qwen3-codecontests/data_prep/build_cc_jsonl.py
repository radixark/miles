#!/usr/bin/env python3
"""Build Miles prompt JSONLs from extracted CodeContests Harbor task dirs.

Walks ``<tasks>/`` (output of extract_codecontests.py), reading each task's
``instruction.md`` as the prompt, and emits BOTH:

  1. Per-difficulty files ``cc_train_<bucket>.jsonl`` (one per difficulty bucket).
  2. A single combined file (default ``cc_train_all_sorted.jsonl``) containing
     every task ordered by INCREASING difficulty (Codeforces rating ascending),
     with unrated tasks last. This is the curriculum file the training run
     consumes as ``--prompt-data``.

Difficulty comes from the Codeforces ``Rating`` field in each task's
``instruction.md`` (the only real per-problem difficulty signal; the
``difficulty`` field in task.toml is a constant "medium" and is ignored).

Each output line is a Miles sample:
    {"prompt": <instruction.md>,
     "metadata": {"instance_id": <dir>, "agent_name": "mini-swe-agent",
                  "split": "train", "rating": <int|null>, "difficulty": <bucket>}}

CRITICAL: ``metadata.instance_id`` MUST equal the task directory name, because
``server.py`` resolves the task under ``HARBOR_TASKS_DIR`` by that id. A mismatch
yields ``TaskNotFound`` and silent 0.0 rewards.

Buckets (by Codeforces rating):
    easy        : rating <  1200
    easy_medium : 1200 <= rating < 1600
    medium      : 1600 <= rating < 2000
    hard        : 2000 <= rating < 2400
    very_hard   : rating >= 2400
    unrated     : no Rating / Rating == 0

Usage:
    # emit per-difficulty files + the combined ascending-difficulty file
    python build_cc_jsonl.py --tasks ~/harbor_tasks_cc --out-dir ./data

    # quick validation on the first 20 tasks
    python build_cc_jsonl.py --tasks ~/harbor_tasks_cc --out-dir ./data --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# Ordered easiest -> hardest; the combined file follows this order (unrated last).
BUCKETS = ["easy", "easy_medium", "medium", "hard", "very_hard", "unrated"]
_RATING_RE = re.compile(r"\*\*Rating\*\*:\s*(\d+)")


def bucket_for(rating: int | None) -> str:
    if not rating:  # None or 0
        return "unrated"
    if rating < 1200:
        return "easy"
    if rating < 1600:
        return "easy_medium"
    if rating < 2000:
        return "medium"
    if rating < 2400:
        return "hard"
    return "very_hard"


def parse_rating(instruction_text: str) -> int | None:
    m = _RATING_RE.search(instruction_text)
    if not m:
        return None
    r = int(m.group(1))
    return r if r > 0 else None


def build(
    tasks_dir: str,
    out_dir: str,
    *,
    agent_name: str = "mini-swe-agent",
    split: str = "train",
    limit: int | None = None,
    combined_name: str = "cc_train_all_sorted.jsonl",
) -> dict[str, int]:
    tasks = Path(os.path.expanduser(tasks_dir))
    if not tasks.is_dir():
        raise FileNotFoundError(f"tasks dir not found: {tasks}")
    out = Path(os.path.expanduser(out_dir))
    out.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    for d in sorted(p for p in tasks.iterdir() if p.is_dir()):
        instr = d / "instruction.md"
        if not instr.is_file():
            continue
        text = instr.read_text(errors="replace")
        prompt = text.strip()
        if not prompt:
            continue
        rating = parse_rating(text)
        samples.append(
            {
                "prompt": prompt,
                "metadata": {
                    "instance_id": d.name,  # MUST match the task dir name
                    "agent_name": agent_name,
                    "split": split,
                    "rating": rating,
                    "difficulty": bucket_for(rating),
                },
            }
        )
        if limit is not None and len(samples) >= limit:
            break

    # Increasing difficulty: rated tasks by ascending rating, then unrated last.
    # (rating is None) sorts True > False, so unrated lands at the end; ties broken
    # by instance_id for determinism.
    samples.sort(key=lambda s: (s["metadata"]["rating"] is None, s["metadata"]["rating"] or 0, s["metadata"]["instance_id"]))

    counts = {b: 0 for b in BUCKETS}
    combined_path = out / combined_name
    from contextlib import ExitStack
    with ExitStack() as stack:
        bucket_files = {
            b: stack.enter_context(open(out / f"cc_train_{b}.jsonl", "w"))
            for b in BUCKETS
        }
        with open(combined_path, "w") as combined:
            for s in samples:
                line = json.dumps(s) + "\n"
                combined.write(line)
                b = s["metadata"]["difficulty"]
                bucket_files[b].write(line)
                counts[b] += 1

    total = sum(counts.values())
    print(f"wrote {total} samples -> {out}")
    for b in BUCKETS:
        print(f"  cc_train_{b}.jsonl : {counts[b]}")
    print(f"combined (increasing difficulty): {total} lines -> {combined_path.name}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CodeContests task dirs -> per-difficulty JSONLs + a combined ascending-difficulty JSONL"
    )
    ap.add_argument("--tasks", default=os.path.expanduser("~/harbor_tasks_cc"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--agent-name", default="mini-swe-agent")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--combined-name", default="cc_train_all_sorted.jsonl",
                    help="filename for the combined increasing-difficulty JSONL (written under --out-dir)")
    args = ap.parse_args()
    build(
        args.tasks,
        args.out_dir,
        agent_name=args.agent_name,
        split=args.split,
        limit=args.limit,
        combined_name=args.combined_name,
    )


if __name__ == "__main__":
    main()
