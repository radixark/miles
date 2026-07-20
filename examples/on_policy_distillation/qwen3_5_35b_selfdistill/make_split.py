"""Carve a disjoint train/eval split from dapo-math-17k for the self-distillation example.

The file is ordered by difficulty, so we shuffle with a FIXED SEED before splitting
(a contiguous tail-N split would be systematically easier and bias the eval). The
512-problem eval split is held out from BOTH phases. Dedup is on prompt text (labels
are not unique). Usage:

    python make_split.py --src /path/dapo-math-17k.jsonl --out-dir /path/split
"""

import argparse
import hashlib
import json
import os
import random


def prompt_text(d):
    p = d["prompt"]
    return "\n".join(m.get("content", "") for m in p) if isinstance(p, list) else str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dapo-math-17k.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows, seen = [], set()
    with open(args.src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key = prompt_text(json.loads(line))
            if key in seen:
                continue
            seen.add(key)
            rows.append(line)

    random.Random(args.seed).shuffle(rows)  # fixed seed: reproducible, unbiased split
    eval_rows, train_rows = rows[-args.eval_n :], rows[: -args.eval_n]

    ek = {prompt_text(json.loads(l)) for l in eval_rows}
    tk = {prompt_text(json.loads(l)) for l in train_rows}
    assert ek.isdisjoint(tk), "LEAK: eval prompt found in train split"

    with open(os.path.join(args.out_dir, "dapo_train.jsonl"), "w") as f:
        f.write("\n".join(train_rows) + "\n")
    with open(os.path.join(args.out_dir, "dapo_eval.jsonl"), "w") as f:
        f.write("\n".join(eval_rows) + "\n")

    md5 = hashlib.md5("\n".join(eval_rows).encode()).hexdigest()
    print(f"train={len(train_rows)} eval={len(eval_rows)} seed={args.seed} eval_md5={md5}")


if __name__ == "__main__":
    main()
