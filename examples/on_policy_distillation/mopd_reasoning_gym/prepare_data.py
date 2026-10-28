"""Build the caesar_cipher + simple_geometry prompts for multi-teacher OPD on reasoning_gym.

Matches the teacher cards (semianalysisai/Qwen3.6-35B-A3B-{caesar-cipher,simple-geometry}-GRPO):
reasoning_gym 0.1.25 default task configs, train pool seed 20000, held-out eval seed 10000,
system prompt reasoning_gym SYSTEM_PROMPTS["default"], prompts over 2,048 tokens dropped.
Thinking stays on; scripts/run_qwen3_6_35b_a3b_mopd.py applies the chat template with enable_thinking=true
and runs this script from its `prepare` entrypoint.

Writes, under --out:
    mixed-train.jsonl                 both tasks, shuffled, routed by metadata.task
    eval-caesar_cipher.jsonl          held-out, seed 10000
    eval-simple_geometry.jsonl        held-out, seed 10000
    manifest.json                     sizes, seeds, and sha256 of every file

Example:
    python examples/on_policy_distillation/mopd_reasoning_gym/prepare_data.py \
        --out /root/datasets/mopd-reasoning-gym --tokenizer /root/models/Qwen3.6-35B-A3B
"""

import argparse
import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path

import reasoning_gym
from reasoning_gym.utils import SYSTEM_PROMPTS

TASKS = ("caesar_cipher", "simple_geometry")
TRAIN_SEED = 20000
EVAL_SEED = 10000


def _prompt_tokens(tokenizer, messages):
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True)
    # transformers v5 returns a BatchEncoding, v4 a list of ids
    return len(encoded["input_ids"] if isinstance(encoded, Mapping) else encoded)


def _rows(task, size, seed, tokenizer, max_prompt_tokens):
    rows, dropped = [], 0
    for index, entry in enumerate(reasoning_gym.create_dataset(task, size=size, seed=seed)):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["default"]},
            {"role": "user", "content": entry["question"]},
        ]
        if tokenizer is not None and _prompt_tokens(tokenizer, messages) > max_prompt_tokens:
            dropped += 1
            continue
        rows.append(
            {
                "messages": messages,
                "label": str(entry["answer"]),
                "metadata": {"task": task, "source_seed": seed, "source_index": index, "entry": entry},
            }
        )
    return rows, dropped


def _write(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default=None, help="HF checkpoint used to drop prompts over the limit")
    parser.add_argument("--train-size", type=int, default=10000, help="Train prompts generated per task")
    parser.add_argument("--eval-size", type=int, default=128, help="Held-out prompts per task")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"reasoning_gym": "0.1.25", "train_seed": TRAIN_SEED, "eval_seed": EVAL_SEED, "files": {}}

    train = []
    for task in TASKS:
        rows, dropped = _rows(task, args.train_size, TRAIN_SEED, tokenizer, args.max_prompt_tokens)
        train += rows
        manifest[f"train_{task}"] = {"rows": len(rows), "dropped_over_limit": dropped}
        rows, dropped = _rows(task, args.eval_size, EVAL_SEED, tokenizer, args.max_prompt_tokens)
        name = f"eval-{task}.jsonl"
        manifest["files"][name] = _write(out / name, rows)
        manifest[f"eval_{task}"] = {"rows": len(rows), "dropped_over_limit": dropped}

    random.Random(0).shuffle(train)
    manifest["files"]["mixed-train.jsonl"] = _write(out / "mixed-train.jsonl", train)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
