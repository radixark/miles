"""Render problem-only student prompts and solution-conditioned teacher prompts."""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from miles.utils.processing_utils import load_tokenizer

SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def _teacher_prompt(problem, solution):
    return (
        f"Problem: {problem}\n\n"
        "Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n\n"
        "After reading the reference solution above, make sure you truly understand the "
        "reasoning behind each step, do not copy or paraphrase it. Now, using your own words "
        "and independent reasoning, derive the same final answer to the problem above. Think "
        "step by step, explore different approaches, and don't be afraid to backtrack or "
        f"reconsider if something doesn't work out:\n\n{SUFFIX}"
    )


def _rows(directory):
    for path in sorted(Path(directory).rglob("*.parquet")):
        for batch in pq.read_table(path).to_batches():
            yield from batch.to_pylist()


def prepare(model, train_dir, aime_dir, out_train, out_eval):
    tokenizer = load_tokenizer(model)

    def render(text, thinking):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )

    n_train = 0
    with open(out_train, "w") as output:
        for row in _rows(train_dir):
            problem, solution = row.get("problem"), row.get("solution")
            if not problem or not solution or row.get("correct") is False:
                continue
            record = {
                "prompt": render(f"Problem: {problem}\n\n{SUFFIX}", thinking=False),
                "metadata": {"teacher_prompt": render(_teacher_prompt(problem, solution[:6000]), thinking=True)},
            }
            output.write(json.dumps(record) + "\n")
            n_train += 1
            if n_train == 30000:
                break

    n_eval = 0
    with open(out_eval, "w") as output:
        for row in _rows(aime_dir):
            problem, answer = row.get("problem"), row.get("answer")
            if not problem or answer is None:
                continue
            record = {
                "prompt": render(f"{problem}\n\n{SUFFIX}", thinking=True),
                "label": str(answer).strip(),
                "metadata": {"opsd_eval": True},
            }
            output.write(json.dumps(record) + "\n")
            n_eval += 1
    print(f"train={n_train} eval={n_eval}")


if __name__ == "__main__":
    prepare(*sys.argv[1:6])
