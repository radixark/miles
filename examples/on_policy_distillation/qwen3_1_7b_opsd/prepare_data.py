"""Build the OPSD train and eval splits with prompts already rendered.

Rendering here rather than via --apply-chat-template lets the student train with
thinking mode off while the teacher and the evaluation keep it on, which is the
configuration the paper adopts.

usage: python3 prepare_data.py <model> <openthoughts_dir> <aime24_dir> <train.jsonl> <eval.jsonl>
"""

import glob
import json
import sys

import pyarrow.parquet as pq
from transformers import AutoTokenizer

MODEL, TRAIN_DIR, AIME_DIR, OUT_TRAIN, OUT_EVAL = sys.argv[1:6]
MAX_TRAIN = 30000
MAX_SOLUTION_CHARS = 6000

SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def student_prompt(problem):
    return f"Problem: {problem}\n\n{SUFFIX}"


def eval_prompt(problem):
    return f"{problem}\n\n{SUFFIX}"


def teacher_prompt(problem, solution):
    """The teacher reads the reference solution, then answers the problem in its own words."""
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


tokenizer = AutoTokenizer.from_pretrained(MODEL)


def render(text, thinking):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def rows(pattern):
    for path in sorted(glob.glob(pattern)):
        for batch in pq.read_table(path).to_batches():
            yield from batch.to_pylist()


n_train = 0
with open(OUT_TRAIN, "w") as out:
    for row in rows(f"{TRAIN_DIR}/**/*.parquet"):
        problem, solution = row.get("problem"), row.get("solution")
        if not problem or not solution or row.get("correct") is False:
            continue
        out.write(
            json.dumps(
                {
                    "prompt": render(student_prompt(problem), thinking=False),
                    "metadata": {
                        "teacher_prompt": render(
                            teacher_prompt(problem, solution[:MAX_SOLUTION_CHARS]), thinking=True
                        )
                    },
                }
            )
            + "\n"
        )
        n_train += 1
        if n_train >= MAX_TRAIN:
            break

n_eval = 0
with open(OUT_EVAL, "w") as out:
    for row in rows(f"{AIME_DIR}/**/*.parquet"):
        problem, answer = row.get("problem"), row.get("answer")
        if not problem or answer is None:
            continue
        out.write(
            json.dumps(
                {
                    "prompt": render(eval_prompt(problem), thinking=True),
                    "label": str(answer).strip(),
                    "metadata": {"opsd_eval": True},
                }
            )
            + "\n"
        )
        n_eval += 1

print(f"train={n_train} eval={n_eval}")
