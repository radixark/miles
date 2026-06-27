"""Generate a tiny prompt dataset for the OpenEnv Coding learning run.

Each row has the ``prompt`` / ``metadata`` keys that run-openenv-coding.py passes
via --input-key / --metadata-key. The prompts ask the policy to write short
Python snippets; the Coding env's safe-coding transform rewards concise, safe,
syntactically valid code, so the policy has a real signal to learn from.

The system instruction nudges the model toward the reward: emit a single short
fenced ```python block, avoid os/subprocess/eval/exec/open, keep it under ~100
chars. The adapter strips the fence before stepping the env.

    python make_coding_data.py --output /root/coding_train.jsonl --n 64
"""

import json

from tap import Tap

_SYSTEM = (
    "You are a Python coding assistant. Respond with a SINGLE short Python "
    "snippet inside one ```python code block and nothing else. Keep it under "
    "100 characters. Do NOT use os, subprocess, eval, exec, __import__, or open."
)

_TASKS = [
    "Print the sum of 2 and 3.",
    "Print the numbers 0 through 4 on one line.",
    "Compute and print the square of 7.",
    "Print 'hello' three times separated by spaces.",
    "Print the maximum of the list [4, 9, 1, 6].",
    "Print the length of the string 'banana'.",
    "Print the first 5 even numbers as a list.",
    "Print the reverse of the string 'python'.",
]


class Args(Tap):
    output: str = "/root/coding_train.jsonl"
    n: int = 64


def main() -> None:
    args = Args().parse_args()
    with open(args.output, "w") as f:
        for i in range(args.n):
            task = _TASKS[i % len(_TASKS)]
            row = {
                "prompt": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": task},
                ],
                "metadata": {"task_id": f"coding-{i:04d}"},
            }
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {args.n} rows to {args.output}")


if __name__ == "__main__":
    main()
