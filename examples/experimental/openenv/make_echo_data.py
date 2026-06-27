"""Generate a tiny prompt dataset for the OpenEnv Echo smoke run.

Each row has the ``prompt`` / ``metadata`` keys that run-openenv-echo.py passes
via --input-key / --metadata-key. Echo is content-agnostic, so the prompts just
give the policy something to say.

    python make_echo_data.py --output /root/echo_train.jsonl --n 64
"""

import json

from tap import Tap

_INSTRUCTIONS = [
    "Greet the world in one short sentence.",
    "Write a single friendly sentence about the weather.",
    "Say something encouraging in one sentence.",
    "Describe a cat in one short sentence.",
    "Give a one-sentence fun fact.",
]


class Args(Tap):
    output: str = "/root/echo_train.jsonl"
    n: int = 64


def main() -> None:
    args = Args().parse_args()
    with open(args.output, "w") as f:
        for i in range(args.n):
            instruction = _INSTRUCTIONS[i % len(_INSTRUCTIONS)]
            row = {
                "prompt": [{"role": "user", "content": instruction}],
                "metadata": {"task_id": f"echo-{i:04d}"},
            }
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {args.n} rows to {args.output}")


if __name__ == "__main__":
    main()
