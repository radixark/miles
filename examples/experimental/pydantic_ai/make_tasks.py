"""Write arithmetic tasks for ``pydantic_ai_agent_function.run`` as JSONL.

  python examples/experimental/pydantic_ai/make_tasks.py /root/datasets/pydantic_ai/train.jsonl
  python examples/experimental/pydantic_ai/make_tasks.py /root/datasets/pydantic_ai/eval.jsonl --num-tasks 128 --seed 1
"""

import json
import operator
import random
from pathlib import Path

import typer

_OPERATORS = {"+": operator.add, "-": operator.sub, "*": operator.mul}


def main(output: Path, num_tasks: int = 2048, seed: int = 0) -> None:
    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for _ in range(num_tasks):
            a, b, c = (rng.randint(100, 9999) for _ in range(3))
            first, second = rng.choice(list(_OPERATORS)), rng.choice(list(_OPERATORS))
            answer = _OPERATORS[second](_OPERATORS[first](a, b), c)
            question = f"What is ({a} {first} {b}) {second} {c}?"
            row = {"messages": [{"role": "user", "content": question}], "metadata": {"answer": answer}}
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    typer.run(main)
