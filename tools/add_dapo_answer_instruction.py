"""Append dapo-math-17k's answer-format instruction to every prompt of a chat-format jsonl dataset.

`--rm-type dapo` scores the line after "Answer:", which only prompts carrying the instruction ask for, so an
eval set such as aime-2024 needs it to be scored like the training prompts.

python tools/add_dapo_answer_instruction.py --src /root/datasets/aime-2024/aime-2024.jsonl \
    --dst /root/datasets/aime-2024-dapo/aime-2024-dapo.jsonl
"""

import argparse
import json
from pathlib import Path

ANSWER_INSTRUCTION = '\n\nRemember to put your answer on its own line after "Answer:".'


def add_instruction(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # written aside and renamed, so an interrupted run never leaves a dst that looks done
    partial = dst.with_name(dst.name + ".partial")
    with src.open() as reader, partial.open("w") as writer:
        for line in reader:
            row = json.loads(line)
            row["prompt"][-1]["content"] += ANSWER_INSTRUCTION
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    partial.rename(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args()
    add_instruction(args.src, args.dst)


if __name__ == "__main__":
    main()
