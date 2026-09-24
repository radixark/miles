"""Convert native Workplace rows to Miles input without exposing gold actions."""

import hashlib
import json
from pathlib import Path

from tap import Tap


def convert(source: Path, target: Path, max_turns: int = 24) -> dict:
    if max_turns < 1 or source.resolve() == target.resolve():
        raise ValueError("Use a positive turn limit and a target distinct from the native dataset")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Expected nonempty rows with unique task IDs")
    target.parent.mkdir(parents=True, exist_ok=True)
    converted = []
    for row in rows:
        params = row["responses_create_params"]
        if set(params) - {"input", "tools", "parallel_tool_calls", "temperature"}:
            raise ValueError("Unreviewed policy field in native row")
        if not params["tools"] or not params["input"] or not row["ground_truth"]:
            raise ValueError("Missing policy input, tools, or verifier target")
        output = {
            "prompt": params["input"],
            "label": str(row["id"]),
            "metadata": {
                "workplace_task_id": row["id"],
                "workplace_category": row["category"],
                "workplace_policy": params,
                "workplace_max_turns": max_turns,
            },
        }
        converted.append(json.dumps(output, ensure_ascii=False) + "\n")
    target.write_text("".join(converted))
    return {
        "tasks": len(rows),
        "native_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "miles_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "max_turns": max_turns,
    }


class Args(Tap):
    source: Path
    target: Path
    max_turns: int = 24


def main() -> None:
    args = Args().parse_args()
    print(json.dumps(convert(args.source, args.target, args.max_turns)))


if __name__ == "__main__":
    main()
