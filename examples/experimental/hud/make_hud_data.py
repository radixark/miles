"""Build a miles prompt jsonl from a HUD taskset on HuggingFace.

The HUD row travels whole in the ``metadata`` column (miles' ``--metadata-key``
puts it on ``Sample.metadata``), so the environment adapter needs no per-task
code: which image to boot, how to set the task up and how to grade it all come
from the row.

    python -m examples.experimental.hud.make_hud_data \
        --dataset hud-evals/2048-basic --repeat 256 --output /root/hud2048_train.jsonl

Tasksets with a handful of rows (2048-basic ships one template) are repeated to
fill a training file; larger ones (OSWorld-Gold, SheetBench-50) are used as-is
and ``--repeat`` stays 1.

Taskset-agnostic apart from one import: 2048 substitutes its own briefing for the
row's prompts, and hud2048_prompt.py says why.
"""

import argparse
import json
import urllib.parse
import urllib.request

from examples.experimental.hud import hud2048_prompt

ROWS_API = "https://datasets-server.huggingface.co/rows"

# Appended to the row's own system prompt: the action vocabulary is ours (a text
# DSL, so the tokens the loss is computed on are the tokens that acted), not
# something the taskset can specify.
DSL_SUFFIX = """

You act by writing text. Each turn: look at the latest screenshot, think in at most 2 short \
sentences, then put exactly ONE action on the LAST line, in one of these forms:
Action: screenshot()
Action: press("Left","Down","Left","Down")
Action: click(840, 320)
Action: type("some text")
Action: done()

{press_note}Coordinates for click are in the screenshot you were shown.
Your very first action must be Action: screenshot() to see the screen."""


def press_note(keys_per_turn: int) -> str:
    """What to tell the model about multi-key actions.

    Spending a turn's whole key budget is advice for a task with a move budget
    and near-free keystrokes. It is wrong where a turn should be one deliberate
    keystroke, so it is only said when the budget is above one.
    """
    if keys_per_turn <= 1:
        return 'press takes one key, e.g. press("Enter"), or a chord, e.g. press("ctrl+c"). '
    return (
        f"press takes up to {keys_per_turn} keys and plays them in order -- "
        f"you have few turns, so use all {keys_per_turn}. "
    )


def fetch_rows(dataset: str, split: str, limit: int) -> list[dict]:
    rows: list[dict] = []
    while len(rows) < limit:
        q = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": "default",
                "split": split,
                "offset": len(rows),
                "length": min(100, limit - len(rows)),
            }
        )
        with urllib.request.urlopen(f"{ROWS_API}?{q}", timeout=60) as resp:
            payload = json.load(resp)
        batch = [r["row"] for r in payload.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
    return rows


def as_json(value):
    """HF ships these columns as JSON strings; keep them as objects."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def task_of(row: dict) -> dict:
    """The part of a row the environment adapter reads at rollout time."""
    return {
        "id": row.get("id"),
        "mcp_config": as_json(row.get("mcp_config")),
        "setup_tool": as_json(row.get("setup_tool")),
        "evaluate_tool": as_json(row.get("evaluate_tool")),
    }


def build_prompt(row: dict, keys_per_turn: int) -> str:
    """A row's own prompts plus our action vocabulary.

    A row carries both a system prompt (how to behave) and a prompt (what this
    instance asks for). For most tasksets the instruction *is* the task, so both
    go in.
    """
    system = row.get("system_prompt") or ""
    instruction = row.get("prompt") or ""
    if hud2048_prompt.applies(as_json(row.get("setup_tool"))):
        system, instruction = hud2048_prompt.GAME_2048_HINT, ""
    head = "\n\n".join(p for p in (system, instruction) if p)
    return (head + DSL_SUFFIX.format(press_note=press_note(keys_per_turn))).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hud-evals/2048-basic")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=1000, help="rows to pull from the taskset")
    ap.add_argument("--repeat", type=int, default=1, help="repeat the taskset to fill the file")
    ap.add_argument("--keys-per-turn", type=int, default=4)
    ap.add_argument("--output", default="/root/hud_train.jsonl")
    args = ap.parse_args()

    rows = fetch_rows(args.dataset, args.split, args.limit)
    if not rows:
        raise SystemExit(f"no rows from {args.dataset}")
    print(f"fetched {len(rows)} row(s) from {args.dataset}")

    written = 0
    with open(args.output, "w") as f:
        for _ in range(args.repeat):
            for row in rows:
                record = {
                    "prompt": build_prompt(row, args.keys_per_turn),
                    "label": "",
                    "metadata": {"hud_task": task_of(row)},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    print(f"wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
