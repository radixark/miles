"""Write the text-to-SQL recipe's task sets as miles prompt-data jsonl.

Reads the recipe's data release (Hugging Face dataset `scottfitz/text-to-sql-recipe-data`) with the recipe's own
loader, which checks every file and database against its pinned SHA-256, and writes one row per task:

    {"prompt": <opening messages>, "tools": [<execute_sql_query schema>], "label": <task id>,
     "metadata": {"task": <the task row, database_path absolute>}}

Splits:
    train       BIRD-Platinum training tasks, without the 31 the recipe's hint mask leaves out (2,033)
    validation  BIRD-Platinum validation tasks (398)
    arcwise     Arcwise-Plat-SQL (498), graded with the ReViSQL authors' 1% single-value tolerance

Usage:
    python examples/text_to_sql/convert_dataset.py --data-dir /root/datasets/text-to-sql-recipe-data \
        --output-dir /root/datasets/text-to-sql
"""

import json
from pathlib import Path
from typing import Annotated

import typer
from text_to_sql_recipe import data, prompts


def _rows(split: str, data_dir: Path) -> list[dict]:
    match split:
        case "train":
            return data.apply_hint_mask(data.load_dataset("revisql-verified", data_dir))
        case "validation":
            return data.load_dataset("revisql-verified-validation", data_dir)
        case "arcwise":
            return [{**row, "numeric_tolerance": True} for row in data.load_dataset(data.ARCWISE, data_dir)]


def main(data_dir: Annotated[Path, typer.Option()], output_dir: Annotated[Path, typer.Option()]):
    output_dir.mkdir(parents=True, exist_ok=True)
    tool = prompts.sql_tool_schema({})
    for split in ("train", "validation", "arcwise"):
        rows = _rows(split, data_dir)
        path = output_dir / f"{split}.jsonl"
        with path.open("w") as f:
            for row in rows:
                record = {
                    "prompt": prompts.opening_messages(row, few_shot=True),
                    "tools": [tool],
                    "label": str(row["id"]),
                    "metadata": {"task": row},
                }
                f.write(json.dumps(record) + "\n")
        print(f"{split}: {len(rows)} tasks -> {path}")


if __name__ == "__main__":
    typer.run(main)
