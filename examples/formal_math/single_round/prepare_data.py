from typing import Annotated

import typer
import re
from pathlib import Path

from datasets import load_dataset

self_stem = Path(__file__).stem
dir_data = Path(f"/host_home/server_data/{self_stem}")

# https://github.com/deepseek-ai/DeepSeek-Prover-V2
_PROMPT_TEMPLATE = """
Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()


def process_flc(
    train_flc_select_num_rows: int,
    val_flc_select_num_rows: int,
):
    ds = load_dataset("m-a-p/FineLeanCorpus", split="train")
    ds = ds.shuffle(seed=42)
    ds = ds.select_columns("id", "statement", "lean_code")
    ds = _add_metadata_column(ds, dataset_name="flc")

    def _process_prompt(x):
        x = _convert_to_by_sorry(x)
        x = _PROMPT_TEMPLATE.format(x)
        x = _to_messages(x)
        return x

    def _process_batch(batch):
        return {"prompt": [_process_prompt(x) for x in batch["prompt"]]}

    ds = ds.map(_process_batch, batched=True, num_proc=128)
    _write_file(ds, "flc_train")
    _write_file(TODO, "flc_val")


def process_minif2f():
    ds = load_dataset("AI-MO/minif2f_test", split="train")
    ds = ds.shuffle(seed=42)
    ds = _add_metadata_column(ds, dataset_name="minif2f")
    ds = ds.remove_columns(["name", "informal_prefix"])

    def _process_prompt(x):
        x = _convert_to_by_sorry(x)
        x = _PROMPT_TEMPLATE.format(x)
        x = _to_messages(x)
        return x

    def _process_batch(batch):
        return {"prompt": [_process_prompt(x) for x in batch["formal_statement"]]}

    ds = ds.map(_process_batch, batched=True)
    _write_file(ds, "minif2f")


def _write_file(ds, stem):
    path = dir_data / f"{stem}.jsonl"
    ds.to_json(path)
    print(f"Write to {path}")
    print("Example data", ds[:3])


def _convert_to_by_sorry(s: str):
    if "by sorry" in s:
        return s
    return _ensure_remove_pattern(s, r' *:=\n? *(by)? *\n?$') + " := by\n  sorry"


def _ensure_remove_pattern(text: str, pattern: str):
    assert re.search(pattern, text, flags=re.MULTILINE), f"{pattern=} {text=}"
    return re.sub(pattern, '', text, flags=re.MULTILINE)


def _to_messages(content):
    return [{"role": "user", "content": content}]



def _add_metadata_column(ds, dataset_name: str):
    return ds.add_column("metadata", [dict(question_id=f"{dataset_name}__idx{i}") for i in range(len(ds))])


def main(
    train_flc_select_num_rows: Annotated[int, typer.Option()] = 20000,
    val_flc_select_num_rows: Annotated[int, typer.Option()] = 200,
):
    process_flc(
        train_flc_select_num_rows=train_flc_select_num_rows,
        val_flc_select_num_rows=val_flc_select_num_rows,
    )
    process_minif2f()


if __name__ == "__main__":
    typer.run(main)
