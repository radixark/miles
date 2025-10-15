import datetime
import pprint
import random
import re
from pathlib import Path
from typing import Annotated

import typer
from datasets import load_dataset

self_stem = Path(__file__).stem

# https://github.com/deepseek-ai/DeepSeek-Prover-V2
_PROMPT_TEMPLATE = """
Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()


_NEEDLE_THEOREM = "theorem "


def process_flc(
    dir_output: Path,
    train_flc_select_num_rows: int,
    val_flc_select_num_rows: int,
):
    ds = load_dataset("m-a-p/FineLeanCorpus", split="train")
    ds = _add_metadata_column(ds, dataset_name="flc")

    def _filter_batch(batch):
        return [
            # we remove multi-theorem data currently
            lean_code.count(_NEEDLE_THEOREM) == 1
            for lean_code in batch["lean_code"]
        ]

    ds = ds.filter(_filter_batch, batched=True, num_proc=64)

    ds = ds.shuffle(seed=42)
    ds = ds.select_columns(["id", "statement", "lean_code"])
    ds = ds.select(range(train_flc_select_num_rows + val_flc_select_num_rows))
    ds = ds.train_test_split(test_size=val_flc_select_num_rows, shuffle=False, seed=42)

    def _process_prompt(statement, lean_code):
        assert lean_code.count(_NEEDLE_THEOREM) == 1, f"{lean_code=}"
        x = lean_code.replace(_NEEDLE_THEOREM, f"/- {statement} -/\n{_NEEDLE_THEOREM}")

        x = _PROMPT_TEMPLATE.format(x)
        x = _to_messages(x)
        return x

    def _process_batch(batch):
        return {
            "prompt": [
                _process_prompt(statement, lean_code)
                for statement, lean_code in zip(batch["statement"], batch["lean_code"], strict=True)
            ]
        }

    ds = ds.map(_process_batch, batched=True, num_proc=64, remove_columns=["statement", "lean_code"])
    _write_file(ds["train"], dir_output / "flc_train.jsonl")
    _write_file(ds["test"], dir_output / "flc_test.jsonl")


def process_minif2f(
    dir_output: Path,
):
    ds = load_dataset("AI-MO/minif2f_test", split="train")
    ds = _add_metadata_column(ds, dataset_name="minif2f")
    ds = ds.shuffle(seed=42)
    ds = ds.remove_columns(["name", "informal_prefix"])

    def _process_prompt(x):
        x = _convert_to_by_sorry(x)
        x = _PROMPT_TEMPLATE.format(x)
        x = _to_messages(x)
        return x

    def _process_batch(batch):
        return {"prompt": [_process_prompt(x) for x in batch["formal_statement"]]}

    ds = ds.map(_process_batch, batched=True)
    _write_file(ds, dir_output / "minif2f_test.jsonl")


def _write_file(ds, path):
    ds.to_json(path)
    print(f"Write to {path}, {len(ds)=}, example data:")
    pprint.pprint([ds[i] for i in range(3)])


def _convert_to_by_sorry(s: str):
    return _ensure_remove_pattern(s, r" *:=\n? *(by)? *\n?$") + " := by\n  sorry"


def _ensure_remove_pattern(text: str, pattern: str):
    assert re.search(pattern, text, flags=re.MULTILINE), f"{pattern=} {text=}"
    return re.sub(pattern, "", text, flags=re.MULTILINE)


def _to_messages(content):
    return [{"role": "user", "content": content}]


def _add_metadata_column(ds, dataset_name: str):
    return ds.add_column("metadata", [dict(question_id=f"{dataset_name}__idx{i}") for i in range(len(ds))])


def main(
    dir_output_base: Annotated[str, typer.Option()],
    train_flc_select_num_rows: Annotated[int, typer.Option()] = 20000,
    val_flc_select_num_rows: Annotated[int, typer.Option()] = 100,
):
    dir_output = (
        Path(dir_output_base) / f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(0, 1000000)}"
    )
    dir_output.mkdir(parents=True, exist_ok=True)

    process_flc(
        dir_output=dir_output,
        train_flc_select_num_rows=train_flc_select_num_rows,
        val_flc_select_num_rows=val_flc_select_num_rows,
    )
    process_minif2f(
        dir_output=dir_output,
    )


if __name__ == "__main__":
    typer.run(main)
