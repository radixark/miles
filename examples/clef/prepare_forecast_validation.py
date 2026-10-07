"""Convert the pinned Decision Index ForecastBench cohort to native labels."""

import gzip
import json
from pathlib import Path
from typing import Any

from tap import Tap

from examples.clef.data import file_sha256, read_examples


class Args(Tap):
    input_path: str
    output_path: str


def convert_forecast(row: dict[str, Any]) -> dict[str, Any]:
    if row["_evaluation"]["catalog_id"] != 48:
        raise ValueError("not a ForecastBench record")
    questions = row["questions"]
    if set(questions) != {"answer"} or set(questions["answer"]["criteria"]) != {"yes", "no"}:
        raise ValueError("expected a binary Yes/No forecast")
    outcome = row["expected"]["answer"]
    if outcome not in {"yes", "no"}:
        raise ValueError("forecast is unresolved")
    return {
        "record": {"id": row["_evaluation"]["run_id"], "state": row["state"], "questions": questions},
        "targets": {"answer": {key: float(key == outcome) for key in ("yes", "no")}},
        "source": "forecastbench_historical",
        "provenance": row["metadata"],
    }


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    source, destination = Path(args.input_path), Path(args.output_path)
    opener = gzip.open if source.suffix == ".gz" else open
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with opener(source, "rt") as reader, destination.open("w") as writer:
        for line in reader:
            row = json.loads(line)
            if row.get("_evaluation", {}).get("catalog_id") == 48:
                writer.write(json.dumps(convert_forecast(row), ensure_ascii=False) + "\n")
                count += 1
    if count == 0 or len(read_examples(destination)) != count:
        raise ValueError("empty or invalid ForecastBench cohort")
    receipt = {
        "records": count, "source_sha256": file_sha256(source), "sha256": file_sha256(destination),
        "scope": "Historical Decision Index ForecastBench cohort; exact Cloudflare private cohort not independently certified",
        "supervision": "validation only; resolved outcome excluded from model input",
        "primary_metric": "binary_brier",
    }
    destination.with_suffix(".manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
