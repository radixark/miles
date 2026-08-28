"""Filter SFT rows with the tokenizer and loss mask used by training.

The output keeps every source field for rows that produce at least one
assistant loss token and fit within ``--max-seq-len``. Rejected rows are
represented only by their source row index and rejection reason, so the audit
file does not duplicate potentially sensitive conversation content.

Example:
  python scripts/tools/filter_sft_dataset.py \
    --dataset /root/datasets/train.jsonl \
    --output-dataset /root/datasets/train.filtered.jsonl \
    --reject-report /root/datasets/train.rejected.jsonl \
    --summary-path /root/datasets/train.filtered.summary.json \
    --model /root/models/Qwen3.6-35B-A3B \
    --tools-key tools \
    --max-seq-len 262144
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from tap import Tap

from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer
from miles.utils.sft_dataset_utils import (
    RowStats,
    iter_sft_dataset,
    summarize_sft_dataset,
    validate_sft_row,
)


class Arguments(Tap):
    dataset: Path
    output_dataset: Path
    reject_report: Path
    model: Path
    summary_path: Path | None = None
    input_key: str = "messages"
    tools_key: str = ""
    chat_template_path: Path | None = None
    loss_mask_type: Literal["qwen", "qwen3", "distill_qwen"] = "qwen3"
    max_seq_len: int = 65536

    def configure(self) -> None:
        self.add_argument("--dataset", type=Path, help="Source JSONL or Parquet dataset.")
        self.add_argument(
            "--output-dataset",
            type=Path,
            help="Filtered JSONL destination.",
        )
        self.add_argument(
            "--reject-report",
            type=Path,
            help="JSONL audit containing row indexes and rejection reasons.",
        )
        self.add_argument("--model", type=Path, help="Hugging Face tokenizer checkpoint.")
        self.add_argument(
            "--summary-path",
            type=Path,
            default=None,
            help="Optional JSON summary destination.",
        )
        self.add_argument("--input-key", type=str, default="messages")
        self.add_argument("--tools-key", type=str, default="")
        self.add_argument("--chat-template-path", type=Path, default=None)
        self.add_argument(
            "--loss-mask-type",
            type=str,
            choices=("qwen", "qwen3", "distill_qwen"),
            default="qwen3",
        )
        self.add_argument("--max-seq-len", type=int, default=65536)


def _classify_error(error: str) -> str:
    if error == "messages must contain at least one non-empty assistant turn":
        return "no_assistant_target"
    if error.startswith("rendered length ") and " exceeds max_seq_len=" in error:
        return "over_max_seq_len"
    return "invalid_row"


def _partial_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial")


def filter_dataset(
    dataset: Path,
    *,
    output_dataset: Path,
    reject_report: Path,
    input_key: str,
    tools_key: str,
    max_seq_len: int,
    mask_generator: MultiTurnLossMaskGenerator,
) -> tuple[dict[str, object], int]:
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    reject_report.parent.mkdir(parents=True, exist_ok=True)
    output_partial = _partial_path(output_dataset)
    reject_partial = _partial_path(reject_report)

    stats: list[RowStats] = []
    schema: dict[str, set[str]] = {}
    rejection_counts: Counter[str] = Counter()
    rows_seen = 0

    with output_partial.open("w", encoding="utf-8") as output_stream, reject_partial.open(
        "w", encoding="utf-8"
    ) as reject_stream:
        for row_index, row, read_error in iter_sft_dataset(dataset):
            rows_seen += 1
            if read_error is not None:
                reason = "invalid_row"
                error_text = read_error
            else:
                assert row is not None
                for key, value in row.items():
                    schema.setdefault(key, set()).add(type(value).__name__)
                try:
                    row_stats = validate_sft_row(
                        row,
                        input_key=input_key,
                        tools_key=tools_key,
                        max_seq_len=max_seq_len,
                        mask_generator=mask_generator,
                    )
                except (AssertionError, KeyError, TypeError, ValueError) as error:
                    error_text = str(error)
                    reason = _classify_error(error_text)
                else:
                    stats.append(row_stats)
                    output_stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                    output_stream.write("\n")
                    continue

            rejection_counts[reason] += 1
            reject_stream.write(
                json.dumps(
                    {
                        "source_dataset": str(dataset.resolve()),
                        "row_index": row_index,
                        "reason": reason,
                        "error": error_text,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            reject_stream.write("\n")

    if rows_seen == 0:
        raise ValueError("dataset is empty")

    output_partial.replace(output_dataset)
    reject_partial.replace(reject_report)
    summary = summarize_sft_dataset(
        dataset,
        rows_seen=rows_seen,
        schema=schema,
        stats=stats,
        max_seq_len=max_seq_len,
        error_count=sum(rejection_counts.values()),
    )
    summary.update(
        {
            "output_dataset": str(output_dataset.resolve()),
            "reject_report": str(reject_report.resolve()),
            "rejection_reasons": dict(sorted(rejection_counts.items())),
        }
    )
    return summary, rejection_counts["invalid_row"]


def main() -> None:
    args = Arguments().parse_args()
    if args.max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if args.output_dataset.suffix not in {".jsonl", ".ndjson"}:
        raise ValueError("output_dataset must end in .jsonl or .ndjson")

    tokenizer = load_tokenizer(
        str(args.model),
        chat_template_path=(str(args.chat_template_path) if args.chat_template_path is not None else None),
        trust_remote_code=True,
    )
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)
    summary, unexpected_count = filter_dataset(
        args.dataset,
        output_dataset=args.output_dataset,
        reject_report=args.reject_report,
        input_key=args.input_key,
        tools_key=args.tools_key,
        max_seq_len=args.max_seq_len,
        mask_generator=mask_generator,
    )
    rendered_summary = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered_summary)
    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(f"{rendered_summary}\n", encoding="utf-8")
    if unexpected_count:
        print(
            f"Filtered dataset contains {unexpected_count} unexpected invalid rows; see reject report.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
