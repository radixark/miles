"""Validate and profile a conversational SFT dataset with the training tokenizer.

The command streams JSONL row by row (so heterogeneous nested tool schemas are
valid), reads Parquet through Polars, renders every conversation through Miles'
real SFT loss-mask implementation, rejects malformed or overlong rows, and
prints token-length quantiles used to choose the training budget.

Example:
  python scripts/tools/validate_sft_dataset.py \
    --dataset /root/datasets/train.parquet \
    --model /root/models/Qwen3.6-35B-A3B \
    --max-seq-len 65536
"""

import json
import sys
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
    model: Path
    input_key: str = "messages"
    tools_key: str = ""
    chat_template_path: Path | None = None
    loss_mask_type: Literal["qwen", "qwen3", "distill_qwen"] = "qwen3"
    max_seq_len: int = 65536
    max_errors: int = 20

    def configure(self) -> None:
        self.add_argument(
            "--dataset",
            type=Path,
            help="JSONL or Parquet dataset to validate.",
        )
        self.add_argument(
            "--model",
            type=Path,
            help="Hugging Face checkpoint containing the tokenizer.",
        )
        self.add_argument(
            "--input-key",
            type=str,
            default="messages",
            help="Column containing the conversation messages.",
        )
        self.add_argument(
            "--tools-key",
            type=str,
            default="",
            help="Optional column containing tool definitions.",
        )
        self.add_argument(
            "--chat-template-path",
            type=Path,
            default=None,
            help="Optional chat-template override.",
        )
        self.add_argument(
            "--loss-mask-type",
            type=str,
            choices=("qwen", "qwen3", "distill_qwen"),
            default="qwen3",
            help="Miles SFT loss-mask implementation.",
        )
        self.add_argument(
            "--max-seq-len",
            type=int,
            default=65536,
            help="Reject rendered rows longer than this token count.",
        )
        self.add_argument(
            "--max-errors",
            type=int,
            default=20,
            help="Maximum row-error examples to print after validating every row.",
        )


def main() -> None:
    args = Arguments().parse_args()
    if args.max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if args.max_errors <= 0:
        raise ValueError("max_errors must be positive")

    tokenizer = load_tokenizer(
        str(args.model),
        chat_template_path=(str(args.chat_template_path) if args.chat_template_path is not None else None),
        trust_remote_code=True,
    )
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)

    stats: list[RowStats] = []
    error_examples: list[str] = []
    error_count = 0
    schema: dict[str, set[str]] = {}
    rows_seen = 0
    for row_index, row, read_error in iter_sft_dataset(args.dataset):
        rows_seen += 1
        if read_error is not None:
            error_count += 1
            if len(error_examples) < args.max_errors:
                error_examples.append(f"row {row_index}: {read_error}")
            continue

        assert row is not None
        for key, value in row.items():
            schema.setdefault(key, set()).add(type(value).__name__)
        try:
            stats.append(
                validate_sft_row(
                    row,
                    input_key=args.input_key,
                    tools_key=args.tools_key,
                    max_seq_len=args.max_seq_len,
                    mask_generator=mask_generator,
                )
            )
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            error_count += 1
            if len(error_examples) < args.max_errors:
                error_examples.append(f"row {row_index}: {error}")

    if rows_seen == 0:
        raise ValueError("dataset is empty")
    print(
        json.dumps(
            summarize_sft_dataset(
                args.dataset,
                rows_seen=rows_seen,
                schema=schema,
                stats=stats,
                max_seq_len=args.max_seq_len,
                error_count=error_count,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    if error_count:
        print("SFT dataset validation failed:", file=sys.stderr)
        for error in error_examples:
            print(f"- {error}", file=sys.stderr)
        omitted_count = error_count - len(error_examples)
        if omitted_count:
            print(f"- ... {omitted_count} additional errors omitted", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
