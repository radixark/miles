"""Validate and profile a conversational SFT dataset with the training tokenizer.

The command reads JSONL or Parquet through Polars, renders every conversation
through Miles' real SFT loss-mask implementation, rejects malformed or overlong
rows, and prints token-length quantiles used to choose the training budget.

Example:
  python scripts/tools/validate_sft_dataset.py \
    --dataset /root/datasets/train.parquet \
    --model /root/models/Qwen3.6-35B-A3B \
    --max-seq-len 65536
"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import polars as pl
from tap import Tap

from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer


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
            help="Maximum row errors to print before stopping.",
        )


@dataclass(frozen=True)
class RowStats:
    total_tokens: int
    loss_tokens: int
    response_span_tokens: int


def _read_dataset(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix in {".jsonl", ".ndjson"}:
        return pl.read_ndjson(path)
    raise ValueError(f"Unsupported dataset extension {path.suffix!r}; expected .jsonl, .ndjson, or .parquet")


def _decode_json_value(value: object, *, field: str) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} is a string but not valid JSON: {error}") from error


def _validate_messages(value: object) -> list[dict[str, object]]:
    value = _decode_json_value(value, field="messages")
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty list")

    messages: list[dict[str, object]] = []
    has_nonempty_assistant = False
    for message_index, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{message_index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{message_index}].role must be a non-empty string")
        if not isinstance(content, (str, list)):
            raise ValueError(f"messages[{message_index}].content must be a string or content-block list")
        if role == "assistant" and bool(content):
            has_nonempty_assistant = True
        messages.append(message)

    if not has_nonempty_assistant:
        raise ValueError("messages must contain at least one non-empty assistant turn")
    return messages


def _validate_tools(value: object) -> list[dict[str, object]] | None:
    if value is None:
        return None
    value = _decode_json_value(value, field="tools")
    if not isinstance(value, list):
        raise ValueError("tools must be a list when present")
    if not all(isinstance(tool, dict) for tool in value):
        raise ValueError("every tools entry must be an object")
    return value


def _validate_row(
    row: dict[str, object],
    *,
    input_key: str,
    tools_key: str,
    max_seq_len: int,
    mask_generator: MultiTurnLossMaskGenerator,
) -> RowStats:
    if input_key not in row:
        raise ValueError(f"missing input column {input_key!r}")
    messages = _validate_messages(row[input_key])
    tools = _validate_tools(row.get(tools_key)) if tools_key else None
    token_ids, loss_mask = mask_generator.get_loss_mask(messages, tools=tools)
    if len(token_ids) != len(loss_mask):
        raise ValueError(f"token/mask length mismatch: {len(token_ids)} != {len(loss_mask)}")
    if len(token_ids) > max_seq_len:
        raise ValueError(f"rendered length {len(token_ids)} exceeds max_seq_len={max_seq_len}")
    loss_tokens = sum(loss_mask)
    if loss_tokens == 0:
        raise ValueError("conversation produces zero assistant loss tokens")
    response_span = mask_generator.get_response_lengths([loss_mask])[0]
    return RowStats(total_tokens=len(token_ids), loss_tokens=loss_tokens, response_span_tokens=response_span)


def _quantiles(values: list[int]) -> dict[str, float | int]:
    series = pl.Series("value", values)
    return {
        "min": series.min(),
        "p50": series.quantile(0.50, interpolation="nearest"),
        "p90": series.quantile(0.90, interpolation="nearest"),
        "p95": series.quantile(0.95, interpolation="nearest"),
        "p99": series.quantile(0.99, interpolation="nearest"),
        "max": series.max(),
    }


def _summary(dataset: Path, frame: pl.DataFrame, stats: list[RowStats], max_seq_len: int) -> dict[str, object]:
    return {
        "dataset": str(dataset.resolve()),
        "rows": frame.height,
        "max_seq_len": max_seq_len,
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        "total_tokens": _quantiles([item.total_tokens for item in stats]),
        "loss_tokens": _quantiles([item.loss_tokens for item in stats]),
        "response_span_tokens": _quantiles([item.response_span_tokens for item in stats]),
        "totals": asdict(
            RowStats(
                total_tokens=sum(item.total_tokens for item in stats),
                loss_tokens=sum(item.loss_tokens for item in stats),
                response_span_tokens=sum(item.response_span_tokens for item in stats),
            )
        ),
    }


def main() -> None:
    args = Arguments().parse_args()
    if args.max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if args.max_errors <= 0:
        raise ValueError("max_errors must be positive")

    frame = _read_dataset(args.dataset)
    if frame.is_empty():
        raise ValueError("dataset is empty")
    tokenizer = load_tokenizer(
        str(args.model),
        chat_template_path=str(args.chat_template_path) if args.chat_template_path is not None else None,
        trust_remote_code=True,
    )
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)

    stats: list[RowStats] = []
    errors: list[str] = []
    for row_index, row in enumerate(frame.iter_rows(named=True)):
        try:
            stats.append(
                _validate_row(
                    row,
                    input_key=args.input_key,
                    tools_key=args.tools_key,
                    max_seq_len=args.max_seq_len,
                    mask_generator=mask_generator,
                )
            )
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            errors.append(f"row {row_index}: {error}")
            if len(errors) >= args.max_errors:
                break

    if errors:
        print("SFT dataset validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(_summary(args.dataset, frame, stats, args.max_seq_len), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
