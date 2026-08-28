"""Shared streaming validation utilities for conversational SFT datasets."""

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from miles.utils.mask_utils import MultiTurnLossMaskGenerator


@dataclass(frozen=True)
class RowStats:
    total_tokens: int
    loss_tokens: int
    response_span_tokens: int


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, object] | None, str | None]]:
    row_index = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                yield row_index, None, f"line {line_number}: invalid JSON: {error}"
            else:
                if isinstance(value, dict):
                    yield row_index, value, None
                else:
                    yield row_index, None, f"line {line_number}: row must be a JSON object"
            row_index += 1


def iter_sft_dataset(path: Path) -> Iterator[tuple[int, dict[str, object] | None, str | None]]:
    if path.suffix == ".parquet":
        frame = pl.read_parquet(path)
        for row_index, row in enumerate(frame.iter_rows(named=True)):
            yield row_index, row, None
        return
    if path.suffix in {".jsonl", ".ndjson"}:
        yield from _iter_jsonl(path)
        return
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


def validate_sft_row(
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
    return RowStats(
        total_tokens=len(token_ids),
        loss_tokens=loss_tokens,
        response_span_tokens=response_span,
    )


def _quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {name: None for name in ("min", "p50", "p90", "p95", "p99", "max")}
    series = pl.Series("value", values)
    return {
        "min": series.min(),
        "p50": series.quantile(0.50, interpolation="nearest"),
        "p90": series.quantile(0.90, interpolation="nearest"),
        "p95": series.quantile(0.95, interpolation="nearest"),
        "p99": series.quantile(0.99, interpolation="nearest"),
        "max": series.max(),
    }


def summarize_sft_dataset(
    dataset: Path,
    *,
    rows_seen: int,
    schema: dict[str, set[str]],
    stats: list[RowStats],
    max_seq_len: int,
    error_count: int,
) -> dict[str, object]:
    total_token_values = [item.total_tokens for item in stats]
    length_thresholds = (8192, 16384, 32768, 65536, 131072, 262144)
    return {
        "dataset": str(dataset.resolve()),
        "rows_seen": rows_seen,
        "validated_rows": len(stats),
        "error_count": error_count,
        "max_seq_len": max_seq_len,
        "schema": {name: sorted(types) for name, types in sorted(schema.items())},
        "rows_above_token_threshold": {
            str(threshold): sum(value > threshold for value in total_token_values) for threshold in length_thresholds
        },
        "total_tokens": _quantiles(total_token_values),
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
