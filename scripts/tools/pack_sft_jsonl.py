#!/usr/bin/env python3
"""Pack OpenAI-style SFT JSONL into pretokenized Miles packed-SFT blocks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from sglang.srt.entrypoints.openai.protocol import Tool

from miles.rollout.sft_rollout import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input SFT JSONL with messages/tools fields.")
    parser.add_argument("--output", required=True, help="Output packed JSONL with tokens/loss_mask fields.")
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    parser.add_argument("--hf-checkpoint", required=True, help="HF tokenizer/checkpoint path.")
    parser.add_argument("--input-key", default="messages")
    parser.add_argument("--tool-key", default="tools")
    parser.add_argument("--metadata-key", default="metadata")
    parser.add_argument("--loss-mask-type", default="deepseek_v4")
    parser.add_argument("--block-size", type=int, default=131072)
    parser.add_argument("--min-fill-ratio", type=float, default=0.0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--log-interval", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_size <= 1:
        raise ValueError("--block-size must be greater than 1")
    if not 0.0 <= args.min_fill_ratio <= 1.0:
        raise ValueError("--min-fill-ratio must be in [0, 1]")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report) if args.report else output.with_suffix(output.suffix + ".report.json")

    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=args.trust_remote_code)
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)

    stats = {
        "input": args.input,
        "output": str(output),
        "hf_checkpoint": args.hf_checkpoint,
        "loss_mask_type": args.loss_mask_type,
        "block_size": args.block_size,
        "min_fill_ratio": args.min_fill_ratio,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "rows_seen": 0,
        "rows_selected": 0,
        "rows_packed": 0,
        "rows_dropped": 0,
        "blocks_written": 0,
        "total_tokens": 0,
        "total_loss_tokens": 0,
        "max_sample_tokens": 0,
        "max_block_tokens": 0,
        "min_block_tokens": None,
        "tool_rows": 0,
        "started_at": time.time(),
        "dropped_examples": [],
    }

    current_tokens: list[int] = []
    current_loss_mask: list[int] = []
    current_rows: list[dict[str, Any]] = []

    def flush_block(writer, *, force: bool = False) -> None:
        if not current_tokens:
            return
        fill_ratio = len(current_tokens) / args.block_size
        if not force and fill_ratio < args.min_fill_ratio:
            return
        metadata = {
            "packed_source_rows": list(current_rows),
            "packed_num_sequences": len(current_rows),
            "packed_fill_ratio": fill_ratio,
        }
        writer.write(
            json.dumps(
                {
                    "tokens": current_tokens,
                    "response_length": len(current_tokens) - 1,
                    "loss_mask": current_loss_mask[1:],
                    "metadata": metadata,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        stats["blocks_written"] += 1
        stats["max_block_tokens"] = max(stats["max_block_tokens"], len(current_tokens))
        stats["min_block_tokens"] = (
            len(current_tokens)
            if stats["min_block_tokens"] is None
            else min(stats["min_block_tokens"], len(current_tokens))
        )
        current_tokens.clear()
        current_loss_mask.clear()
        current_rows.clear()

    with open(args.input, encoding="utf-8") as reader, open(output, "w", encoding="utf-8") as writer:
        for row_id, line in enumerate(reader):
            if args.max_rows is not None and row_id >= args.max_rows:
                break
            line = line.strip()
            if not line:
                continue

            stats["rows_seen"] += 1
            if row_id % args.num_shards != args.shard_id:
                continue
            stats["rows_selected"] += 1
            try:
                row = json.loads(line)
                messages = row[args.input_key]
                tools = row.get(args.tool_key)
                if tools:
                    tools = [Tool.model_validate(t).model_dump() for t in tools]
                    stats["tool_rows"] += 1
                tokens, loss_mask = mask_generator.get_loss_mask(messages, tools=tools)
            except Exception as exc:
                _record_drop(stats, row_id, "parse_or_tokenize_error", str(exc))
                continue

            if len(tokens) != len(loss_mask):
                _record_drop(stats, row_id, "mask_length_mismatch", f"{len(tokens)} != {len(loss_mask)}")
                continue
            if len(tokens) > args.block_size:
                _record_drop(stats, row_id, "too_long", f"{len(tokens)} > {args.block_size}")
                continue
            if 1 not in loss_mask:
                _record_drop(stats, row_id, "empty_loss_mask", "no trainable assistant tokens")
                continue

            if current_tokens and len(current_tokens) + len(tokens) > args.block_size:
                flush_block(writer, force=True)

            current_tokens.extend(tokens)
            current_loss_mask.extend(loss_mask)
            current_rows.append(
                {
                    "row_id": row_id,
                    "tokens": len(tokens),
                    "loss_tokens": sum(loss_mask),
                    "metadata": row.get(args.metadata_key) or {},
                }
            )
            stats["rows_packed"] += 1
            stats["total_tokens"] += len(tokens)
            stats["total_loss_tokens"] += sum(loss_mask)
            stats["max_sample_tokens"] = max(stats["max_sample_tokens"], len(tokens))

            if args.log_interval and stats["rows_seen"] % args.log_interval == 0:
                elapsed = max(time.time() - stats["started_at"], 1e-6)
                print(
                    "PACK shard={shard_id}/{num_shards} rows_seen={rows_seen} rows_selected={rows_selected} "
                    "rows_packed={rows_packed} blocks={blocks_written} total_tokens={total_tokens} "
                    "tok_s={tok_s:.1f}".format(
                        tok_s=stats["total_tokens"] / elapsed,
                        **stats,
                    ),
                    flush=True,
                )

        flush_block(writer, force=True)

    stats["finished_at"] = time.time()
    stats["elapsed_seconds"] = stats["finished_at"] - stats["started_at"]
    stats["avg_tokens_per_block"] = stats["total_tokens"] / max(stats["blocks_written"], 1)
    stats["avg_loss_tokens_per_block"] = stats["total_loss_tokens"] / max(stats["blocks_written"], 1)
    stats["packing_fill_ratio"] = stats["avg_tokens_per_block"] / args.block_size
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        "PACK DONE shard={shard_id}/{num_shards} rows_seen={rows_seen} rows_selected={rows_selected} "
        "rows_packed={rows_packed} rows_dropped={rows_dropped} blocks={blocks_written} "
        "avg_tokens_per_block={avg_tokens_per_block:.1f} fill={packing_fill_ratio:.4f} report={report}".format(
            report=str(report_path), **stats
        ),
        flush=True,
    )


def _record_drop(stats: dict[str, Any], row_id: int, reason: str, detail: str) -> None:
    stats["rows_dropped"] += 1
    if len(stats["dropped_examples"]) < 20:
        stats["dropped_examples"].append({"row_id": row_id, "reason": reason, "detail": detail})


if __name__ == "__main__":
    main()
