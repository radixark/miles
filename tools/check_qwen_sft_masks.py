"""Check SFT tokenization against the complete thinking-preserving chat template.

Example:
    python tools/check_qwen_sft_masks.py --data /data/sft.jsonl \
        --tokenizer /models/Qwen3.6-35B-A3B --output /data/mask-audit.json
Only numeric diagnostics and row indices are written; conversation text is never logged.
"""

import json
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

from tap import Tap

from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer


class Args(Tap):
    data: Path
    tokenizer: Path
    output: Path
    workers: int = 8


@lru_cache(maxsize=1)
def _resources(tokenizer_path: str) -> tuple[Any, MultiTurnLossMaskGenerator]:
    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=True)
    return tokenizer, MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")


def _rows(path: Path, tokenizer_path: str) -> Iterator[tuple[int, dict, str]]:
    with path.open() as stream:
        for index, line in enumerate(stream):
            if line.strip():
                yield index, json.loads(line), tokenizer_path


def _check_row(payload: tuple[int, dict, str]) -> dict:
    index, row, tokenizer_path = payload
    tokenizer, generator = _resources(tokenizer_path)
    messages = row["messages"]
    tools = row.get("tools", (row.get("metadata") or {}).get("tools"))
    reference = tokenizer.apply_chat_template(
        messages, tools=tools, preserve_thinking=True, tokenize=True, return_dict=False
    )
    tokens, mask = generator.get_loss_mask(messages, tools=tools)
    first_difference = next((i for i, (a, b) in enumerate(zip(reference, tokens, strict=False)) if a != b), None)
    target_tokens = sum(mask)
    ok = tokens == reference and len(mask) == len(tokens) and target_tokens > 0 and len(tokens) <= 262144
    return {
        "row": index,
        "ok": ok,
        "source": row.get("blend_source", "unspecified"),
        "reference_tokens": len(reference),
        "sft_tokens": len(tokens),
        "target_tokens": target_tokens,
        "first_difference": first_difference,
        "has_tools": bool(tools),
        "explicit_masks": sum("step_loss_mask" in message for message in messages),
    }


def main() -> None:
    args = Args().parse_args()
    started = time.monotonic()
    result = {
        "rows": 0,
        "accepted": 0,
        "rendered_tokens": 0,
        "target_tokens": 0,
        "max_tokens": 0,
        "rows_with_tools": 0,
        "explicit_masks": 0,
        "failures": [],
        "by_source": {},
    }
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for item in executor.map(_check_row, _rows(args.data, str(args.tokenizer)), chunksize=4):
            result["rows"] += 1
            result["accepted"] += item["ok"]
            result["rendered_tokens"] += item["reference_tokens"]
            result["target_tokens"] += item["target_tokens"]
            result["max_tokens"] = max(result["max_tokens"], item["reference_tokens"])
            result["rows_with_tools"] += item["has_tools"]
            result["explicit_masks"] += item["explicit_masks"]
            result["by_source"][item["source"]] = result["by_source"].get(item["source"], 0) + 1
            if not item["ok"]:
                result["failures"].append(item)
            if result["rows"] % 128 == 0:
                print(json.dumps({"checked": result["rows"], "accepted": result["accepted"]}), flush=True)
    result["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["accepted"] == result["rows"] and result["rows"] > 0 else 1)


if __name__ == "__main__":
    main()
