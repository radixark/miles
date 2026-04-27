#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def _convert(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing OCR prompts: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            prompt = line.strip()
            if not prompt:
                continue
            f_out.write(json.dumps({"input": prompt}, ensure_ascii=True) + "\n")
    print(f"Wrote {dst}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "flow_grpo" / "dataset" / "ocr"
    dst_dir = repo_root / "data" / "ocr"
    _convert(src_dir / "train.txt", dst_dir / "train.jsonl")
    _convert(src_dir / "test.txt", dst_dir / "test.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
