"""geo3k RL data: hiyouga/geometry3k -> local PNGs + Miles jsonl (a user turn with <image> placeholders + math label)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--dataset", default="hiyouga/geometry3k")
    args = parser.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    rows, stats = [], {"samples": 0, "images": 0, "placeholders_added": 0, "skipped_no_image": 0}
    for index, row in enumerate(ds):
        if len(rows) >= args.max_samples:
            break
        images = row.get("images") or ([row["image"]] if row.get("image") is not None else [])
        if not images:
            stats["skipped_no_image"] += 1
            continue
        paths = []
        for j, image in enumerate(images):
            path = out / "images" / f"{args.split}_{index:05d}_{j}.png"
            image.convert("RGB").save(path, format="PNG")
            paths.append(str(path))
        problem = row["problem"]
        missing = len(paths) - problem.count("<image>")
        if missing > 0:
            problem = "<image>\n" * missing + problem
            stats["placeholders_added"] += missing
        rows.append(
            {
                "messages": [{"role": "user", "content": problem + "\nPut the final answer in \\boxed{}."}],
                "images": paths,
                "metadata": {"source": args.dataset, "index": index},
                "label": str(row["answer"]),
            }
        )
        stats["samples"] += 1
        stats["images"] += len(paths)
    with (out / f"{args.split}_rl.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / f"{args.split}_rl-stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
