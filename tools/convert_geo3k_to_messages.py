"""
python tools/convert_geo3k_to_messages.py --src-dir SRC_DIR --dst-dir DST_DIR

Convert geo3k (the `chenhegu/geo3k_imgurl` parquet) into structured-message jsonl with local PNGs:
each row becomes {"prompt": [{"role": "user", "content": [...]}], "label": answer}, the image part
placed where the problem's "<image>" tag sits. Writes geo3k_train.jsonl and geo3k_val.jsonl.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

_SPLITS = {"train": "geo3k_train", "test": "geo3k_val"}


def _to_messages(problem: str, image_path: Path) -> list[dict]:
    before, after = problem.split("<image>")
    content = [{"type": "text", "text": before}] if before else []
    content.append({"type": "image", "image": str(image_path)})
    if after:
        content.append({"type": "text", "text": after})
    return [{"role": "user", "content": content}]


def main(src_dir: Path, dst_dir: Path) -> None:
    (dst_dir / "images").mkdir(parents=True, exist_ok=True)
    for split, name in _SPLITS.items():
        rows = pd.read_parquet(src_dir / f"{split}.parquet")
        with open(dst_dir / f"{name}.jsonl", "w") as f:
            for i, row in enumerate(rows.itertuples()):
                images = list(row.preprocessed_images)
                assert len(images) == 1 and row.problem.count("<image>") == 1, f"{split} row {i}"
                image_path = dst_dir / "images" / f"{split}_{i:05d}.png"
                image_path.write_bytes(images[0]["bytes"])
                f.write(json.dumps({"prompt": _to_messages(row.problem, image_path), "label": row.answer}) + "\n")
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--dst-dir", type=Path, required=True)
    cli_args = parser.parse_args()
    main(cli_args.src_dir, cli_args.dst_dir)
