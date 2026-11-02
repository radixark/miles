"""prepare_data.py ROOT: TimeLens GRPO prompts (2.5K, gaussian-difficulty sampled) and a TimeLens-Bench eval set.

Train sampling follows TimeLens run_grpo_qwen3_8b.sh + training/data/grounding.py: the
Qwen3-VL-8B-SFT filter IoUs, 5-500 s videos, <= 200-word queries, span inside the video,
9 duration buckets x 277, weights exp(-((iou-0.05)/0.2)^2/2) times inverse IoU density, seed 42.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(sys.argv[1])
PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)


def parse_query(query: str) -> str:
    return " ".join(query.split()).strip().strip(".")


def write(rows, path):
    pq.write_table(
        pa.table(
            {
                "problem": ["<video>" + PROMPT.format(r["query"]) for r in rows],
                "videos": [[r["video"]] for r in rows],
                "label": [json.dumps({"span": r["span"]}) for r in rows],
            }
        ),
        path,
    )
    print(f"{path}: {len(rows)} rows, {len({r['video'] for r in rows})} videos")


def train_rows():
    filter_file = next((ROOT / "filter-data").glob("*/gemini_refined_data.jsonl"))
    annos = []
    for line in filter_file.open():
        a = json.loads(line)
        spans = a["span"] if isinstance(a["span"][0], list) else [a["span"]]
        if len(a["query"].split(" ")) > 200 or not 5 <= a["duration"] <= 500:
            continue
        if not any(0 <= s <= e <= a["duration"] for s, e in spans):
            continue
        annos.append({**a, "span": spans})
    rng = np.random.default_rng(42)
    ranges = [(i, i + 30) for i in range(0, 240, 30)] + [(240, float("inf"))]
    buckets = {r: [] for r in ranges}
    for idx, a in enumerate(annos):
        for lo, hi in ranges:
            if lo <= a["duration"] <= hi:
                buckets[(lo, hi)].append(idx)
                break
    kept = []
    for indices in buckets.values():
        if not indices:
            continue
        ious = np.array([annos[i]["iou"] for i in indices], dtype=np.float64)
        weights = np.exp(-0.5 * ((ious - 0.05) / 0.2) ** 2)
        counts, edges = np.histogram(ious, bins=20, range=(0, 1))
        bins = np.clip(np.digitize(ious, bins=edges), 1, 20) - 1
        weights *= (1.0 / (counts + 1e-6))[bins]
        kept += rng.choice(
            indices, size=min(int(2500 / len(ranges)), len(indices)), replace=False, p=weights / weights.sum()
        ).tolist()
    kept = set(kept)
    rows = []
    for i, a in enumerate(annos):
        if i not in kept:
            continue
        video = ROOT / a["video_path"].removeprefix("data/")
        rows.append({"query": parse_query(a["query"]), "video": str(video.resolve()), "span": list(a["span"][0])})
    missing = [r for r in rows if not Path(r["video"]).is_file()]
    assert not missing, f"{len(missing)} sampled videos are missing, e.g. {missing[0]['video']}"
    return rows


def eval_rows(name: str, limit: int):
    bench = json.load(open(ROOT / "TimeLens-Bench" / f"{name}-timelens.json"))
    rows = []
    for vid, item in bench.items():
        video = ROOT / "TimeLens-Bench" / "videos" / name / name / f"{vid}.mp4"
        for query, span in zip(item["queries"], item["spans"], strict=True):
            rows.append({"query": parse_query(query), "video": str(video), "span": list(span)})
    random.Random(0).shuffle(rows)
    return rows[:limit]


if __name__ == "__main__":
    write(train_rows(), ROOT / "timelens_grpo_train.parquet")
    write(eval_rows("charades", 600), ROOT / "timelens_bench_charades_600.parquet")
