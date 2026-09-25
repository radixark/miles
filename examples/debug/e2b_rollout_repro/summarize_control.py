"""Summarize measured payload blocks without publishing task or sandbox contents."""

import json
import statistics
from pathlib import Path

from tap import Tap


class Args(Tap):
    root: str


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def summarize(block: dict) -> dict:
    rows = block["rows"]
    metrics = block["metrics"]
    result = {"arm": block["arm"], "block": block["block"], "count": len(rows),
              "wall_s": block["duration_s"], "server_cpu_s": block["server_cpu_s"],
              "request_errors": sum("error" in row for row in rows),
              "sandbox_errors": sum(row.get("sandbox_ok") is False for row in rows),
              "sandbox_successes": sum(row.get("sandbox_ok") is True for row in rows),
              "server_loop_max_s": max((max(m["lag_s"], default=0) for m in metrics), default=0),
              "driver_loop_max_s": max(block["driver_metrics"]["lag_s"], default=0),
              "sum_worker_peak_gib": sum(m["rss_peak_bytes"] for m in metrics) / 1024**3,
              "sum_worker_initial_gib": sum(block["initial_rss"]) / 1024**3}
    for key in ["chat_s", "samples_s", "sandbox_s", "reply_bytes", "sample_bytes"]:
        values = [row[key] for row in rows if key in row]
        result[key] = {"median": statistics.median(values) if values else None,
                       "p95": percentile(values, 0.95), "max": max(values, default=None)}
    result["errors"] = [row for row in rows if "error" in row or row.get("sandbox_ok") is False][:5]
    return result


def main() -> None:
    args = Args().parse_args()
    root = Path(args.root)
    blocks = [json.loads(path.read_text()) for path in sorted(root.glob("result-*.json"))]
    rows = [row for block in blocks for row in block["rows"]]
    result = {"blocks": [summarize(block) for block in blocks],
              "training_hash_count": len({row["training_hash"] for row in rows if "training_hash" in row}),
              "candidate_hash_count": len({row["candidate_hash"] for row in rows if "candidate_hash" in row})}
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
