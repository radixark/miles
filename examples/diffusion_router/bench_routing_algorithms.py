#!/usr/bin/env python3
"""
Compare routing algorithms by running bench_router.py for each algorithm
and collecting results into a summary table and CSV.

Example:
  python examples/diffusion_router/bench_routing_algorithms.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --num-workers 2 \
    --num-prompts 10 \
    --max-concurrency 2
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ALL_ALGORITHMS = ["least-request", "round-robin", "random"]


def _require_non_empty_model(model: str) -> str:
    normalized = model.strip()
    if not normalized:
        raise ValueError(
            "--model must be a non-empty model ID/path. "
            "Detected an empty value, which often means a shell variable such as "
            "$MODEL was unset."
        )
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare routing algorithms by running bench_router.py for each."
    )
    parser.add_argument("--model", type=str, required=True, help="Diffusion model HF ID or local path.")
    parser.add_argument(
        "--algorithms", nargs="+", default=ALL_ALGORITHMS,
        choices=ALL_ALGORITHMS, help="Algorithms to compare (default: all three).",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results.")

    # Pass-through args for bench_router.py
    parser.add_argument("--router-host", type=str, default="127.0.0.1")
    parser.add_argument("--router-port", type=int, default=30080)
    parser.add_argument("--router-verbose", action="store_true")
    parser.add_argument("--router-extra-args", type=str, default="")
    parser.add_argument("--worker-host", type=str, default="127.0.0.1")
    parser.add_argument("--worker-urls", nargs="*", default=[])
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-base-port", type=int, default=10090)
    parser.add_argument("--worker-port-stride", type=int, default=2)
    parser.add_argument("--worker-master-port-base", type=int, default=30005)
    parser.add_argument("--worker-scheduler-port-base", type=int, default=5555)
    parser.add_argument("--worker-internal-port-stride", type=int, default=1000)
    parser.add_argument("--num-gpus-per-worker", type=int, default=1)
    parser.add_argument("--worker-gpu-ids", nargs="*", default=None)
    parser.add_argument("--worker-extra-args", type=str, default="")
    parser.add_argument("--skip-workers", action="store_true")
    parser.add_argument("--dataset", type=str, default="random", choices=["vbench", "random"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--bench-extra-args", type=str, default="")
    parser.add_argument("--wait-timeout", type=int, default=1200)

    args = parser.parse_args()
    args.model = _require_non_empty_model(args.model)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs") / f"routing_algo_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    results: dict[str, dict] = {}

    for algo in args.algorithms:
        print(f"\n{'='*60}", flush=True)
        print(f"  Running benchmark with routing algorithm: {algo}", flush=True)
        print(f"{'='*60}\n", flush=True)

        out_file = output_dir / f"bench_{algo}.json"

        bench_cmd = [
            py, "examples/diffusion_router/bench_router.py",
            "--model", args.model,
            "--routing-algorithm", algo,
            "--num-workers", str(args.num_workers),
            "--num-prompts", str(args.num_prompts),
            "--max-concurrency", str(args.max_concurrency),
            "--num-gpus-per-worker", str(args.num_gpus_per_worker),
            "--worker-base-port", str(args.worker_base_port),
            "--worker-port-stride", str(args.worker_port_stride),
            "--worker-master-port-base", str(args.worker_master_port_base),
            "--worker-scheduler-port-base", str(args.worker_scheduler_port_base),
            "--worker-internal-port-stride", str(args.worker_internal_port_stride),
            "--router-host", args.router_host,
            "--router-port", str(args.router_port),
            "--dataset", args.dataset,
            "--request-rate", str(args.request_rate),
            "--wait-timeout", str(args.wait_timeout),
            "--log-level", args.log_level,
            "--output-file", str(out_file),
        ]
        if args.worker_urls:
            bench_cmd += ["--worker-urls", *args.worker_urls]
        if args.worker_gpu_ids:
            bench_cmd += ["--worker-gpu-ids", *args.worker_gpu_ids]
        if args.dataset_path:
            bench_cmd += ["--dataset-path", args.dataset_path]
        if args.task:
            bench_cmd += ["--task", args.task]
        if args.width:
            bench_cmd += ["--width", str(args.width)]
        if args.height:
            bench_cmd += ["--height", str(args.height)]
        if args.num_frames:
            bench_cmd += ["--num-frames", str(args.num_frames)]
        if args.fps:
            bench_cmd += ["--fps", str(args.fps)]
        if args.disable_tqdm:
            bench_cmd.append("--disable-tqdm")
        if args.router_verbose:
            bench_cmd.append("--router-verbose")
        if args.skip_workers:
            bench_cmd.append("--skip-workers")
        if args.worker_extra_args:
            bench_cmd += ["--worker-extra-args", args.worker_extra_args]
        if args.router_extra_args:
            bench_cmd += ["--router-extra-args", args.router_extra_args]
        if args.bench_extra_args:
            bench_cmd += ["--bench-extra-args", args.bench_extra_args]
        if args.worker_host != "127.0.0.1":
            bench_cmd += ["--worker-host", args.worker_host]

        print("[run]", " ".join(shlex.quote(x) for x in bench_cmd), flush=True)
        rc = subprocess.call(bench_cmd)

        if rc != 0:
            print(f"[warn] bench_router.py exited with code {rc} for algorithm '{algo}'", flush=True)
            results[algo] = {"error": f"exit_code={rc}"}
            continue

        if out_file.exists():
            try:
                results[algo] = json.loads(out_file.read_text())
            except json.JSONDecodeError as e:
                print(f"[warn] Failed to parse {out_file}: {e}", flush=True)
                results[algo] = {"error": f"json_parse_error: {e}"}
        else:
            print(f"[warn] Output file not found: {out_file}", flush=True)
            results[algo] = {"error": "output_file_missing"}

    BASELINE = "random"
    metric_keys = ["throughput_qps", "latency_mean", "latency_median", "latency_p99"]

    csv_rows: list[dict] = []
    parsed: dict[str, dict] = {}
    for algo in args.algorithms:
        data = results.get(algo, {})
        if "error" in data:
            parsed[algo] = None
            csv_rows.append({
                "algorithm": algo, "throughput_qps": "", "latency_mean": "",
                "latency_median": "", "latency_p99": "", "duration": "",
                "completed_requests": "", "failed_requests": "",
                "throughput_qps_delta_pct": "", "latency_mean_delta_pct": "",
                "latency_median_delta_pct": "", "latency_p99_delta_pct": "",
                "error": data["error"],
            })
            continue

        row = {
            "algorithm": algo,
            "throughput_qps": data.get("throughput_qps", ""),
            "latency_mean": data.get("latency_mean", ""),
            "latency_median": data.get("latency_median", ""),
            "latency_p99": data.get("latency_p99", ""),
            "duration": data.get("duration", ""),
            "completed_requests": data.get("completed_requests", ""),
            "failed_requests": data.get("failed_requests", ""),
            "error": "",
        }
        parsed[algo] = row
        csv_rows.append(row)

    baseline_row = parsed.get(BASELINE)
    for row in csv_rows:
        if row.get("error"):
            continue
        for key in metric_keys:
            val = row.get(key, "")
            base = baseline_row.get(key, "") if baseline_row else ""
            delta_key = f"{key}_delta_pct"
            if isinstance(val, (int, float)) and isinstance(base, (int, float)) and base:
                row[delta_key] = ((val - base) / abs(base)) * 100
            else:
                row[delta_key] = ""

    print(f"\n{'='*100}", flush=True)
    print(f"  Routing Algorithm Comparison  (baseline: {BASELINE})", flush=True)
    print(f"{'='*100}", flush=True)

    header = (
        f"{'Algorithm':<16} {'Throughput':>14} {'Tput Delta':>11}"
        f" {'Mean Lat':>12} {'Delta':>8}"
        f" {'Median Lat':>12} {'Delta':>8}"
        f" {'P99 Lat':>12} {'Delta':>8}"
        f" {'Done':>6} {'Fail':>6}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    def _fmt_qps(v):
        return f"{v:.2f} req/s" if isinstance(v, (int, float)) else str(v)

    def _fmt_lat(v):
        return f"{v:.3f} s" if isinstance(v, (int, float)) else str(v)

    def _fmt_delta(v):
        if isinstance(v, (int, float)):
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.1f}%"
        return ""

    def _fmt_int(v):
        return str(v) if v != "" else "N/A"

    for row in csv_rows:
        if row.get("error"):
            print(f"{row['algorithm']:<16} {'ERROR':>14} {row['error']}", flush=True)
            continue
        print(
            f"{row['algorithm']:<16}"
            f" {_fmt_qps(row['throughput_qps']):>14} {_fmt_delta(row.get('throughput_qps_delta_pct', '')):>11}"
            f" {_fmt_lat(row['latency_mean']):>12} {_fmt_delta(row.get('latency_mean_delta_pct', '')):>8}"
            f" {_fmt_lat(row['latency_median']):>12} {_fmt_delta(row.get('latency_median_delta_pct', '')):>8}"
            f" {_fmt_lat(row['latency_p99']):>12} {_fmt_delta(row.get('latency_p99_delta_pct', '')):>8}"
            f" {_fmt_int(row['completed_requests']):>6} {_fmt_int(row['failed_requests']):>6}",
            flush=True,
        )

    csv_path = output_dir / "routing_algorithm_comparison.csv"
    fieldnames = [
        "algorithm", "throughput_qps", "throughput_qps_delta_pct",
        "latency_mean", "latency_mean_delta_pct",
        "latency_median", "latency_median_delta_pct",
        "latency_p99", "latency_p99_delta_pct",
        "duration", "completed_requests", "failed_requests", "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[done] CSV written to {csv_path}", flush=True)
    print(f"[done] Per-algorithm JSON results in {output_dir}/", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
