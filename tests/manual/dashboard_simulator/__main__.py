"""Simulate a heavy miles run for the dashboard: replay a real run's dump widened to N GPUs, then time the dashboard on it.

    python -m tests.manual.dashboard_simulator generate --gpus 256 --hours 26 --out <dir> \\
        --calibration <real --dump-details> [--attempt-start-calibration <real dump with a trajectories stream>]
    python -m tests.manual.dashboard_simulator bench-static --dump-details <dir>
    python -m tests.manual.dashboard_simulator bench-follow --dump-details <copy of dir> --duration 180

``bench-follow`` appends new hour partitions to the dump it is given: run it on
a hardlink copy (``cp -al``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from tests.manual.dashboard_simulator.bench import bench_collector, bench_follow, bench_static
from tests.manual.dashboard_simulator.calibration import CalibrationRun
from tests.manual.dashboard_simulator.dumps import ActorLayout
from tests.manual.dashboard_simulator.replay import Replay

_DUMP_READER_FRESHNESS_S = 100  # DumpReader hides rollout files younger than 10 s


def generate(args: argparse.Namespace) -> None:
    calibration = CalibrationRun(args.calibration, attempt_start_source=args.attempt_start_calibration)
    calibration_gpus = len(calibration.lanes)
    assert (
        args.gpus % calibration_gpus == 0
    ), f"--gpus {args.gpus} is not a multiple of the calibration's {calibration_gpus} GPUs"
    replicas = args.gpus // calibration_gpus
    print(
        f"calibration {calibration.store.meta.run_name}: {calibration_gpus} GPUs, {len(calibration.rollout_ids)} steps, "
        f"{(calibration.end_ts - calibration.start_ts) / 3600:.1f} h -> x{replicas} = {args.gpus} GPUs for {args.hours} h",
        flush=True,
    )
    actor_layout = (
        None
        if args.no_train_dumps
        else ActorLayout(
            tp_size=args.actor_tp_size,
            pp_size=args.actor_pp_size,
            cp_size=args.actor_cp_size,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            with_ref=any(event.name == "ref_log_probs" for event in calibration.phases),
        )
    )
    if actor_layout is not None:
        actor_layout.dp_size(calibration.train_gpus * replicas)  # fail before anything is written to --out
    args.out.mkdir(parents=True)
    for name in ("tokenizer", "events"):
        if (calibration.dump_dir / name).is_dir():
            shutil.copytree(calibration.dump_dir / name, args.out / name)
    started = time.monotonic()
    replay = Replay(
        calibration,
        replicas=replicas,
        hours=args.hours,
        out=args.out,
        actor_layout=actor_layout,
        train_dumps_last_hours=args.train_dumps_last_hours,
        rollout_dumps_last_hours=args.rollout_dumps_last_hours,
    )
    replay.run()
    aged = time.time() - _DUMP_READER_FRESHNESS_S
    for path in (args.out / "rollout_data").glob("*.pt"):
        os.utime(path, (aged, aged))
    print(f"done: {replay.telemetry_rows:,} telemetry rows in {time.monotonic() - started:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--gpus", type=int, required=True)
    generate_parser.add_argument("--hours", type=float, default=26.0)
    generate_parser.add_argument("--out", type=Path, required=True)
    generate_parser.add_argument("--calibration", type=Path, required=True)
    generate_parser.add_argument("--attempt-start-calibration", type=Path, default=None)
    generate_parser.add_argument("--no-train-dumps", action="store_true")
    # the reader opens one step's dumps at a time: older steps only cost disk
    generate_parser.add_argument("--train-dumps-last-hours", type=float, default=None)
    generate_parser.add_argument("--rollout-dumps-last-hours", type=float, default=None)
    # the calibration run's actor flags: its dashboard dump does not record them
    for flag in ("--actor-tp-size", "--actor-pp-size", "--actor-cp-size", "--max-tokens-per-gpu"):
        generate_parser.add_argument(flag, type=int)
    static_parser = commands.add_parser("bench-static")
    static_parser.add_argument("--dump-details", type=Path, required=True)
    static_parser.add_argument("--port", type=int, default=7799)
    static_parser.add_argument("--repeats", type=int, default=2)
    follow_parser = commands.add_parser("bench-follow")
    follow_parser.add_argument("--dump-details", type=Path, required=True)
    follow_parser.add_argument("--port", type=int, default=7798)
    follow_parser.add_argument("--duration", type=float, default=180.0)
    collector_parser = commands.add_parser("bench-collector")
    collector_parser.add_argument("--metrics-page", type=Path, required=True, help="a real sglang /metrics response")
    collector_parser.add_argument("--engines", type=int, required=True)
    collector_parser.add_argument("--duration", type=float, default=60.0)
    collector_parser.add_argument("--push-rows-per-s", type=int, default=0)
    args = parser.parse_args()

    if args.command == "generate":
        assert not args.out.exists(), f"{args.out} exists; generate into a fresh directory"
        layout_flags = (args.actor_tp_size, args.actor_pp_size, args.actor_cp_size, args.max_tokens_per_gpu)
        assert (
            args.no_train_dumps or None not in layout_flags
        ), "train dumps need the calibration run's --actor-tp-size/--actor-pp-size/--actor-cp-size/--max-tokens-per-gpu"
        generate(args)
    elif args.command == "bench-collector":
        print(
            json.dumps(
                bench_collector(
                    args.metrics_page,
                    engines=args.engines,
                    duration_s=args.duration,
                    push_rows_per_s=args.push_rows_per_s,
                ),
                indent=1,
            )
        )
    elif args.command == "bench-static":
        print(json.dumps(bench_static(args.dump_details, port=args.port, repeats=args.repeats), indent=1))
    else:
        result = bench_follow(args.dump_details, port=args.port, duration_s=args.duration)
        print(json.dumps({key: value for key, value in result.items() if key != "cycle_detail"}, indent=1))


if __name__ == "__main__":
    main()
