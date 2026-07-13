"""Standalone dashboard server over a ``--dump-details`` directory.

    python -m miles.dashboard.serve --dump-details /path/to/dump_details \\
        [--host 0.0.0.0] [--port 7788] [--follow] [--tensor-lru 2] [--cache-dir DIR]

Works on any machine that can see the directory (login node, laptop over
NFS, the training node itself). ``--follow`` tails the JSONL telemetry
streams every few seconds for quasi-live viewing of a running job; dump
files are re-discovered per request either way. Typical remote usage is
behind an SSH port-forward.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import uvicorn

from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.server import make_app
from miles.dashboard.store import MetricStore

FOLLOW_INTERVAL_SECONDS = 2.0


def make_demo_dir(target: Path) -> Path:
    """Populate ``target`` with generated demo data (dumps + telemetry).

    Uses the dummy generators from the test suite, so it requires a miles
    repo checkout (they are deliberately not shipped in the wheel)."""
    try:
        from tests.fast.dashboard.dummy_dump import dump_dummy_run
        from tests.fast.dashboard.dummy_telemetry import dump_dummy_telemetry
    except ImportError as e:
        raise SystemExit("--demo needs the dummy generators under tests/; run from a miles repo checkout") from e
    dump_dummy_run(target, steps=3, num_prompts=8, n_samples_per_prompt=4, max_response_len=48)
    dump_dummy_telemetry(target, steps=3, samples_per_step=32)
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-details", default=None, help="the run's --dump-details directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7788)
    parser.add_argument("--follow", action="store_true", help="tail telemetry streams of a still-running job")
    parser.add_argument("--tensor-lru", type=int, default=2, help="rollout steps kept resident in tensor memory")
    parser.add_argument("--cache-dir", default=None, help="summary cache dir (default: <dump>/dashboard/cache)")
    parser.add_argument("--demo", action="store_true", help="serve generated demo data (needs a repo checkout)")
    args = parser.parse_args(argv)

    if args.demo:
        import tempfile

        dump_dir = make_demo_dir(Path(tempfile.mkdtemp(prefix="miles_dashboard_demo_")))
    else:
        assert args.dump_details is not None, "--dump-details is required (or use --demo)"
        dump_dir = Path(args.dump_details)
    assert dump_dir.is_dir(), f"--dump-details directory not found: {dump_dir}"

    store = MetricStore.load(dump_dir / "dashboard")
    reader = DumpReader(dump_dir, cache_dir=args.cache_dir, tensor_lru=args.tensor_lru)
    app = make_app(store, reader, follow=args.follow)

    if args.follow:
        # Append-only streams + GIL-atomic list appends make concurrent reads
        # from request handlers safe; a reader may just miss the newest records.
        def _tail() -> None:
            while True:
                time.sleep(FOLLOW_INTERVAL_SECONDS)
                store.follow()

        threading.Thread(target=_tail, daemon=True, name="dashboard-follow").start()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
