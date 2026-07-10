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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-details", required=True, help="the run's --dump-details directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7788)
    parser.add_argument("--follow", action="store_true", help="tail telemetry streams of a still-running job")
    parser.add_argument("--tensor-lru", type=int, default=2, help="rollout steps kept resident in tensor memory")
    parser.add_argument("--cache-dir", default=None, help="summary cache dir (default: <dump>/dashboard/cache)")
    args = parser.parse_args(argv)

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
