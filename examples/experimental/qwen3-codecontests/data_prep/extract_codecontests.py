#!/usr/bin/env python3
"""Extract CodeContests (or any Harbor-packaged HF dataset) into task dirs.

The dataset ``open-thoughts/CodeContests`` ships parquet files with two columns:
``path`` (task id, e.g. ``code_contests-0000``) and ``task_binary`` (a gzip tar
archive of that task's Harbor directory: instruction.md, environment/Dockerfile,
tests/, ...). This script downloads the dataset and extracts every task into

    <out>/<task_id>/...

It depends ONLY on ``pyarrow`` + ``tarfile`` (+ ``huggingface_hub`` for download).
No SkyRL, no Harbor adapter. Logic mirrors the dataset's own
``extract_parquet_tasks.py`` with hardened (path-traversal-safe) extraction.

Usage:
    # download from HF then extract
    python extract_codecontests.py --dataset open-thoughts/CodeContests --out ~/harbor_tasks_cc

    # extract from a parquet you already have (offline)
    python extract_codecontests.py --parquet /path/tasks.parquet --out ~/harbor_tasks_cc

    # only the first N tasks (quick validation)
    python extract_codecontests.py --dataset open-thoughts/CodeContests --out ~/harbor_tasks_cc --limit 20
"""

from __future__ import annotations

import argparse
import io
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath


def _is_within(base: Path, target: Path) -> bool:
    try:
        return os.path.commonpath([str(base.resolve()), str(target.resolve())]) == str(
            base.resolve()
        )
    except Exception:
        return False


def _sanitize_member_name(name: str) -> str:
    parts = [p for p in PurePosixPath(name).parts if p not in ("..", ".", "", "/")]
    return str(PurePosixPath(*parts)) if parts else ""


def _safe_extract_tar(archive_bytes: bytes, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as tf:
        for member in tf.getmembers():
            name = _sanitize_member_name(member.name)
            if not name:
                continue
            if ".snapshot" in PurePosixPath(name).parts:
                continue
            target = (dest_dir / name).resolve()
            if not _is_within(dest_dir, target):
                raise RuntimeError(f"unsafe path in archive: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                with src, open(target, "wb") as dst:
                    dst.write(src.read())


def _extract_one(args: tuple) -> bool:
    rel_path, data, out_dir_str = args
    if not isinstance(rel_path, str) or not isinstance(data, (bytes, bytearray, memoryview)):
        return False
    out_dir = Path(out_dir_str)
    parts = [p for p in PurePosixPath(rel_path).parts if p not in ("..", "")]
    target = (out_dir / Path(*parts)).resolve() if parts else (out_dir / "task_unknown")
    if not _is_within(out_dir, target):
        return False
    if target.exists() and (target / "instruction.md").exists():
        return True  # idempotent: already extracted
    try:
        _safe_extract_tar(bytes(data), target)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  WARN failed to extract {rel_path}: {e}")
        return False


def _find_task_parquets(root: Path) -> list[Path]:
    import pyarrow.parquet as pq

    found = []
    for f in sorted(root.glob("**/*.parquet")):
        try:
            names = pq.read_schema(f).names
        except Exception:  # noqa: BLE001
            continue
        if "path" in names and "task_binary" in names:
            found.append(f)
    return found


def extract_parquet(parquet_path: Path, out_dir: Path, limit: int | None = None, workers: int = 8) -> int:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["path", "task_binary"])
    paths = table.column("path").to_pylist()
    data = table.column("task_binary").to_pylist()
    if limit is not None:
        paths, data = paths[:limit], data[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, d, str(out_dir)) for p, d in zip(paths, data)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_extract_one, jobs, chunksize=32))
    return sum(results)


def rewrite_dockerfiles(tasks_dir: str, base_image: str) -> int:
    """Point every task's ``environment/Dockerfile`` at a prebuilt ``base_image`` and drop
    the per-task ``apt-get install python3`` line.

    All CodeContests tasks ship the SAME Dockerfile (``FROM ubuntu:24.04`` + an apt
    python install). Harbor builds a separate image per task, so that ~40s apt fetch is
    paid on almost every sandbox spawn. If Python is baked once into ``base_image`` and
    each task Dockerfile just ``FROM``s it (no apt), the per-task build becomes ~instant
    (all heavy layers cached) and spawn collapses toward the ~5s container-create floor.

    Idempotent: a Dockerfile already pointing at ``base_image`` is left unchanged.
    Returns the number of Dockerfiles actually modified.
    """
    root = Path(os.path.expanduser(tasks_dir)).resolve()
    changed = 0
    for df in sorted(root.glob("*/environment/Dockerfile")):
        try:
            original = df.read_text()
        except Exception:  # noqa: BLE001
            continue
        out_lines = []
        for ln in original.splitlines():
            low = ln.strip().lower()
            if low.startswith("from "):
                out_lines.append(f"FROM {base_image}")
                continue
            # the expensive line — python is baked into base_image now, so drop it
            if low.startswith("run ") and "apt-get" in low and "python3" in low:
                continue
            out_lines.append(ln)
        new = "\n".join(out_lines) + "\n"
        if new != original:
            df.write_text(new)
            changed += 1
    return changed


def prepare(
    *,
    dataset: str | None,
    parquet: str | None,
    out: str,
    limit: int | None = None,
    workers: int = 8,
    base_image: str | None = None,
    rewrite_only: bool = False,
) -> str:
    out_dir = Path(os.path.expanduser(out)).resolve()

    if rewrite_only:
        if not base_image:
            raise ValueError("--rewrite-only requires --base-image")
        n = rewrite_dockerfiles(str(out_dir), base_image)
        print(f"rewrote {n} task Dockerfiles -> FROM {base_image}  (under {out_dir})")
        return str(out_dir)

    if parquet:
        pqs = [Path(os.path.expanduser(parquet))]
    else:
        if not dataset:
            raise ValueError("either --dataset or --parquet is required")
        from huggingface_hub import snapshot_download

        print(f"downloading {dataset} ...")
        snap = Path(snapshot_download(repo_id=dataset, repo_type="dataset"))
        print(f"  -> {snap}")
        pqs = _find_task_parquets(snap)
        if not pqs:
            raise RuntimeError(f"no parquet with (path, task_binary) columns found under {snap}")

    total = 0
    for p in pqs:
        if limit is not None and total >= limit:
            break
        print(f"extracting {p.name} ...")
        remaining_limit = limit - total if limit is not None else None
        total += extract_parquet(p, out_dir, limit=remaining_limit, workers=workers)
    print(f"done: {total} task dirs under {out_dir}")
    if base_image:
        n = rewrite_dockerfiles(str(out_dir), base_image)
        print(f"rewrote {n} task Dockerfiles -> FROM {base_image}")
    return str(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract Harbor CodeContests tasks (no SkyRL).")
    ap.add_argument("--dataset", default="open-thoughts/CodeContests", help="HF dataset id")
    ap.add_argument("--parquet", default=None, help="extract from a local parquet instead of downloading")
    ap.add_argument("--out", default=os.path.expanduser("~/harbor_tasks_cc"))
    ap.add_argument("--limit", type=int, default=None, help="only first N tasks")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-image", default=None,
                    help="if set, rewrite each task Dockerfile to FROM this prebuilt image "
                         "and drop the per-task apt python install (faster sandbox spawn)")
    ap.add_argument("--rewrite-only", action="store_true",
                    help="skip download/extract; only rewrite Dockerfiles in --out to use --base-image")
    args = ap.parse_args()
    prepare(
        dataset=None if args.parquet else args.dataset,
        parquet=args.parquet,
        out=args.out,
        limit=args.limit,
        workers=args.workers,
        base_image=args.base_image,
        rewrite_only=args.rewrite_only,
    )


if __name__ == "__main__":
    main()
