#!/usr/bin/env python3
# doc-dev: docs/ci/02-docker-build.md
"""Content hash of everything that feeds a Miles Docker image build.

Single source of truth for "what changes the image". CI compares this hash against
the one recorded on an already-published tag (label ``miles.image-inputs``) to decide
whether a rebuild is needed; ``build.py`` stamps the label at build time.

Stdlib only: CI computes the hash on a plain hosted runner before any dependency
is installed.

Usage:
    python docker/image_inputs.py                 # hash the working tree
    python docker/image_inputs.py --rev HEAD^1    # hash a git revision
"""

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# A build reads exactly these. Anything outside them cannot change the image, so a
# PR touching only such files reuses the released image. Dockerfile.rocm is absent on
# purpose: it feeds pr-test-rocm.yml, not the cu13 multi-arch image built here.
INPUT_GLOBS = (
    "docker/Dockerfile",
    "docker/build.py",
    "docker/install-kube-tools.sh",
    "docker/verify_transformer_engine.py",
    "docker/patch/*",
    "requirements.txt",
)

LABEL_KEY = "miles.image-inputs"


def _matches(path: str) -> bool:
    return any(fnmatch.fnmatch(path, glob) for glob in INPUT_GLOBS)


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE).stdout


def _paths_at(rev: str | None) -> list[str]:
    if rev is None:
        listing = _git("ls-files").decode()
    else:
        listing = _git("ls-tree", "-r", "--name-only", rev).decode()
    return sorted(p for p in listing.splitlines() if _matches(p))


def _content_at(path: str, rev: str | None) -> bytes:
    if rev is None:
        return (REPO_ROOT / path).read_bytes()
    return _git("show", f"{rev}:{path}")


def compute(rev: str | None = None) -> str:
    """Hash the build inputs in the working tree (``rev=None``) or at a revision."""
    digest = hashlib.sha256()
    for path in _paths_at(rev):
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(_content_at(path, rev)).digest())
    return digest.hexdigest()


def read_label(manifest: str) -> str:
    """Pull the recorded hash out of ``docker buildx imagetools inspect`` JSON.

    The shape differs between single- and multi-platform tags, so search rather than
    index; an unpublished tag yields empty input and no hash.
    """

    def search(node: object) -> str:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == LABEL_KEY and isinstance(value, str):
                    return value
                if found := search(value):
                    return found
        elif isinstance(node, list):
            for item in node:
                if found := search(item):
                    return found
        return ""

    manifest = manifest.strip()
    if not manifest:
        return ""
    try:
        return search(json.loads(manifest))
    except json.JSONDecodeError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", default=None, help="Git revision to hash (default: working tree).")
    parser.add_argument(
        "--read-label",
        action="store_true",
        help="Read imagetools inspect JSON on stdin and print the hash it recorded.",
    )
    args = parser.parse_args()
    print(read_label(sys.stdin.read()) if args.read_label else compute(args.rev))
    return 0


if __name__ == "__main__":
    sys.exit(main())
