#!/usr/bin/env python3
# doc-dev: docs/developer/ci/02-docker-build.md
"""Content hash of everything that feeds a Miles Docker image build.

Single source of truth for common and hardware-specific repository build inputs.
CI compares this hash against the one recorded on an already-published tag
(label ``miles.image-inputs``) to decide whether a rebuild is needed;
``build.py`` stamps the label at build time.

Stdlib only: CI computes the hash on a plain hosted runner before any dependency
is installed.

Usage:
    python docker/image_inputs.py                 # hash the working tree
    python docker/image_inputs.py --rev HEAD^1    # hash a git revision
    python docker/image_inputs.py --variant rocm10-mi35x-pr --rev HEAD   # hash another variant's inputs
    python docker/image_inputs.py --pr 123 --arch arm64   # PR image variant and tag
"""

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Repository inputs only: upstream revisions and published wheels are polled by CI.
COMMON_INPUT_GLOBS = (
    ".dockerignore",
    "docker/build.py",
    "docker/image_inputs.py",
    "docker/install-kube-tools.sh",
    "requirements.txt",
)

INPUT_GLOBS_BY_HARDWARE = {
    "cuda": (
        "docker/Dockerfile",
        "docker/Dockerfile.dockerignore",
        "docker/Dockerfile.rubin",
        "docker/Dockerfile.rubin.dockerignore",
        "docker/verify_transformer_engine.py",
        "docker/patch/*",
    ),
    "rocm": (
        "docker/Dockerfile.rocm",
        "docker/Dockerfile.rocm.dockerignore",
        "docker/amd_patch/*",
        "docker/amd/*",
    ),
    "tpu": (
        "docker/Dockerfile.tpu",
        "docker/Dockerfile.tpu.dockerignore",
        "docker/tpu_patch/*",
    ),
}

HARDWARE_BY_VARIANT = {
    **dict.fromkeys(("cu13", "cu13-x86", "cu13-aarch64", "rubin"), "cuda"),
    **dict.fromkeys(("rocm724-mi35x", "rocm10-mi35x", "rocm10-mi30x", "rocm10-mi35x-pr"), "rocm"),
    "tpu": "tpu",
}

LABEL_KEY = "miles.image-inputs"

# A PR image is one tag per architecture, each stamped with LABEL_KEY so it is reused
# independently. ROCm's MI35x image is one more entry, built by pr-test-rocm.yml.
PR_IMAGE_ARCHES = {
    "amd64": ("cu13-x86", ""),
    "arm64": ("cu13-aarch64", "-arm64"),
    "rocm10-mi35x": ("rocm10-mi35x-pr", "-rocm10-mi35x"),
}


def _matches(path: str, variant: str = "cu13") -> bool:
    globs = COMMON_INPUT_GLOBS + INPUT_GLOBS_BY_HARDWARE[HARDWARE_BY_VARIANT[variant]]
    return any(fnmatch.fnmatch(path, glob) for glob in globs)


def _git(*args: str, root: Path = REPO_ROOT) -> bytes:
    return subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.PIPE).stdout


def _paths_at(rev: str | None, root: Path = REPO_ROOT, variant: str = "cu13") -> list[str]:
    if rev is None:
        listing = _git("ls-files", root=root).decode()
    else:
        listing = _git("ls-tree", "-r", "--name-only", rev, root=root).decode()
    return sorted(p for p in listing.splitlines() if _matches(p, variant))


def _content_at(path: str, rev: str | None, root: Path = REPO_ROOT) -> bytes:
    if rev is None:
        return (root / path).read_bytes()
    return _git("show", f"{rev}:{path}", root=root)


def compute(rev: str | None = None, *, root: Path = REPO_ROOT, variant: str = "cu13") -> str:
    """Hash the build inputs in the working tree (``rev=None``) or at a revision."""
    digest = hashlib.sha256()
    for path in _paths_at(rev, root, variant):
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(_content_at(path, rev, root)).digest())
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


def pr_image(arch: str, pr: int) -> tuple[str, str]:
    """Return the ``build.py`` variant and the tag of one architecture's PR image."""
    variant, suffix = PR_IMAGE_ARCHES[arch]
    return variant, f"pr-{pr}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", default=None, help="Git revision to hash (default: working tree).")
    parser.add_argument(
        "--variant", choices=HARDWARE_BY_VARIANT, default="cu13", help="Build variant whose inputs to hash."
    )
    parser.add_argument(
        "--read-label",
        action="store_true",
        help="Read imagetools inspect JSON on stdin and print the hash it recorded.",
    )
    parser.add_argument("--pr", type=int, help="Print the variant and tag of this PR's image for --arch.")
    parser.add_argument("--arch", choices=sorted(PR_IMAGE_ARCHES))
    args = parser.parse_args()
    if args.pr is not None:
        if args.arch is None:
            parser.error("--pr requires --arch")
        print(*pr_image(args.arch, args.pr))
    else:
        print(read_label(sys.stdin.read()) if args.read_label else compute(args.rev, variant=args.variant))
    return 0


if __name__ == "__main__":
    sys.exit(main())
