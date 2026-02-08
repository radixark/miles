#!/usr/bin/env python3
"""Build and push Miles Docker images.

Replaces the justfile with a single script that handles all build variants.

Usage:
    python docker/build.py --variant primary
    python docker/build.py --variant cu129-arm64
    python docker/build.py --variant cu13-arm64
    python docker/build.py --variant debug
    python docker/build.py --variant primary --dry-run
"""

import argparse
import os
import subprocess
from pathlib import Path

CACHE_DIR = "/tmp/miles-docker-cache"

VARIANTS = {
    "primary": {
        "image": "radixark/miles",
        "tag_postfix": "",
        "build_args": {},
        "tag_latest": True,
    },
    "cu129-arm64": {
        "image": "radixark/miles",
        "tag_postfix": "-cu129-arm64",
        "build_args": {
            "SGLANG_IMAGE_TAG": "v0.5.5.post3-cu129-arm64",
            "ENABLE_SGLANG_PATCH": "0",
        },
        "tag_latest": False,
    },
    "cu13-arm64": {
        "image": "radixark/miles",
        "tag_postfix": "-cu13-arm64",
        "build_args": {
            "SGLANG_IMAGE_TAG": "dev-arm64-cu13-20251122",
            "ENABLE_CUDA_13": "1",
            "ENABLE_SGLANG_PATCH": "0",
        },
        "tag_latest": False,
    },
    "debug": {
        "image": "radixark/miles-test",
        "tag_postfix": "",
        "build_args": {},
        "tag_latest": True,
    },
}


def get_version(repo_root: Path) -> str:
    version_file = repo_root / "docker" / "version.txt"
    return version_file.read_text().strip()


def run(cmd: list[str], dry_run: bool) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_and_push(variant: str, dry_run: bool) -> None:
    config = VARIANTS[variant]
    repo_root = Path(__file__).resolve().parent.parent

    version = get_version(repo_root)
    image = config["image"]
    image_tag = f"{version}{config['tag_postfix']}"
    full_tag = f"{image}:{image_tag}"

    cache_key = f"{CACHE_DIR}/{variant}"

    # Build command using buildx with cache and --push
    cmd = [
        "docker", "buildx", "build",
        "-f", "docker/Dockerfile",
        "--cache-from", f"type=local,src={cache_key}",
        "--cache-to", f"type=local,dest={cache_key},mode=max",
        "--push",
    ]

    # Proxy args (pass through if set in environment)
    for env_var, arg_name in [
        ("http_proxy", "HTTP_PROXY"),
        ("https_proxy", "HTTPS_PROXY"),
    ]:
        value = os.environ.get(env_var, "")
        if value:
            cmd += ["--build-arg", f"{arg_name}={value}"]

    cmd += ["--build-arg", "NO_PROXY=localhost,127.0.0.1"]

    # Variant-specific build args
    for key, value in config["build_args"].items():
        cmd += ["--build-arg", f"{key}={value}"]

    # Tags
    cmd += ["-t", full_tag]
    if config["tag_latest"]:
        cmd += ["-t", f"{image}:latest"]

    # Context is repo root
    cmd += ["."]

    print(f"\n=== Building and pushing {full_tag} ===", flush=True)
    run(cmd, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and push Miles Docker images.")
    parser.add_argument(
        "--variant",
        required=True,
        choices=list(VARIANTS.keys()),
        help="Build variant to use.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    build_and_push(args.variant, args.dry_run)


if __name__ == "__main__":
    main()
