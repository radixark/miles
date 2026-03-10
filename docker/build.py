#!/usr/bin/env python3
"""Build and push Miles Docker images.

Replaces the justfile with a single script that handles all build variants.

Usage:
    python docker/build.py --variant primary
    python docker/build.py --variant cu129-arm64
    python docker/build.py --variant cu13-arm64
    python docker/build.py --variant debug
    python docker/build.py --variant dev --push
    python docker/build.py --variant primary --dry-run
"""

import os
import subprocess
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import typer

CACHE_DIR = "/tmp/miles-docker-cache"
REPO_ROOT = Path(__file__).resolve().parent.parent

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
    "dev": {
        "image": "radixark/miles",
        "build_args": {
            "MEGATRON_COMMIT": "main",
        },
        "tag_latest": False,
    },
}


def get_version() -> str:
    version_file = REPO_ROOT / "docker" / "version.txt"
    return version_file.read_text().strip()


def run(cmd: list[str], dry_run: bool) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_and_push(variant: str, dry_run: bool, dockerfile: str, push: bool = False) -> None:
    config = VARIANTS[variant]
    image = config["image"]

    # Dev variant uses rolling + timestamped tags instead of version.txt
    if variant == "dev":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        tags = [f"{image}:dev", f"{image}:dev-{timestamp}"]
    else:
        version = get_version()
        image_tag = f"{version}{config.get('tag_postfix', '')}"
        tags = [f"{image}:{image_tag}"]
        if config.get("tag_latest"):
            tags.append(f"{image}:latest")

    cmd = [
        "docker",
        "buildx",
        "build",
        "-f",
        dockerfile,
    ]

    if push:
        cmd += ["--push"]

    # Proxy args (pass through if set in environment, check both cases)
    for arg_name in ["HTTP_PROXY", "HTTPS_PROXY"]:
        value = os.environ.get(arg_name.lower()) or os.environ.get(arg_name)
        if value:
            cmd += ["--build-arg", f"{arg_name}={value}"]

    cmd += ["--build-arg", "NO_PROXY=localhost,127.0.0.1"]

    # Variant-specific build args
    for key, value in config.get("build_args", {}).items():
        cmd += ["--build-arg", f"{key}={value}"]

    for tag in tags:
        cmd += ["-t", tag]

    # Context is repo root
    cmd += ["."]

    print(f"\n=== Building {' '.join(tags)} ===", flush=True)
    run(cmd, dry_run)


class Variant(str, Enum):
    primary = "primary"
    cu129_arm64 = "cu129-arm64"
    cu13_arm64 = "cu13-arm64"
    debug = "debug"
    dev = "dev"


def main(
    variant: Variant = typer.Option(..., help="Build variant to use."),  # noqa: B008
    dockerfile: str = typer.Option("docker/Dockerfile", help="Path to the Dockerfile."),  # noqa: B008
    dry_run: bool = typer.Option(False, help="Print commands without executing them."),  # noqa: B008
    push: bool = typer.Option(False, help="Push images to registry after building."),  # noqa: B008
) -> None:
    build_and_push(variant.value, dry_run, dockerfile, push=push)


if __name__ == "__main__":
    typer.run(main)
