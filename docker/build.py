#!/usr/bin/env python3
"""Build and push Miles Docker images.

Usage:
    python docker/build.py --variant primary --image-tag dev --push
    python docker/build.py --variant cu129-arm64 --image-tag latest
    python docker/build.py --variant primary --image-tag custom --custom-tag v1.0.0
    python docker/build.py --variant dev-glm --image-tag dev --push
    python docker/build.py --variant primary --image-tag dev --image-repo radixark/miles-staging --dry-run
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
    },
    "cu129-arm64": {
        "image": "radixark/miles",
        "tag_postfix": "-cu129-arm64",
        "build_args": {
            "SGLANG_IMAGE_TAG": "v0.5.5.post3-cu129-arm64",
            "ENABLE_SGLANG_PATCH": "0",
        },
    },
    "cu13-arm64": {
        "image": "radixark/miles",
        "tag_postfix": "-cu13-arm64",
        "build_args": {
            "SGLANG_IMAGE_TAG": "dev-arm64-cu13-20251122",
            "ENABLE_CUDA_13": "1",
            "ENABLE_SGLANG_PATCH": "0",
        },
    },
    "dev-glm": {
        "image": "radixark/miles",
        "tag_postfix": "-glm",
        "dockerfile": "docker/glm5/Dockerfile.dev-glm",
        # Layer on top of the freshly-built dev image of the same repo. BASE_IMAGE
        # is injected dynamically (see build_and_push) so staging layers on staging.
        "base_from": "dev",
        "build_args": {},
    },
    "debug": {
        "image": "radixark/miles-test",
        "tag_postfix": "",
        "build_args": {},
    },
}


def run(cmd: list[str], dry_run: bool) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_and_push(
    variant: str,
    image_tag: str,
    dry_run: bool,
    dockerfile: str,
    push: bool = False,
    custom_tag: str = "",
    image_repo: str = "",
) -> None:
    config = VARIANTS[variant]
    # --image-repo overrides the variant's default image (e.g. push to a staging repo).
    image = image_repo or config["image"]
    postfix = config.get("tag_postfix", "")
    # A variant may pin its own Dockerfile (e.g. dev-glm); otherwise use the CLI value.
    dockerfile = config.get("dockerfile", dockerfile)

    if image_tag == "latest":
        tags = [f"{image}:latest{postfix}"]
    elif image_tag == "dev":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        tags = [f"{image}:dev{postfix}", f"{image}:dev{postfix}-{timestamp}"]
    elif image_tag == "custom":
        if not custom_tag:
            raise typer.BadParameter("--custom-tag is required when --image-tag is custom")
        tags = [f"{image}:{custom_tag}{postfix}"]
    else:
        raise typer.BadParameter(f"Unknown image tag: {image_tag}")

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

    # Base image for variants that layer on another tag of the same repo (e.g. dev-glm).
    base_from = config.get("base_from")
    if base_from:
        cmd += ["--build-arg", f"BASE_IMAGE={image}:{base_from}"]

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
    dev_glm = "dev-glm"
    debug = "debug"


class ImageTag(str, Enum):
    latest = "latest"
    dev = "dev"
    custom = "custom"


def main(
    variant: Variant = typer.Option(..., help="Build variant to use."),  # noqa: B008
    image_tag: ImageTag = typer.Option(..., help="Tag mode: latest, dev, or custom."),  # noqa: B008
    dockerfile: str = typer.Option("docker/Dockerfile", help="Path to the Dockerfile."),  # noqa: B008
    dry_run: bool = typer.Option(False, help="Print commands without executing them."),  # noqa: B008
    push: bool = typer.Option(False, help="Push images to registry after building."),  # noqa: B008
    custom_tag: str = typer.Option("", help="Custom tag name (required when --image-tag is custom)."),  # noqa: B008
    image_repo: str = typer.Option("", help="Override the image repo (e.g. radixark/miles-staging)."),  # noqa: B008
) -> None:
    build_and_push(
        variant.value,
        image_tag.value,
        dry_run,
        dockerfile,
        push=push,
        custom_tag=custom_tag,
        image_repo=image_repo,
    )


if __name__ == "__main__":
    typer.run(main)
