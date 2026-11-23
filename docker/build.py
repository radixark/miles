#!/usr/bin/env python3
# doc-dev: docs/developer/ci/02-docker-build.md
"""Build and push Miles Docker images.

Usage:
    python docker/build.py --variant cu13 --image-tag dev --push          # multi-arch (amd64+arm64)
    python docker/build.py --variant cu13-x86 --image-tag dev --push      # single arch
    python docker/build.py --variant cu13 --image-tag dev --dry-run
"""

import os
import subprocess
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import image_inputs
import typer

CACHE_DIR = "/tmp/miles-docker-cache"
REPO_ROOT = Path(__file__).resolve().parent.parent

# Variants without an "image" push to the --image-repo the caller names.
VARIANTS = {
    "cu13": {
        "platforms": ["linux/amd64", "linux/arm64"],
        "tag_postfix": "",
        "build_args": {},
    },
    "cu13-x86": {
        "platforms": ["linux/amd64"],
        "tag_postfix": "",
        "build_args": {},
    },
    "cu13-aarch64": {
        "platforms": ["linux/arm64"],
        "tag_postfix": "",
        "build_args": {},
    },
    "rubin": {
        "platforms": ["linux/arm64"],
        "tag_postfix": "-rubin",
        "dockerfile": "docker/Dockerfile.rubin",
        "build_args": {},
    },
    "tpu": {
        # TPU CI (pr-test-tpu.yml) and its images belong to radixark/miles.
        "image": "radixark/miles",
        "platforms": ["linux/amd64"],
        "tag_postfix": "-tpu",
        "dockerfile": "docker/Dockerfile.tpu",
        "build_args": {},
    },
    "rocm724-mi35x": {
        "image": "rocm/sgl-dev",
        "tag_postfix": "-rocm724-mi35x",
        "tag_prefix": "miles",
        "dockerfile": "docker/Dockerfile.rocm",
        "build_args": {
            "GPU_ARCH": "gfx950",
            "SGLANG_IMAGE_REPO": "lmsysorg/sglang",
            "SGLANG_IMAGE_TAG": "v0.5.23-rocm724-mi35x",
            "WHEELS_TAG_ROCM": "rocm724-gfx950-v0.5.20",
            "APPLY_ROCR_VMMFIX": "1",
            "TE_USE_WHEEL": "1",
        },
    },
    "rocm10-mi35x": {
        "image": "rocm/sgl-dev",
        "tag_postfix": "-rocm10-mi35x",
        "tag_prefix": "miles",
        "dockerfile": "docker/Dockerfile.rocm",
        "build_args": {
            "GPU_ARCH": "gfx950",
            "SGLANG_IMAGE_REPO": "lmsysorg/sglang",
            "SGLANG_IMAGE_TAG": "v0.5.23-rocm10-mi35x",
            "WHEELS_TAG_ROCM": "rocm10-gfx950-v0.5.18",
            "APEX_USE_PREBUILT": "1",
            "TE_USE_WHEEL": "1",
        },
    },
    "rocm10-mi30x": {
        "image": "rocm/sgl-dev",
        "tag_postfix": "-rocm10-mi30x",
        "tag_prefix": "miles",
        "dockerfile": "docker/Dockerfile.rocm",
        "build_args": {
            "GPU_ARCH": "gfx942",
            "SGLANG_IMAGE_REPO": "lmsysorg/sglang",
            "SGLANG_IMAGE_TAG": "v0.5.23-rocm10-mi30x",
            "WHEELS_TAG_ROCM": "rocm10-gfx942-v0.5.20",
            "APEX_USE_PREBUILT": "1",
            "TE_USE_WHEEL": "1",
            "AITER_PREBUILD_JIT": "1",
        },
    },
}

# PR images live with Miles' CUDA/TPU PR tags; rolling ROCm tags keep their
# existing repository and native-platform behavior. The PR tag carries its own
# suffix (image_inputs.pr_image), so this variant adds none.
VARIANTS["rocm10-mi35x-pr"] = {
    **VARIANTS["rocm10-mi35x"],
    "image": "radixark/miles",
    "tag_postfix": "",
    "platforms": ["linux/amd64"],
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
    extra_build_args: list[str] | None = None,
    context: Path = REPO_ROOT,
    output: str = "",
    image_repo: str = "radixark/miles",
) -> None:
    extra_build_args = extra_build_args or []
    config = VARIANTS[variant]
    # A variant may pin its own Dockerfile (e.g. ROCm); otherwise use the CLI default.
    dockerfile = config.get("dockerfile", dockerfile)
    image = config.get("image", image_repo)
    postfix = config.get("tag_postfix", "")
    platforms = config.get("platforms")

    if image_tag == "latest":
        tags = [f"{image}:latest{postfix}"]
    elif image_tag == "dev":
        prefix = config.get("tag_prefix", "dev")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        tags = [f"{image}:{prefix}{postfix}", f"{image}:{prefix}{postfix}-{timestamp}"]
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
        str(context / dockerfile),
    ]

    if platforms:
        cmd += ["--platform", ",".join(platforms)]

    if push:
        cmd += ["--push"]
    if output:
        cmd += ["--output", output]

    # Proxy args (pass through if set in environment, check both cases)
    for arg_name in ["HTTP_PROXY", "HTTPS_PROXY"]:
        value = os.environ.get(arg_name.lower()) or os.environ.get(arg_name)
        if value:
            cmd += ["--build-arg", f"{arg_name}={value}"]

    cmd += ["--build-arg", "NO_PROXY=localhost,127.0.0.1"]

    # Authenticates the Dockerfiles' GitHub API release lookups: a shared builder IP exhausts the anonymous limit
    # (a 403). A BuildKit secret stays out of the image; never set it for an untrusted (fork) build.
    if os.environ.get("BUILD_GITHUB_TOKEN"):
        cmd += ["--secret", "id=github_token,env=BUILD_GITHUB_TOKEN"]

    # Variant-specific build args
    for key, value in config.get("build_args", {}).items():
        cmd += ["--build-arg", f"{key}={value}"]

    # Caller overrides (e.g. release builds pinning SGLANG_COMMIT / MILES_COMMIT
    # from release-lock.json) come last so they win over variant defaults.
    for spec in extra_build_args:
        assert "=" in spec, f"--build-arg expects KEY=VALUE, got {spec!r}"
        cmd += ["--build-arg", spec]

    # CI reads this back off the published tag to skip rebuilds whose inputs are unchanged.
    cmd += ["--label", f"{image_inputs.LABEL_KEY}={image_inputs.compute(root=context, variant=variant)}"]

    for tag in tags:
        cmd += ["-t", tag]

    # Context is repo root
    cmd += [str(context)]

    print(f"\n=== Building {' '.join(tags)} ===", flush=True)
    run(cmd, dry_run)


class Variant(str, Enum):
    cu13 = "cu13"
    cu13_x86 = "cu13-x86"
    cu13_aarch64 = "cu13-aarch64"
    rubin = "rubin"
    tpu = "tpu"
    rocm724_mi35x = "rocm724-mi35x"
    rocm10_mi35x = "rocm10-mi35x"
    rocm10_mi35x_pr = "rocm10-mi35x-pr"
    rocm10_mi30x = "rocm10-mi30x"


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
    build_arg: list[str] = typer.Option([], help="Extra KEY=VALUE build-arg (repeatable)."),  # noqa: B008
    context: Path = typer.Option(REPO_ROOT, help="Repository build context, separate from this driver."),  # noqa: B008
    output: str = typer.Option("", help="Buildx exporter, e.g. type=oci,dest=/tmp/image,tar=false."),  # noqa: B008
    image_repo: str = typer.Option("radixark/miles", help="Repository of the Miles images (not ROCm)."),  # noqa: B008
) -> None:
    build_and_push(
        variant.value,
        image_tag.value,
        dry_run,
        dockerfile,
        push=push,
        custom_tag=custom_tag,
        extra_build_args=build_arg,
        context=context.resolve(),
        output=output,
        image_repo=image_repo,
    )


if __name__ == "__main__":
    typer.run(main)
