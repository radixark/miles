#!/usr/bin/env python3
# doc-dev: docs/ci/02-docker-build.md
"""Build and push Miles Docker images.

Usage:
    python docker/build.py --variant cu13 --image-tag dev --push          # multi-arch (amd64+arm64)
    python docker/build.py --variant cu13-x86 --image-tag dev --push      # single arch
    python docker/build.py --variant cu12-x86 --image-tag latest
    python docker/build.py --variant cu13 --image-tag dev --dry-run
"""

import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import resolve_upstream

CACHE_DIR = "/tmp/miles-docker-cache"
REPO_ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    "cu13": {
        "image": "radixark/miles",
        "platforms": ["linux/amd64", "linux/arm64"],
        "tag_postfix": "",
        "build_args": {},
    },
    "cu13-x86": {
        "image": "radixark/miles",
        "platforms": ["linux/amd64"],
        "tag_postfix": "",
        "build_args": {},
    },
    "cu13-aarch64": {
        "image": "radixark/miles",
        "platforms": ["linux/arm64"],
        "tag_postfix": "",
        "build_args": {},
    },
    "cu12-x86": {
        "image": "radixark/miles",
        "platforms": ["linux/amd64"],
        "tag_postfix": "-cu12",
        "build_args": {
            "ENABLE_CUDA_13": "0",
            "SGLANG_IMAGE_TAG": "v0.5.16-cu129",
            "WHEELS_TAG_X86": "cu129-x86_64",
        },
    },
    "rocm700-mi35x": {
        "image": "rocm/sgl-dev",
        "tag_postfix": "-rocm700-mi35x",
        "tag_prefix": "miles",
        "dockerfile": "docker/Dockerfile.rocm",
        "build_args": {
            "GPU_ARCH": "gfx950",
            "SGLANG_IMAGE_REPO": "rocm/sgl-dev",
            "SGLANG_IMAGE_TAG": "v0.5.14-rocm700-mi35x-20260627",
            "SGLANG_USE_ROCM700A": "1",
        },
    },
    "rocm700-mi30x": {
        "image": "rocm/sgl-dev",
        "tag_postfix": "-rocm700-mi30x",
        "tag_prefix": "miles",
        "dockerfile": "docker/Dockerfile.rocm",
        "build_args": {
            "GPU_ARCH": "gfx942",
            "SGLANG_IMAGE_TAG": "v0.5.10-rocm700-mi30x",
            "SGLANG_USE_ROCM700A": "1",
        },
    },
    "rocm720-mi35x": {
        "image": "rocm/sgl-dev",
        "tag_postfix": "-rocm720-mi35x",
        "tag_prefix": "miles",
        "dockerfile": "docker/Dockerfile.rocm",
        "build_args": {
            "GPU_ARCH": "gfx950",
            "SGLANG_IMAGE_REPO": "rocm/sgl-dev",
            "SGLANG_IMAGE_TAG": "v0.5.16-rocm720-mi35x-20260730",
            "APPLY_ROCR_VMMFIX": "1",
            "TE_USE_WHEEL": "1",
        },
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
    load: bool = False,
    custom_tag: str = "",
    build_args: list[str] | None = None,
    builder: str = "",
) -> None:
    config = VARIANTS[variant]
    # A variant may pin its own Dockerfile (e.g. ROCm); otherwise use the CLI default.
    dockerfile = config.get("dockerfile", dockerfile)
    image = config["image"]
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
            raise SystemExit("--custom-tag is required when --image-tag is custom")
        tags = [f"{image}:{custom_tag}{postfix}"]
    else:
        raise SystemExit(f"Unknown image tag: {image_tag}")

    cmd = [
        "docker",
        "buildx",
        "build",
        "-f",
        dockerfile,
    ]

    # Explicit builder selection (CI passes its persistent node-local builder);
    # stateful `docker buildx use` defaults stay out of the picture.
    if builder:
        cmd += ["--builder", builder]

    if platforms:
        cmd += ["--platform", ",".join(platforms)]

    if push:
        cmd += ["--push"]
    if load:
        if platforms and len(platforms) > 1:
            raise SystemExit("--load requires a single-platform variant (buildx cannot load a manifest list)")
        cmd += ["--load"]

    # Proxy args (pass through if set in environment, check both cases)
    for arg_name in ["HTTP_PROXY", "HTTPS_PROXY"]:
        value = os.environ.get(arg_name.lower()) or os.environ.get(arg_name)
        if value:
            cmd += ["--build-arg", f"{arg_name}={value}"]

    cmd += ["--build-arg", "NO_PROXY=localhost,127.0.0.1"]

    # Variant-specific build args
    for key, value in config.get("build_args", {}).items():
        cmd += ["--build-arg", f"{key}={value}"]

    # Caller-resolved extras (e.g. CI-resolved upstream SHAs / wheels fingerprints).
    # Appended last so an explicit passthrough wins over a variant default.
    for arg in build_args or []:
        if "=" not in arg or not arg.split("=", 1)[0]:
            raise SystemExit(f"--build-arg expects KEY=VALUE, got {arg!r}")
        cmd += ["--build-arg", arg]

    # Pinned inputs docker/Dockerfile refuses to build without (see its ARG block).
    # CI passes the source SHAs it gated on; whatever the caller didn't pass — always
    # the per-variant wheels fingerprints, everything on a local build — is resolved
    # here, so one command still builds and every pin is printed.
    if dockerfile == "docker/Dockerfile":
        provided = set(config.get("build_args", {}))
        provided |= {arg.split("=", 1)[0] for arg in build_args or []}
        for arg_name, (repo, branch) in resolve_upstream.SOURCE_PINS.items():
            if arg_name in provided:
                continue
            sha = resolve_upstream.git_branch_head(repo, branch)
            print(f"resolved {arg_name}={sha} ({repo}@{branch})", flush=True)
            cmd += ["--build-arg", f"{arg_name}={sha}"]
        # Tag defaults mirror the Dockerfile's WHEELS_TAG_* ARGs; variants override via build_args.
        wheels_fp_args = {
            "linux/amd64": ("WHEELS_FP_X86", config.get("build_args", {}).get("WHEELS_TAG_X86", "cu130-x86_64")),
            "linux/arm64": ("WHEELS_FP_ARM64", config.get("build_args", {}).get("WHEELS_TAG_ARM64", "cu130-aarch64")),
        }
        for platform in platforms or []:
            arg_name, tag = wheels_fp_args[platform]
            if arg_name in provided:
                continue
            fp = resolve_upstream.wheels_fingerprint(tag)
            print(f"resolved {arg_name}={fp} (wheels release {tag})", flush=True)
            cmd += ["--build-arg", f"{arg_name}={fp}"]

    for tag in tags:
        cmd += ["-t", tag]

    # Context is repo root
    cmd += ["."]

    print(f"\n=== Building {' '.join(tags)} ===", flush=True)
    run(cmd, dry_run)


# stdlib-only on purpose: this runs on bare build nodes (docker + stock python3),
# so it must not require pip-installing anything on the host.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build and push Miles Docker images.")
    parser.add_argument("--variant", required=True, choices=list(VARIANTS), help="Build variant to use.")
    parser.add_argument("--image-tag", required=True, choices=["latest", "dev", "custom"], help="Tag mode.")
    parser.add_argument("--dockerfile", default="docker/Dockerfile", help="Path to the Dockerfile.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--push", action="store_true", help="Push images to registry after building.")
    parser.add_argument("--custom-tag", default="", help="Custom tag name (required when --image-tag is custom).")
    parser.add_argument(
        "--build-arg",
        dest="build_args",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra KEY=VALUE forwarded to docker buildx build (repeatable).",
    )
    parser.add_argument("--builder", default="", help="buildx builder to use (default: current).")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load the image into the local docker store (single-platform variants only).",
    )
    args = parser.parse_args()
    build_and_push(
        args.variant,
        args.image_tag,
        args.dry_run,
        args.dockerfile,
        push=args.push,
        load=args.load,
        custom_tag=args.custom_tag,
        build_args=args.build_args,
        builder=args.builder,
    )


if __name__ == "__main__":
    main()
