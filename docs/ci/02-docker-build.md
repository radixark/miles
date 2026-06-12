---

## title: Docker build
description: The Dockerfiles, the build script, the remote build workflow, and how to build & push manually.

# Docker build

CI runs inside `radixark/miles`. This doc maps which Dockerfiles exist, the script that builds them, how the remote build is triggered, and how to build & push manually.

## Dockerfiles


| Path | Builds | Wired into |
| ---- | ------ | ---------- |
| `docker/Dockerfile`      | `radixark/miles` — the CUDA image (x86 + arm64) | `docker-build.yml` |
| `docker/Dockerfile.rocm` | AMD ROCm (MI30x / MI35x)                        | `docker-build.yml` (`rocm-*` variants) |

`docker/Dockerfile` is multi-stage: `FROM lmsysorg/sglang:${SGLANG_IMAGE_TAG}` (a multi-arch `lmsysorg/sglang` release), installs sglang from the `sglang-miles` branch, then layers Megatron-LM (`radixark/Megatron-LM@miles-main`), miles, and prebuilt wheels from `yueming-yuan/miles-wheels` (sgl-router among them) — selected per-arch from `TARGETARCH`, so one build serves both x86 and arm64. `docker/Dockerfile.rocm` builds the AMD variants.

## Build script

`docker/build.py` builds and pushes the images. Select a build with `--variant` and a tag mode with `--image-tag {dev,latest,custom}`. A single `VARIANTS` table is the source of truth for each variant's image, target platforms, Dockerfile, and build-args.

| `--variant` | Tag (`--image-tag dev`) | Platforms | Notes |
|---|---|---|---|
| `cu13`         | `radixark/miles:dev`               | `linux/amd64` + `linux/arm64` | **multi-arch**, one manifest — the daily image |
| `cu13-x86`     | `radixark/miles:dev`               | `linux/amd64`                 | x86-only build of the same image |
| `cu13-aarch64` | `radixark/miles:dev`               | `linux/arm64`                 | arm64-only build of the same image |
| `cu12-x86`     | `radixark/miles:dev-cu12`          | `linux/amd64`                 | CUDA 12.9 legacy |
| `rocm-mi300`   | `rocm/sgl-dev:miles-rocm700-mi30x` | native                        | AMD MI30x — `docker/Dockerfile.rocm` |
| `rocm-mi350`   | `rocm/sgl-dev:miles-rocm720-mi35x` | native                        | AMD MI35x — `docker/Dockerfile.rocm` |

The cu13 variants share one CUDA base (`lmsysorg/sglang:v0.5.12`, multi-arch) and differ only in platforms. `cu13` runs a single `buildx --platform linux/amd64,linux/arm64`, picking each layer's wheels inside the Dockerfile from `TARGETARCH` (amd64→`x86_64`, arm64→`aarch64`), and pushes one manifest so `docker pull` auto-selects by host arch.

The **Tag** column is for `--image-tag dev`, which also pushes a timestamped `dev-<YYYYMMDDHHMM>` sibling; `latest` swaps the prefix to `latest`, `custom` uses `--custom-tag`. `cu13` / `cu13-x86` / `cu13-aarch64` intentionally share `radixark/miles:dev` — the daily build runs `cu13` (multi-arch), while a single-arch variant overwrites `dev` with one arch when run alone.

A multi-arch build (`cu13`) needs Buildx's `docker-container` driver and is push-only — it can't load into the local image store; use `cu13-x86` / `cu13-aarch64` for local single-arch iteration. Other flags: `--push`, `--dry-run`, `--dockerfile`, `--custom-tag`.

## Remote docker build (`docker-build.yml`)

The only automated builder of `radixark/miles`. Two jobs:

- **`check-upstream`** (schedule / `simulate_schedule` only) — polls the inputs the image bakes: the HEAD SHA of sglang `sglang-miles` (`sgl-project/sglang`) and Megatron-LM `miles-main` (`radixark/Megatron-LM`) — the source branches it builds — plus a fingerprint of the `yueming-yuan/miles-wheels` release it installs, so a rebuilt sgl-router or other wheel also triggers a build (the wheels are pinned by `WHEELS_TAG`, so re-uploads to the same tag are caught by fingerprint, not commit SHA). It compares against the values cached from the last build and sets `should_build=true` if any moved. `miles` itself is intentionally not polled. This is what stops the 12-hour cron from rebuilding an unchanged image.
- **`build-and-push`** (self-hosted runner) — calls `docker/build.py` to build + push, then conditionally points `latest` at the new `dev` and prunes old timestamped tags.

`build-and-push` runs when `check-upstream` was skipped, or ran and reported `should_build=true`.

### Per-trigger behavior

| Trigger | `check-upstream` | builds | `latest`→`dev` | prune |
|---|---|---|---|---|
| schedule (cron 00:00 / 12:00 UTC) | runs; build only if upstream moved | `dev` (default) | yes | yes |
| push to `main` touching `docker/Dockerfile` | skipped | `dev` (default) | no | no |
| `workflow_dispatch` | skipped | per inputs | no | no |
| `workflow_dispatch` + `simulate_schedule` | runs | `dev` (default) | yes | no |

### Steps (`build-and-push`)

1. checkout → set up Buildx → install Python + typer → log in to Docker Hub.
2. **Build and push** — `python3 docker/build.py --variant … --image-tag … [--custom-tag …] --dockerfile … --push`. Empty dispatch inputs fall back to the scheduled default `--variant cu13 --image-tag dev`; `--image-tag dev` pushes both `dev` and `dev-<YYYYMMDDHHMM>` (the multi-arch image).
3. **Point `latest` to `dev`** (schedule / `simulate_schedule`) — `docker buildx imagetools create -t …:latest …:dev`.
4. **Prune old `dev` tags** (schedule only) — keep the newest 20 `dev-<timestamp>`, delete the rest via the Docker Hub API.

## Manual build & push

Use `docker-build.yml`'s manual `workflow_dispatch` (Actions → Run workflow): it builds `docker/Dockerfile` via `build.py` and pushes under the tag you pick (`dev` / `latest` / `custom`).

To pin specific repo versions, `docker/Dockerfile` already takes `MEGATRON_BRANCH` / `SGLANG_COMMIT` / `MILES_COMMIT` build-args. `build.py` does not yet forward arbitrary build-args, so commit-pinning from the workflow needs a small `build.py` change first.

## Image retention (open)

`docker-build.yml` prunes `dev-<timestamp>` tags to the newest 20 (~10 days at 2 builds/day), and `dev` / `latest` move forward. So there is no durable record of which image a past CI run used — reproducing an old run needs retention / immutable tagging, which is a separate, unsolved design.