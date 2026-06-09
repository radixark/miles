---

## title: Docker build
description: The Dockerfiles, the build script, the remote build workflow, and how to build & push manually.

# Docker build

CI runs inside `radixark/miles`. This doc maps which Dockerfiles exist, the script that builds them, how the remote build is triggered, and how to build & push manually.

## Dockerfiles


| Path                                                             | Builds                               | Wired into                            |
| ---------------------------------------------------------------- | ------------------------------------ | ------------------------------------- |
| `docker/Dockerfile`                                              | `radixark/miles` — the CI base image | `docker-build.yml`, `docker/justfile` |
| `docker/Dockerfile.rocm_MI300`, `docker/Dockerfile.rocm_MI350-5` | AMD ROCm                             | manual                                |


Only `docker/Dockerfile` is built by automation; the ROCm Dockerfiles are arch-specific and built by hand. The main Dockerfile is multi-stage: it starts `FROM lmsysorg/sglang:${SGLANG_IMAGE_TAG}` (a pinned `lmsysorg/sglang` release, set by the `SGLANG_IMAGE_TAG` default), installs the sglang source from the `sglang-miles` branch, then layers Megatron-LM (`radixark/Megatron-LM@miles-main`), miles, sgl-router, and prebuilt wheels from `yueming-yuan/miles-wheels`.

## Build script

*To discuss — left blank. `docker/build.py`'s variant / tag-mode model is unclear and needs alignment before documenting.*

## Remote docker build (`docker-build.yml`)

Builds the rolling `radixark/miles:dev` / `:latest`.

Triggers:

- push to `main` that touches `docker/Dockerfile`
- schedule: cron at 00:00 and 12:00 UTC
- manual `workflow_dispatch` (inputs: `variant`, `image_tag`, `custom_tag`, `dockerfile`, `simulate_schedule`)

Scheduled process:

1. `**check-upstream*`* — fetch sglang `sglang-miles` HEAD and Megatron-LM `main` HEAD, compare to the cached SHAs, and set `should_build` only if either changed. This stops the 12-hour cron from rebuilding an unchanged image.
2. `**build-and-push**` — build and push `radixark/miles:dev` on a self-hosted runner (via `docker/build.py`), then retag `latest` → `dev` and prune `dev-<timestamp>` tags down to the newest 20.

On a `docker/Dockerfile` push, `check-upstream` is skipped and `build-and-push` runs directly. Manual runs build whatever the dispatch inputs select, from the Actions UI.

## Manual build & push

Use `docker-build.yml`'s manual `workflow_dispatch` (Actions → Run workflow): it builds `docker/Dockerfile` via `build.py` and pushes under the tag you pick (`dev` / `latest` / `custom`).

To pin specific repo versions, `docker/Dockerfile` already takes `MEGATRON_BRANCH` / `SGLANG_COMMIT` / `MILES_COMMIT` build-args. `build.py` does not yet forward arbitrary build-args, so commit-pinning from the workflow needs a small `build.py` change first.

## Image retention (open)

`docker-build.yml` prunes `dev-<timestamp>` tags to the newest 20 (~10 days at 2 builds/day), and `dev` / `latest` move forward. So there is no durable record of which image a past CI run used — reproducing an old run needs retention / immutable tagging, which is a separate, unsolved design.