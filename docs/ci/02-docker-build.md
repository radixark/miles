---

## title: Docker build
description: The Dockerfiles, the build script, the remote build workflow, and how to build & push manually.

# Docker build

CI runs inside `radixark/miles`. This doc maps which Dockerfiles exist, the script that builds them, how the remote build is triggered, and how to build & push manually.

## Dockerfiles


| Path                                                             | Builds                               | Wired into                            |
| ---------------------------------------------------------------- | ------------------------------------ | ------------------------------------- |
| `docker/Dockerfile`                                              | `radixark/miles` — the CI base image | `docker-build.yml` |
| `docker/Dockerfile.rocm_MI300`, `docker/Dockerfile.rocm_MI350-5` | AMD ROCm                             | manual                                |


Only `docker/Dockerfile` is built by automation; the ROCm Dockerfiles are arch-specific and built by hand. The main Dockerfile is multi-stage: it starts `FROM lmsysorg/sglang:${SGLANG_IMAGE_TAG}` (a pinned `lmsysorg/sglang` release, set by the `SGLANG_IMAGE_TAG` default), installs the sglang source from the `sglang-miles` branch, then layers Megatron-LM (`radixark/Megatron-LM@miles-main`), miles, sgl-router, and prebuilt wheels from `yueming-yuan/miles-wheels`.

## Build script

`docker/build.py` builds and pushes the image. Pick a build with `--cuda {129,130}` (CUDA 12.9 / 13.0) and `--arch {x86,aarch64}`; it derives the build-args and tag from a single `(cuda, arch)` table — the single source of truth that `docker/Dockerfile`'s header defers to. A safety check rejects any build-arg the Dockerfile doesn't declare as an `ARG`, so the table can't drift from the Dockerfile.

The image tag is **prefix + suffix**:

- **prefix** — set by `--tag`, one of:
  - `dev` (default) — rolling CI image: `dev` + timestamped `dev-<YYYYMMDDHHMM>` sibling. PR CI pulls `dev`.
  - `latest` — stable pointer; nightly repoints it at the new `dev` instead of building it (`--tag latest` pushes verbatim).
  - any literal — one-off tag, pushed verbatim (no sibling, never pruned).
- **suffix** — the CUDA variant: cu13 (`--cuda 130`) has no suffix; cu12 (`--cuda 129`) is `-cu12`. The arch is never in the tag — cu13 is meant to be one multi-arch image (x86 + aarch64; aarch64 lands in Step B), cu12 is x86-only.

So `--tag dev` → `dev` (cu13) or `dev-cu12` (cu12); `--tag latest` → `latest` / `latest-cu12`.

Other flags: `--test` (append `-test` to the tag, e.g. `dev-test` — a single, non-timestamped throwaway image), `--push`, `--dry-run`, `--dockerfile`.

| `--cuda` | `--arch` | Tag (prefix `dev`) | Wired? |
|---|---|---|---|
| 130 | x86 | `dev` | yes |
| 130 | aarch64 | `dev` (same multi-arch manifest) | Step B |
| 129 | x86 | `dev-cu12` | yes |

## Remote docker build (`docker-build.yml`)

The only automated builder of `radixark/miles`. Two jobs:

- **`check-upstream`** (schedule / `simulate_schedule` only) — fetches the HEAD SHA of sglang `sglang-miles` (`sgl-project/sglang`) and Megatron-LM `miles-main` (`radixark/Megatron-LM`) — the same branches the Dockerfile builds (`SGLANG_BRANCH`, `MEGATRON_REPO`/`MEGATRON_BRANCH`) — compares to the SHAs cached from the last build, and sets `should_build=true` only if one moved. This is what stops the 12-hour cron from rebuilding an unchanged image.
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
2. **Build and push** — `python3 docker/build.py --cuda … --arch … --tag … --dockerfile … [--test] --push`. Empty dispatch inputs fall back to the scheduled default `--cuda 130 --arch x86 --tag dev`; `tag=dev` pushes both `dev` and `dev-<YYYYMMDDHHMM>`.
3. **Point `latest` to `dev`** (schedule / `simulate_schedule`) — `docker buildx imagetools create -t …:latest …:dev`.
4. **Prune old `dev` tags** (schedule only) — keep the newest 20 `dev-<timestamp>`, delete the rest via the Docker Hub API.

## Manual build & push

Use `docker-build.yml`'s manual `workflow_dispatch` (Actions → Run workflow): it builds `docker/Dockerfile` via `build.py` and pushes under the tag you pick (`dev` / `latest` / `custom`).

To pin specific repo versions, `docker/Dockerfile` already takes `MEGATRON_BRANCH` / `SGLANG_COMMIT` / `MILES_COMMIT` build-args. `build.py` does not yet forward arbitrary build-args, so commit-pinning from the workflow needs a small `build.py` change first.

## Image retention (open)

`docker-build.yml` prunes `dev-<timestamp>` tags to the newest 20 (~10 days at 2 builds/day), and `dev` / `latest` move forward. So there is no durable record of which image a past CI run used — reproducing an old run needs retention / immutable tagging, which is a separate, unsolved design.