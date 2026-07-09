---
title: Docker build
description: The Dockerfiles, the build script, the remote build workflow, and how to build & push manually.
---

# Docker build

CI runs inside `radixark/miles`. This doc maps which Dockerfiles exist, the script that builds them, how the remote build is triggered, and how to build & push manually.

## Dockerfiles


| Path                     | Builds                   | Wired into                             |
| ------------------------ | ------------------------ | -------------------------------------- |
| `docker/Dockerfile`      | `radixark/miles` (CUDA)  | `docker-build.yml`                     |
| `docker/Dockerfile.rocm` | AMD ROCm (MI30x / MI35x) | `docker-build.yml` (`rocm-*` variants) |


### `docker/Dockerfile` — inputs & output

The Dockerfile is the build recipe and nothing more: it knows no variants and no tags. Everything it needs arrives as build-args; it emits one image. `build.py` owns the variant → build-arg mapping (see Build script), so the boundary stays clean — e.g. the wheels-repo tag naming lives only in `build.py`, never here.

**Inputs (build-args)**


| Arg                                                                                                    | Meaning                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SGLANG_IMAGE_TAG`                                                                                     | base `lmsysorg/sglang` tag (default `v0.5.12`, a multi-arch release)                                                                                                                                                                                                                                                                                |
| `ENABLE_CUDA_13`                                                                                       | `1` = CUDA 13 (default), `0` = CUDA 12.9                                                                                                                                                                                                                                                                                                            |
| `WHEELS_REPO`                                                                                          | prebuilt-wheels GitHub repo (`yueming-yuan/miles-wheels`)                                                                                                                                                                                                                                                                                           |
| `WHEELS_TAG_X86` / `WHEELS_TAG_ARM64`                                                                  | the two **complete** wheels release tags (e.g. `cu130-x86_64-v0.5.12` / `cu130-aarch64-v0.5.12`), the wheels repo's own names. In a multi-arch build the Dockerfile **picks one by `TARGETARCH`** (the only per-platform value buildx varies) and installs it **verbatim** — never assembling a tag from parts; cu12-x86 overrides `WHEELS_TAG_X86` |
| `SGLANG_BRANCH` / `SGLANG_COMMIT`, `MEGATRON_REPO` / `MEGATRON_BRANCH`, `MILES_COMMIT`, `SGL_ROUTER_*` | source pins for the layered repos                                                                                                                                                                                                                                                                                                                   |


**Output** — one `radixark/miles` image for the platform buildx targets: the sglang base, then Megatron-LM (`radixark/Megatron-LM@miles-main`), miles, and the prebuilt wheels (`sgl-router` among them). A multi-arch build is one `buildx` run executed once per platform — `TARGETARCH` differs each time, so each arch installs its own wheels — and buildx pushes the two as a single manifest.

`docker/Dockerfile.rocm` is the ROCm counterpart (build-args `GPU_ARCH` + a ROCm `SGLANG_IMAGE_TAG`).

## Build script

`docker/build.py` builds and pushes the images. Select a build with `--variant` and a tag mode with `--image-tag {dev,latest,custom}`. A single `VARIANTS` table is the source of truth for each variant's image, target platforms, Dockerfile, and build-args.


| `--variant`    | Tag (`--image-tag dev`)            | Platforms                     | Notes                                          |
| -------------- | ---------------------------------- | ----------------------------- | ---------------------------------------------- |
| `cu13`         | `radixark/miles:dev`               | `linux/amd64` + `linux/arm64` | **multi-arch**, one manifest — the daily image |
| `cu13-x86`     | `radixark/miles:dev`               | `linux/amd64`                 | x86-only build of the same image               |
| `cu13-aarch64` | `radixark/miles:dev`               | `linux/arm64`                 | arm64-only build of the same image             |
| `cu12-x86`     | `radixark/miles:dev-cu12`          | `linux/amd64`                 | CUDA 12.9 legacy                               |
| `rocm-mi300`   | `rocm/sgl-dev:miles-rocm700-mi30x` | native                        | AMD MI30x — `docker/Dockerfile.rocm`           |
| `rocm-mi350`   | `rocm/sgl-dev:miles-rocm720-mi35x` | native                        | AMD MI35x — `docker/Dockerfile.rocm`           |


The cu13 variants share one CUDA base (`lmsysorg/sglang:v0.5.12`, multi-arch) and differ only in platforms. `cu13` runs a single `buildx --platform linux/amd64,linux/arm64` — buildx builds both arches and pushes them as one manifest in a single shot, with the Dockerfile picking each layer's wheels by `TARGETARCH` (see Dockerfile inputs), so `docker pull` auto-selects by host arch.

The **Tag** column is for `--image-tag dev`, which also pushes a timestamped `dev-<YYYYMMDDHHMM>` sibling; `latest` swaps the prefix to `latest`, `custom` uses `--custom-tag`. `cu13` / `cu13-x86` / `cu13-aarch64` intentionally share `radixark/miles:dev` — the daily build runs `cu13` (multi-arch), while a single-arch variant overwrites `dev` with one arch when run alone.

A multi-arch build (`cu13`) needs Buildx's `docker-container` driver and is push-only — buildx writes the manifest straight to the registry, it can't load into the local image store. Use `cu13-x86` / `cu13-aarch64` (single-platform; the arm64 one cross-builds via QEMU on an x86 host) for local single-arch iteration. Other flags: `--push`, `--dry-run`, `--dockerfile`, `--custom-tag`.

## Remote docker build (`docker-build.yml`)

The only automated builder of `radixark/miles`. Two jobs:

- **`check-upstream`** (schedule / `simulate_schedule` only) — polls the inputs the image bakes: the HEAD SHA of sglang `sglang-miles` (`sgl-project/sglang`) and Megatron-LM `miles-main` (`radixark/Megatron-LM`) — the source branches it builds — plus a fingerprint of the `yueming-yuan/miles-wheels` release it installs, so a rebuilt sgl-router or other wheel also triggers a build (the wheels are pinned by `WHEELS_TAG`, so re-uploads to the same tag are caught by fingerprint, not commit SHA). It compares against the values cached from the last build and sets `should_build=true` if any moved. `miles` itself is intentionally not polled. This is what stops the 12-hour cron from rebuilding an unchanged image.
- **`build-and-push`** (self-hosted runner) — calls `docker/build.py` to build + push, then conditionally points `latest` at the new `dev` and prunes old timestamped tags.

`build-and-push` runs when `check-upstream` was skipped, or ran and reported `should_build=true`.

### Triggers: automatic vs manual

- **Automatic** (no human) — the **schedule** (cron 00:00 / 12:00 UTC, gated by `check-upstream`) and any **push to `main` that touches `docker/Dockerfile`**. Both leave `--variant` empty and build **two images**: `cu13` → `radixark/miles` (multi-arch) and `cu12-x86` → `radixark/miles:dev-cu12`.
- **Manual** — `workflow_dispatch` (pick one variant — see Trigger a build yourself below) or running `docker/build.py` locally. Only the `rocm-*` images have **no automatic path** (`cu13-x86` / `cu13-aarch64` just rebuild the same `dev` image single-arch).


| Trigger                                     | `check-upstream`                   | builds                | `latest` move     | prune      |
| ------------------------------------------- | ---------------------------------- | --------------------- | ----------------- | ---------- |
| schedule (cron 00:00 / 12:00 UTC)           | runs; build only if upstream moved | `cu13` + `cu12-x86`   | yes (both)        | yes (both) |
| push to `main` touching `docker/Dockerfile` | skipped                            | `cu13` + `cu12-x86`   | no                | no         |
| `workflow_dispatch`                         | skipped                            | the one input variant | no                | no         |
| `workflow_dispatch` + `simulate_schedule`   | runs                               | the one input variant | no                | no         |


### Tags & where it pushes

All images push to **Docker Hub**. CUDA variants → `radixark/miles`; ROCm variants → `rocm/sgl-dev` (a separate namespace). The tag a build writes depends only on `--image-tag` and the variant's postfix:

| variant(s) | `--image-tag dev` writes | `latest` mode writes |
| --- | --- | --- |
| `cu13` / `cu13-x86` / `cu13-aarch64` | `radixark/miles:dev` + `radixark/miles:dev-<YYYYMMDDHHMM>` | `radixark/miles:latest` |
| `cu12-x86` | `radixark/miles:dev-cu12` (+ timestamped sibling) | `radixark/miles:latest-cu12` |
| `rocm-mi300` / `rocm-mi350` | `rocm/sgl-dev:miles-rocm7xx-mi3xx` (+ timestamped sibling) | `rocm/sgl-dev:latest-rocm7xx-mi3xx` |

What **moves a shared tag**: `--image-tag dev` overwrites `:dev` (or `:dev-cu12`) and adds a timestamped sibling; on a **scheduled** run `latest`→`dev` *and* `latest-cu12`→`dev-cu12` both advance; pruning likewise runs **only on schedule**, keeping the newest 20 of **each** series — `dev-<ts>` and `dev-cu12-<ts>` independently. Any `workflow_dispatch` — **including** `simulate_schedule` — writes its own tag(s) but never moves `latest` or prunes; only the real cron mutates published tags. See the trigger table above.

### Trigger a build yourself

Manual builds run through `workflow_dispatch` — by default the image is built straight from the inputs you pass; with `simulate_schedule`, `check-upstream` runs first for signal but does not gate the manual build (see the `workflow_dispatch` rows in the table above for how this differs from schedule). Start one two ways:

- **Web UI** — Actions → "Docker Build & Push" → **Run workflow**, then fill the inputs below.
- **CLI** — `gh` dispatches on the repo's default branch; pass `--ref <branch>` to build another branch's workflow.

```bash
gh workflow run docker-build.yml -f variant=cu13 -f image_tag=dev
# custom tag instead of dev/latest:
gh workflow run docker-build.yml -f variant=cu13-x86 -f image_tag=custom -f custom_tag=my-tag
```

| input | required | values / default |
| ----- | -------- | ---------------- |
| `variant` | yes | `cu13` / `cu13-x86` / `cu13-aarch64` / `cu12-x86` / `rocm-mi300` / `rocm-mi350` |
| `image_tag` | yes | `dev` / `latest` / `custom` |
| `custom_tag` | no | tag name; required when `image_tag=custom` |
| `dockerfile` | no | path to Dockerfile (default `docker/Dockerfile`) |
| `simulate_schedule` | no | `true` runs the `check-upstream` poll first (default `false`) |

### Steps (`build-and-push`)

1. checkout → Buildx → install Python + typer → Docker Hub login.
2. **Build + push** via `build.py` — automatic runs build **both** `cu13` and `cu12-x86`; a manual dispatch builds only the one variant you picked.
3. **schedule only** — point `latest`→`dev` and `latest-cu12`→`dev-cu12`.
4. **schedule only** — prune each timestamp series to the newest 20.

### Push auth & permissions

Pushes use a Docker Hub credential, not your identity:

- **Remote (CI)** — the workflow logs in with repo secrets `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`, so you don't hold the key — you just trigger the run, which needs repo **Write** access. No approval gate, but `build-and-push` runs on a `self-hosted` runner, so it only fires when one is online.
- **Local** — `build.py --push` uses your own `docker login`; you need push rights to the target namespace (`radixark/miles`, or `rocm/sgl-dev` for ROCm).

### Pinning specific repo versions

`docker/Dockerfile` already takes `MEGATRON_BRANCH` / `SGLANG_COMMIT` / `MILES_COMMIT` build-args, but `build.py` does not yet forward arbitrary build-args and `workflow_dispatch` exposes no input for them — so commit-pinning from the workflow needs two changes first: a passthrough in `build.py` and matching inputs in `docker-build.yml`.

## Image retention (open)

`docker-build.yml` prunes `dev-<timestamp>` and `dev-cu12-<timestamp>` as separate series, keeping the newest 20 of each; `dev` / `latest` and `dev-cu12` / `latest-cu12` move forward. So there is no durable record of which image a past CI run used — reproducing an old run needs retention / immutable tagging, which is a separate, unsolved design.
