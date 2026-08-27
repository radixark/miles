---
title: Docker build
description: The Dockerfiles, the build script, rolling and release build workflows, and how to build and push manually.
---
GPU CI runs inside `radixark/miles`. This doc maps which Dockerfiles exist, the script that builds them, the PR-side build check, the rolling and versioned release workflows, and manual build and push behavior.

## Dockerfiles


| Path                     | Builds                   | Wired into                             |
| ------------------------ | ------------------------ | -------------------------------------- |
| `docker/Dockerfile`      | `radixark/miles` (CUDA)  | `docker-build.yml`, `release-docker.yml` |
| `docker/Dockerfile.rocm` | AMD ROCm (MI30x / MI35x) | `docker-build.yml` (`rocm7xx-mi3xx` variants) |


### `docker/Dockerfile` — inputs & output

The Dockerfile is the build recipe: it provides the cu13 defaults and emits one image. `build.py` owns the variant → build-arg overrides (see Build script), including the cu12 base and wheels release.

**Inputs (build-args)**


| Arg                                                                                                    | Meaning                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SGLANG_IMAGE_TAG`                                                                                     | base `lmsysorg/sglang` image tag                                                                                                                                                                                                                                                                                                                    |
| `ENABLE_CUDA_13`                                                                                       | `1` = CUDA 13 (default) and installs the Mooncake wheel from the selected wheels release; `0` = CUDA 12.9 and keeps the base image's Mooncake                                                                                                                                                                                                         |
| `WHEELS_REPO`                                                                                          | prebuilt-wheels GitHub repo (`yueming-yuan/miles-wheels`)                                                                                                                                                                                                                                                                                           |
| `WHEELS_TAG_X86` / `WHEELS_TAG_ARM64`                                                                  | the two **complete** wheels release tags selected by `TARGETARCH` and installed **verbatim**. cu13 uses the rolling `cu130-torch213-x86_64` / `cu130-torch213-aarch64` releases; cu12-x86 overrides `WHEELS_TAG_X86` with the rolling `cu129-x86_64` release                                                                                                                                                                                              |
| `SGLANG_BRANCH` / `SGLANG_COMMIT`, `MEGATRON_REPO` / `MEGATRON_BRANCH` / `MEGATRON_COMMIT`, `MILES_COMMIT`, `SGL_ROUTER_*` | source pins for the layered repos; empty commit args follow their branch, while release builds provide exact SHAs |


**Output** — one `radixark/miles` image for the platform buildx targets: the SGLang base, then the Python dependencies declared in `requirements.txt`, Megatron-LM at its branch default or caller-pinned commit, Miles, and the prebuilt wheels (`sgl-router` among them). A multi-arch build is one `buildx` run executed once per platform — `TARGETARCH` differs each time, so each arch installs its own wheels — and buildx pushes the two as a single manifest.

`docker/Dockerfile.rocm` is the ROCm counterpart (build-args `GPU_ARCH` + a ROCm `SGLANG_IMAGE_TAG`; the 7.2 variants also set `APPLY_ROCR_VMMFIX=1`, which downloads the ROCr VMM-pause fix `.so` from the `WHEELS_TAG_ROCM` release and installs it — ROCm 7.0 has no such regression and leaves it off).

## Build script

`docker/build.py` builds and pushes the images. Select a build with `--variant` and a tag mode with `--image-tag {dev,latest,custom}`. The `VARIANTS` table is the source of truth for each variant's image, target platforms, Dockerfile, and default build-args. Repeatable `--build-arg KEY=VALUE` options are appended after those defaults, so an explicit caller override wins.


| `--variant`    | Tag (`--image-tag dev`)            | Platforms                     | Notes                                          |
| -------------- | ---------------------------------- | ----------------------------- | ---------------------------------------------- |
| `cu13`         | `radixark/miles:dev`               | `linux/amd64` + `linux/arm64` | **multi-arch**, one manifest — the daily image |
| `cu13-x86`     | `radixark/miles:dev`               | `linux/amd64`                 | x86-only build of the same image               |
| `cu13-aarch64` | `radixark/miles:dev`               | `linux/arm64`                 | arm64-only build of the same image             |
| `cu12-x86`     | `radixark/miles:dev-cu12`          | `linux/amd64`                 | CUDA 12.9 legacy                               |
| `rocm700-mi30x` | `rocm/sgl-dev:miles-rocm700-mi30x` | native                       | AMD MI30x (`gfx942`, ROCm 7.0) — `docker/Dockerfile.rocm` |
| `rocm700-mi35x` | `rocm/sgl-dev:miles-rocm700-mi35x` | native                       | AMD MI35x (`gfx950`, ROCm 7.0) — `docker/Dockerfile.rocm` |
| `rocm720-mi35x` | `rocm/sgl-dev:miles-rocm720-mi35x` | native                       | AMD MI35x (`gfx950`, ROCm 7.2) — `docker/Dockerfile.rocm` |


The cu13 variants share one multi-arch CUDA base image and differ only in platforms. `cu13` runs a single `buildx --platform linux/amd64,linux/arm64` — buildx builds both arches and pushes them as one manifest in a single shot, with the Dockerfile picking each layer's wheels by `TARGETARCH` (see Dockerfile inputs), so `docker pull` auto-selects by host arch.

The **Tag** column is for `--image-tag dev`, which also pushes a timestamped `dev-<YYYYMMDDHHMM>` sibling; `latest` swaps the prefix to `latest`, `custom` uses `--custom-tag`. `cu13` / `cu13-x86` / `cu13-aarch64` intentionally share `radixark/miles:dev` — the daily build runs `cu13` (multi-arch), while a single-arch variant overwrites `dev` with one arch when run alone.

A multi-arch build (`cu13`) needs Buildx's `docker-container` driver and is push-only — buildx writes the manifest straight to the registry, it can't load into the local image store. Use `cu13-x86` / `cu13-aarch64` (single-platform; the arm64 one cross-builds via QEMU on an x86 host) for local single-arch iteration. Other flags: `--push`, `--dry-run`, `--dockerfile`, `--custom-tag`, and repeatable `--build-arg KEY=VALUE`.

## PR build check (in `pr-test.yml`)

Dockerfile changes are build-tested on the PR itself, before merge — `docker-build.yml` only runs after a push to `main`, so without this breakage lands on `main` first.

`pr-test.yml` calls `_build-pr-ci-image.yml` after `stage-a-cpu` satisfies its success/bypass gate, while both CPU stages run without waiting for it. A PR keeps **one** image tag, `radixark/miles:pr-<num>`, for its whole life, and rebuilds it only when the content that feeds it changes:

| Job | What it does |
| --- | --- |
| `docker-decide` | hashes the build inputs (`docker/image_inputs.py`) and compares. Inputs equal to the base branch → no PR image, suites run on `dev`. Otherwise it reads the `miles.image-inputs` label off the published `pr-<num>` tag: same hash → reuse it, different or absent → rebuild. Fork PRs cannot publish, so they always test on `dev` |
| `docker-build` | builds `cu13` for `linux/amd64` and `linux/arm64` and pushes `pr-<num>`, stamped with the inputs hash. Runs on a `docker-build` node only when a rebuild is due; otherwise it is a hosted-runner no-op |
| `resolve-ci-image` | resolves the CI image to `pr-<num>` whenever that tag is current — whether this run built it or an earlier one did — so **every GPU suite runs inside the PR's image**; a failed build stops the matrix instead of testing a stale image. A PR image outranks a `ci-image-tag:` PR-body directive, which applies only when the PR has no image (non-docker or fork PRs) |
| `delete-pr-tag` (`docker-pr-tag-cleanup.yml`) | removes the `pr-<num>` tag when the PR closes; the tag stays available for re-runs while the PR is open |

So a rerun, or a push that touches only source files, reuses the image the PR already has instead of rebuilding an identical one. Non-docker PRs are untouched: no PR image, matrix on `dev`, as before.

`docker/image_inputs.py` is the single source of truth for what counts as an input (`docker/Dockerfile`, `docker/build.py`, `docker/install-kube-tools.sh`, `docker/verify_transformer_engine.py`, `docker/patch/**`, `requirements.txt`). `Dockerfile.rocm` is deliberately excluded — it feeds `pr-test-rocm.yml`, not the `cu13` image built here.

To rebuild when the inputs did not change — a moved base image, a floating dependency, a corrupt push — add the **`rebuild-ci-image`** label. Applying it starts a run that rebuilds and then removes the label, so it acts once rather than forcing a rebuild on every later run.

## Rolling Docker build (`docker-build.yml`)

This workflow owns the rolling `dev` and `latest` image families. It has two jobs:

- **`check-upstream`** (schedule / `simulate_schedule` only) — polls the inputs the image bakes: the HEAD SHA of sglang `sglang-miles` (`sgl-project/sglang`) and Megatron-LM `miles-main` (`radixark/Megatron-LM`) — the source branches it builds — plus a fingerprint of the selected `yueming-yuan/miles-wheels` rolling release, so a rebuilt sgl-router or other wheel also triggers a build (re-uploads to the same tag are caught by fingerprint, not commit SHA). It compares against the values cached from the last build and sets `should_build=true` if any moved. `miles` itself is intentionally not polled — that would rebuild far too often. This is what stops the 12-hour cron from rebuilding an unchanged image, with one staleness bound: because the image also bakes a `miles` checkout, `should_build` is forced to `true` once the last triggered build is **24h** old, so `dev` never drifts more than a day behind the `miles` repo even when sglang / Megatron / wheels are quiet. (The cache file's last line records the epoch of the last triggered build; it is only re-saved when a build fires.)
- **`build-and-push`** (self-hosted `docker-build` runner) — calls `docker/build.py` to build + push, then conditionally points `latest` at the new `dev` and prunes old timestamped tags.

`build-and-push` runs when `check-upstream` was skipped, or ran and reported `should_build=true`.

### Triggers: automatic vs manual

- **Automatic** (no human) — the **schedule** (cron 00:00 / 12:00 UTC, gated by `check-upstream`) and any **push to `main` that touches `docker/Dockerfile`, `docker/install-kube-tools.sh`, `docker/verify_transformer_engine.py`, or `requirements.txt`**. Both leave `--variant` empty and build **two images**: `cu13` → `radixark/miles` (multi-arch) and `cu12-x86` → `radixark/miles:dev-cu12`.
- **Manual** — `workflow_dispatch` (pick one variant — see Trigger a build yourself below) or running `docker/build.py` locally. Only the `rocm7xx-mi3xx` images have **no automatic path** (`cu13-x86` / `cu13-aarch64` just rebuild the same `dev` image single-arch).

`docker/build.py` and `docker/patch/**` participate in PR image validation but are not `main`-push triggers; [Release a Version](/ci/04-release) treats them as manual preflight cases.


| Trigger                                     | `check-upstream`                   | builds                | `latest` move     | prune      |
| ------------------------------------------- | ---------------------------------- | --------------------- | ----------------- | ---------- |
| schedule (cron 00:00 / 12:00 UTC)           | runs; build if upstream moved or last build ≥ 24h ago | `cu13` + `cu12-x86`   | yes (both)        | yes (both) |
| push to `main` touching `docker/Dockerfile`, `docker/install-kube-tools.sh`, `docker/verify_transformer_engine.py`, or `requirements.txt` | skipped                            | `cu13` + `cu12-x86`   | no                | no         |
| `workflow_dispatch`                         | skipped                            | the one input variant | no                | no         |
| `workflow_dispatch` + `simulate_schedule`   | runs                               | the one input variant | no                | no         |


### Tags & where it pushes

All images push to **Docker Hub**. CUDA variants → `radixark/miles`; ROCm variants → `rocm/sgl-dev` (a separate namespace). The tag a build writes depends only on `--image-tag` and the variant's postfix:

| variant(s) | `--image-tag dev` writes | `latest` mode writes |
| --- | --- | --- |
| `cu13` / `cu13-x86` / `cu13-aarch64` | `radixark/miles:dev` + `radixark/miles:dev-<YYYYMMDDHHMM>` | `radixark/miles:latest` |
| `cu12-x86` | `radixark/miles:dev-cu12` (+ timestamped sibling) | `radixark/miles:latest-cu12` |
| `rocm700-mi30x` / `rocm700-mi35x` / `rocm720-mi35x` | `rocm/sgl-dev:miles-rocm7xx-mi3xx` (+ timestamped sibling) | `rocm/sgl-dev:latest-rocm7xx-mi3xx` |

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
| `variant` | yes | `cu13` / `cu13-x86` / `cu13-aarch64` / `cu12-x86` / `rocm700-mi30x` / `rocm700-mi35x` / `rocm720-mi35x` |
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

## Versioned release build (`release-docker.yml`)

`release-docker.yml` checks out the supplied release ref, requires its committed `release-lock.json`, and passes the locked SGLang and Megatron-LM commits plus the checked-out Miles SHA as final `--build-arg` overrides. Rolling `docker-build.yml` keeps the branch defaults. Published tags and the guarded dispatch and retry procedure are documented in [Release a Version](/ci/04-release).

## Image retention (open)

`docker-build.yml` prunes `dev-<timestamp>` and `dev-cu12-<timestamp>` as separate series, keeping the newest 20 of each; `dev` / `latest` and `dev-cu12` / `latest-cu12` move forward. Ordinary PR, nightly, and weekly CI therefore has no durable image record. A release branch cut instead retags the newest timestamped `dev` image, or mutable `dev` when none exists, as prune-exempt `release-vX.Y.Z-ci` and records that tag in `release-lock.json`. The guarded preflight is documented in [Release a Version](/ci/04-release).
