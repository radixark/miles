---
title: Docker build
description: The Dockerfiles, the build script, the remote build workflow, and how to build & push manually.
---

# Docker build

GPU CI runs inside `radixark/miles`. This doc maps which Dockerfiles exist, the script that builds them, the PR-side build check, how the remote build is triggered, and how to build & push manually.

## Dockerfiles


| Path                     | Builds                   | Wired into                             |
| ------------------------ | ------------------------ | -------------------------------------- |
| `docker/Dockerfile`      | `radixark/miles` (CUDA)  | `docker-build.yml`                     |
| `docker/Dockerfile.rocm` | AMD ROCm (MI30x / MI35x) | `docker-build.yml` (`rocm-*` variants) |


### `docker/Dockerfile` — inputs & output

The Dockerfile is the build recipe: it provides the cu13 defaults and emits one image. `build.py` owns the variant → build-arg overrides (see Build script), including the cu12 base and wheels release.

**Inputs (build-args)**


| Arg                                                                                                    | Meaning                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SGLANG_IMAGE_TAG`                                                                                     | base `lmsysorg/sglang` image tag                                                                                                                                                                                                                                                                                                                    |
| `ENABLE_CUDA_13`                                                                                       | `1` = CUDA 13 (default) and installs the Mooncake wheel from the selected wheels release; `0` = CUDA 12.9 and keeps the base image's Mooncake                                                                                                                                                                                                         |
| `WHEELS_REPO`                                                                                          | prebuilt-wheels GitHub repo (`yueming-yuan/miles-wheels`)                                                                                                                                                                                                                                                                                           |
| `WHEELS_TAG_X86` / `WHEELS_TAG_ARM64`                                                                  | the two **complete** wheels release tags selected by `TARGETARCH` and installed **verbatim**. cu13 uses the rolling `cu130-x86_64` / `cu130-aarch64` releases; cu12-x86 overrides `WHEELS_TAG_X86` with the rolling `cu129-x86_64` release                                                                                                                                                                                              |
| `SGLANG_COMMIT` / `MEGATRON_COMMIT` / `MILES_COMMIT`                                                   | **required** exact commits for the layered source repos — the build refuses branch-HEAD fallbacks so a cached layer can never silently serve a stale tree. `docker/resolve_upstream.py` resolves current values; `build.py` fills in any the caller didn't pass                                                                                       |
| `WHEELS_FP_X86` / `WHEELS_FP_ARM64`                                                                    | **required** per built arch: asset-list fingerprint of the selected wheels release (from `docker/resolve_upstream.py`) — the cache-buster for the download layer, since rolling tags keep their name while assets get replaced                                                                                                                        |
| `MEGATRON_REPO`, `SGL_ROUTER_*`                                                                        | remaining source knobs for the layered repos                                                                                                                                                                                                                                                                                                          |


**Output** — one `radixark/miles` image for the platform buildx targets. Layer order is ascending change frequency: the sglang base, prebuilt release wheels and pinned third-party installs (TE, apex, `sgl-router` among them), the `requirements.txt` resolve (constrained so it cannot silently move anything already installed), then the fast-moving source trees last — Megatron-LM, sglang, miles (each at its required commit pin). A multi-arch build is one `buildx` run executed once per platform — `TARGETARCH` differs each time, so each arch installs its own wheels — and buildx pushes the two as a single manifest.

`docker/Dockerfile.rocm` is the ROCm counterpart (build-args `GPU_ARCH` + a ROCm `SGLANG_IMAGE_TAG`; the 7.2 variants also set `APPLY_ROCR_VMMFIX=1`, which downloads the ROCr VMM-pause fix `.so` from the `WHEELS_TAG_ROCM` release and installs it — ROCm 7.0 has no such regression and leaves it off).

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


The cu13 variants share one multi-arch CUDA base image and differ only in platforms. `cu13` runs a single `buildx --platform linux/amd64,linux/arm64` — buildx builds both arches and pushes them as one manifest in a single shot, with the Dockerfile picking each layer's wheels by `TARGETARCH` (see Dockerfile inputs), so `docker pull` auto-selects by host arch.

The **Tag** column is for `--image-tag dev`, which also pushes a timestamped `dev-<YYYYMMDDHHMM>` sibling; `latest` swaps the prefix to `latest`, `custom` uses `--custom-tag`. `cu13` / `cu13-x86` / `cu13-aarch64` intentionally share `radixark/miles:dev` — the daily build runs `cu13` (multi-arch), while a single-arch variant overwrites `dev` with one arch when run alone.

A multi-arch build (`cu13`) needs Buildx's `docker-container` driver and is push-only — buildx writes the manifest straight to the registry, it can't load into the local image store. Use `cu13-x86` / `cu13-aarch64` (single-platform; the arm64 one cross-builds via QEMU on an x86 host) for local single-arch iteration. Other flags: `--push`, `--dry-run`, `--dockerfile`, `--custom-tag`, `--build-arg` (repeatable `KEY=VALUE` forwarded verbatim to `docker buildx build`, appended after the variant's own build-args so an explicit value wins), `--builder` (buildx builder to use; CI passes its persistent `miles-builder`).

## PR build check (in `pr-test.yml`)

Dockerfile changes are build-tested on the PR itself, before merge — `docker-build.yml` only runs after a push to `main`, so without this breakage lands on `main` first.

When a PR touches `docker/Dockerfile`, `docker/build.py`, `docker/resolve_upstream.py`, `docker/fetch_wheels.py`, `docker/smoke_test.py`, `docker/requirements-nodeps.txt`, `docker/verify_transformer_engine.py`, `docker/patch/**`, or `requirements.txt` (detected by the `docker-paths` job), `pr-test.yml` inserts a build in front of the test matrix:

| Job | What it does |
| --- | --- |
| `docker-build` | builds `cu13` for `linux/amd64` and `linux/arm64`, then pushes one multi-arch PR-scoped `radixark/miles:pr-<num>` tag (same-repo PRs; fork PRs skip it and test on `dev`) |
| `resolve-ci-image` | waits for the build and resolves the CI image to `pr-<num>`, so **every GPU suite runs inside the freshly built image**; a failed build stops the matrix instead of testing the stale image. The fresh build outranks a `ci-image-tag:` PR-body directive — the directive applies only when no PR image was built (non-docker or fork PRs) |
| `delete-pr-tag` (`docker-pr-tag-cleanup.yml`) | removes the `pr-<num>` tag when the PR closes; the tag stays available for re-runs while the PR is open |

Non-docker PRs are untouched: `docker-paths` reports no change, `docker-build` skips, and the matrix runs on `dev` as before.

## Remote docker build (`docker-build.yml`)

The only automated builder of `radixark/miles`. Two jobs:

- **`resolve-upstream`** (always runs) — runs `docker/resolve_upstream.py` (the single resolver, shared with `build.py`) to resolve the inputs the image bakes: the HEAD SHAs of sglang `sglang-miles` (`sgl-project/sglang`), Megatron-LM `miles-main` (`radixark/Megatron-LM`), and miles itself (the pushed commit on push events, else `main` HEAD), plus a fingerprint of each `yueming-yuan/miles-wheels` rolling release (re-uploads to the same tag are caught by fingerprint, not commit SHA). All values are exposed as job outputs; an empty resolution fails the job, and `build-and-push` bakes the source SHAs via `--build-arg`. On **schedule / `simulate_schedule`** the values additionally gate the rebuild by comparing against the cache from the last gated build. `miles` is intentionally excluded from the value comparison because ordinary source changes would rebuild too often, but the gate forces a build once the last triggered build is **24h** old, so `dev` never drifts more than a day behind the repo when sglang, Megatron, and wheels are quiet.
- **`build-and-push`** (self-hosted `docker-build` runner) — calls `docker/build.py` to build + push, then conditionally points `latest` at the new `dev` and prunes old timestamped tags.

`build-and-push` requires `resolve-upstream` to succeed (a failed resolve blocks the build rather than building unpinned), and on schedule additionally requires `should_build=true`.

### Triggers: automatic vs manual

- **Automatic** (no human) — the **schedule** (cron 00:00 / 12:00 UTC, gated by `resolve-upstream`) and any **push to `main` that touches the same docker paths the PR check watches** (see PR build check above). Both leave `--variant` empty and build **two images**: `cu13` → `radixark/miles` (multi-arch) and `cu12-x86` → `radixark/miles:dev-cu12`.
- **Manual** — `workflow_dispatch` (pick one variant — see Trigger a build yourself below) or running `docker/build.py` locally. Only the `rocm-*` images have **no automatic path** (`cu13-x86` / `cu13-aarch64` just rebuild the same `dev` image single-arch).


| Trigger                                     | rebuild gate (`resolve-upstream`)  | builds                | `latest` move     | prune      |
| ------------------------------------------- | ---------------------------------- | --------------------- | ----------------- | ---------- |
| schedule (cron 00:00 / 12:00 UTC)           | gates; build if upstream moved or the last build is 24h old | `cu13` + `cu12-x86`   | yes (both)        | yes (both) |
| push to `main` touching the watched docker paths | resolves only, no gate             | `cu13` + `cu12-x86`   | no                | no         |
| `workflow_dispatch`                         | resolves only, no gate             | the one input variant | no                | no         |
| `workflow_dispatch` + `simulate_schedule`   | reports the gate signal, doesn't gate | the one input variant | no                | no         |


### Tags & where it pushes

All images push to **Docker Hub**. CUDA variants → `radixark/miles`; ROCm variants → `rocm/sgl-dev` (a separate namespace). The tag a build writes depends only on `--image-tag` and the variant's postfix:

| variant(s) | `--image-tag dev` writes | `latest` mode writes |
| --- | --- | --- |
| `cu13` / `cu13-x86` / `cu13-aarch64` | `radixark/miles:dev` + `radixark/miles:dev-<YYYYMMDDHHMM>` | `radixark/miles:latest` |
| `cu12-x86` | `radixark/miles:dev-cu12` (+ timestamped sibling) | `radixark/miles:latest-cu12` |
| `rocm-mi300` / `rocm-mi350` | `rocm/sgl-dev:miles-rocm7xx-mi3xx` (+ timestamped sibling) | `rocm/sgl-dev:latest-rocm7xx-mi3xx` |

What **moves a shared tag**: `--image-tag dev` overwrites `:dev` (or `:dev-cu12`) and adds a timestamped sibling; on a **scheduled** run `latest`→`dev` *and* `latest-cu12`→`dev-cu12` both advance; pruning likewise runs **only on schedule**, keeping the newest 20 of **each** series — `dev-<ts>` and `dev-cu12-<ts>` independently. Any `workflow_dispatch` — **including** `simulate_schedule` — writes its own tag(s) but never moves `latest` or prunes; only the real cron mutates published tags. See the trigger table above.

### Trigger a build yourself

Manual builds run through `workflow_dispatch` — by default the image is built straight from the inputs you pass; with `simulate_schedule`, `resolve-upstream` additionally reports the gate signal but does not gate the manual build (see the `workflow_dispatch` rows in the table above for how this differs from schedule). Start one two ways:

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
| `simulate_schedule` | no | `true` makes `resolve-upstream` report the rebuild-gate signal (default `false`) |

### Steps (`build-and-push`)

1. checkout → ensure the persistent `miles-builder` buildx builder (node-local layer cache; created once per node, reused by every build including PR builds) → Docker Hub login. No host package installs: `build.py` is stdlib-only, so build nodes need just docker and stock `python3`.
2. **GPU smoke gate** — before anything is pushed, the amd64 image is `--load`ed from the builder cache and booted on the build node's GPU running `docker/smoke_test.py` (CUDA visible + tensor math, real TE/sglang/miles imports, nccl-tests binaries). A broken image never reaches a tag; the CUDA-variant builds all pass through it (arm64 halves ship unprobed — no ARM GPU nodes; `rocm-*` and `cu13-aarch64` skip it).
3. **Build + push** via `build.py` — automatic runs build **both** `cu13` and `cu12-x86` (amd64 layers all cache hits from the gate build); a manual dispatch builds only the one variant you picked.
4. **schedule only** — point `latest`→`dev` and `latest-cu12`→`dev-cu12`.
5. **schedule only** — prune each timestamp series to the newest 20.

### Push auth & permissions

Pushes use a Docker Hub credential, not your identity:

- **Remote (CI)** — the workflow logs in with repo secrets `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`, so you don't hold the key — you just trigger the run, which needs repo **Write** access. No approval gate, but `build-and-push` runs on a `self-hosted` runner, so it only fires when one is online.
- **Local** — `build.py --push` uses your own `docker login`; you need push rights to the target namespace (`radixark/miles`, or `rocm/sgl-dev` for ROCm).

### Pinning specific repo versions — reproducible builds

Every image is now built from exact pins: CI passes the `resolve-upstream` SHAs as `--build-arg SGLANG_COMMIT/MEGATRON_COMMIT/MILES_COMMIT`, `build.py` resolves the per-variant wheels fingerprints, and the Dockerfile hard-fails on any missing pin instead of following a branch HEAD. To rebuild a historical image, pass its pins explicitly — e.g. `python docker/build.py --variant cu13-x86 --image-tag custom --custom-tag repro --build-arg SGLANG_COMMIT=<sha> --build-arg MEGATRON_COMMIT=<sha> --build-arg MILES_COMMIT=<sha>` (wheels fingerprints only key the cache, so bit-exact wheel reproduction additionally needs the release assets to be unchanged). A plain local `build.py` run with no `--build-arg` resolves all pins itself via `docker/resolve_upstream.py` and prints each one.

## Image retention (open)

`docker-build.yml` prunes `dev-<timestamp>` and `dev-cu12-<timestamp>` as separate series, keeping the newest 20 of each; `dev` / `latest` and `dev-cu12` / `latest-cu12` move forward. So there is no durable record of which image a past CI run used — reproducing an old run needs retention / immutable tagging, which is a separate, unsolved design.
