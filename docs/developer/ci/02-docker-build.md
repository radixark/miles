---
title: Docker build
description: The Dockerfiles, the build script, rolling and release build workflows, and how to build and push manually.
---
GPU CI runs inside `radixark/miles`. This doc maps which Dockerfiles exist, the script that builds them, the PR-side build check, the rolling and versioned release workflows, and manual build and push behavior.

## Dockerfiles


| Path                     | Builds                   | Wired into                             |
| ------------------------ | ------------------------ | -------------------------------------- |
| `docker/Dockerfile`      | `radixark/miles` (CUDA)  | `docker-build.yml`, `release-docker.yml` |
| `docker/Dockerfile.rubin` | Vera Rubin (aarch64/SM107) | `docker-build.yml` (`rubin` manual variant) |
| `docker/Dockerfile.tpu` | TPU v7x through torch_tpu (x86_64) | `docker-build.yml` (`tpu` variant, rolling `dev-tpu`) |
| `docker/Dockerfile.rocm` | AMD ROCm (MI35x, MI30x) | `docker-build.yml` (`rocm*-mi35x`, `rocm10-mi30x` variants) |
| `docker/Dockerfile.cu12` | Deprecated CUDA 12.9 (x86_64), unmaintained | Manual builds only |


CUDA 12 image builds are retired, following [SGLang v0.5.20](https://github.com/sgl-project/sglang/releases/tag/v0.5.20). Existing CUDA 12 image tags remain available, but are no longer rebuilt, retagged, or pruned by this workflow. The old recipe is preserved in `docker/Dockerfile.cu12` with the former `cu12-x86` defaults (`v0.5.19-cu129`, `sglang-miles-v0.5.19-final`, and `cu129-x86_64`); it is deprecated and no longer maintained.

### `docker/Dockerfile` — inputs & output

The Dockerfile is the build recipe: it provides the cu13 defaults and emits one image. `build.py` owns the variant → build-arg overrides (see Build script), including the ROCm base and wheels releases.

The CUDA 13 defaults use `lmsysorg/sglang:v0.5.22` and the stable branch `sglang-miles`. `SGLANG_COMMIT` remains empty unless the caller pins a revision. From that checkout it installs `python/` and `radix_transfer_plugins/`, the output store package SGLang imports, both editable, and the build fails unless that package imports; its Mooncake backend is left out because `mooncake.store` needs the GPU driver. ROCm variants keep their separate base and wheels pins in `docker/build.py`.

**Inputs (build-args)**


| Arg                                                                                                    | Meaning                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SGLANG_IMAGE_TAG`                                                                                     | base `lmsysorg/sglang` image tag                                                                                                                                                                                                                                                                                                                    |
| `WHEELS_REPO`                                                                                          | prebuilt-wheels GitHub repo (`radixark/miles-wheels`)                                                                                                                                                                                                                                                                                           |
| `WHEELS_TAG_X86` / `WHEELS_TAG_ARM64`                                                                  | the two **complete** wheels release tags selected by `TARGETARCH` and installed **verbatim**. cu13 uses the rolling `cu130-torch214-x86_64` / `cu130-torch214-aarch64` releases                                                                                                                                                                                              |
| `SGLANG_BRANCH` / `SGLANG_COMMIT`, `MEGATRON_REPO` / `MEGATRON_BRANCH` / `MEGATRON_COMMIT`, `MILES_COMMIT`, `SGL_ROUTER_*` | source pins for the layered repos; empty commit args follow their branch, while release builds provide exact SHAs |
| `TARGETARCH`                                                                                           | set by buildx; also the one argument `docker/install-kube-tools.sh` takes                                                                                                                                                                                                                                                                           |


**Output** — one `radixark/miles` image for the platform buildx targets: the SGLang base, then the Python dependencies declared in `requirements.txt`, Megatron-LM at its branch default or caller-pinned commit, Miles, and the prebuilt wheels (`sgl-router` among them). It also carries `kubectl`, `helm` and `tmux`, which the k8s-native launch path drives the cluster with. A multi-arch build is one `buildx` run executed once per platform — `TARGETARCH` differs each time, so each arch installs its own wheels — and buildx pushes the two as a single manifest.

`docker/Dockerfile.rocm` is the ROCm counterpart (build-args `GPU_ARCH`, a ROCm `SGLANG_IMAGE_TAG`, and a `WHEELS_TAG_ROCM` release from `XinyuJiangCMU/miles-wheels-rocm`). Rolling ROCm variants and the PR variant use the `lmsysorg/sglang` ROCm releases pinned in `docker/build.py`, so `docker/Dockerfile.rocm` fetches the matching stable branch `sglang-miles`. `rocm724-mi35x` uses the Python 3.12 base and ROCm 7.2.4 wheels from the `rocm724-gfx950-v0.5.20` release. It sets `APPLY_ROCR_VMMFIX=1` to install the point-release-matched ROCr VMM-pause fix; ROCm 10 has the fix upstream. The MI300X / MI325X (`gfx942`) variant `rocm10-mi30x` uses the `rocm10-gfx942-v0.5.20` release. That release holds Transformer Engine, flash-attn and apex built for `gfx942` from the same sources as the `gfx950` shelf, so the variant installs every wheel prebuilt, like `rocm10-mi35x`. It also sets `AITER_PREBUILD_JIT=1`. `docker/amd/prebuild_aiter_jit.py` then builds the bf16 aiter attention kernels that sglang otherwise JIT-compiles on first use: the `mha_batch_prefill` extend modules, and `pa_ragged` decode for GQA ratios 1–16, head sizes 64 and 128, and contexts up to 64K. It builds them for `GPU_ARCH`, because the build host has no GPU.

### `docker/install-kube-tools.sh`

`kubectl` and `helm` are not optional extras of a training image: the workbench installs releases with them and the launcher drives the run with them, so **every** image `docker/build.py` can build must carry them. The pinned versions and their checksum verification therefore live in one script that each training Dockerfile copies and runs, rather than in a `RUN` block per Dockerfile that only one of them ever got. Bumping either client is an edit to `docker/install-kube-tools.sh` alone.

## Build script

`docker/build.py` builds and pushes the images. Select a build with `--variant` and a tag mode with `--image-tag {dev,latest,custom}`. The `VARIANTS` table is the source of truth for each variant's image, target platforms, Dockerfile, and default build-args. Repeatable `--build-arg KEY=VALUE` options are appended after those defaults, so an explicit caller override wins.


| `--variant`    | Tag (`--image-tag dev`)            | Platforms                     | Notes                                          |
| -------------- | ---------------------------------- | ----------------------------- | ---------------------------------------------- |
| `cu13`         | `radixark/miles:dev`               | `linux/amd64` + `linux/arm64` | **multi-arch**, one manifest — the daily image |
| `cu13-x86`     | `radixark/miles:dev`               | `linux/amd64`                 | x86-only build of the same image               |
| `cu13-aarch64` | `radixark/miles:dev`               | `linux/arm64`                 | arm64-only build of the same image             |
| `rubin` | `radixark/miles:dev-rubin` | `linux/arm64` | Vera Rubin — `docker/Dockerfile.rubin` |
| `tpu` | `radixark/miles:dev-tpu` | `linux/amd64` | TPU v7x (torch_tpu) — `docker/Dockerfile.tpu` |
| `rocm724-mi35x` | `rocm/sgl-dev:miles-rocm724-mi35x` | native                       | AMD MI35x (`gfx950`, ROCm 7.2.4) — `docker/Dockerfile.rocm` |
| `rocm10-mi35x`  | `rocm/sgl-dev:miles-rocm10-mi35x`  | native                       | AMD MI35x (`gfx950`, ROCm 10, ROCm SDK as pip wheels) — `docker/Dockerfile.rocm` |
| `rocm10-mi35x-pr` | PR CI custom tag: `radixark/miles:pr-<number>-rocm10-mi35x` | `linux/amd64` | Same ROCm recipe; PR CI uses `--image-tag custom --custom-tag pr-<number>` |
| `rocm10-mi30x`  | `rocm/sgl-dev:miles-rocm10-mi30x`  | native                       | AMD MI300X / MI325X (`gfx942`, ROCm 10) — `docker/Dockerfile.rocm` |


The cu13 variants share one multi-arch CUDA base image and differ only in platforms. `cu13` runs a single `buildx --platform linux/amd64,linux/arm64` — buildx builds both arches and pushes them as one manifest in a single shot, with the Dockerfile picking each layer's wheels by `TARGETARCH` (see Dockerfile inputs), so `docker pull` auto-selects by host arch.

The **Tag** column is for `--image-tag dev`, which also pushes a timestamped `dev-<YYYYMMDDHHMM>` sibling; `latest` swaps the prefix to `latest`, `custom` uses `--custom-tag`. `cu13` / `cu13-x86` / `cu13-aarch64` intentionally share `radixark/miles:dev` — the daily build runs `cu13` (multi-arch), while a single-arch variant overwrites `dev` with one arch when run alone.

A multi-arch build (`cu13`) needs Buildx's `docker-container` driver. Use `--push` to publish directly, or `--output type=oci,dest=/tmp/image,tar=false` to export a local OCI layout for later publication; it cannot load into the local image store. Use `cu13-x86` / `cu13-aarch64` (single-platform; the arm64 one cross-builds via QEMU on an x86 host) for local single-arch iteration. Other flags: `--push`, `--dry-run`, `--dockerfile`, `--custom-tag`, and repeatable `--build-arg KEY=VALUE`. `--context <repository>` selects a separate build context while keeping the driver and hash reader in their original checkout. With `BUILD_GITHUB_TOKEN` set, `build.py` hands it to the build as the BuildKit secret `github_token`, which the Dockerfiles' GitHub API release lookups authenticate with; anonymous lookups from a shared builder IP hit GitHub's rate limit (HTTP 403). The trusted CI builds pass the job token; fork builds never get it.

The `rubin` variant uses the existing `docker-build` runner and targets `linux/arm64`, like `cu13-aarch64`. It is not included in scheduled builds, automatic tag promotion/pruning, regular CUDA PR images or versioned releases.

The `tpu` variant builds from sglang-torchtpu's release image (`SGLANG_TPU_IMAGE_TAG`): it swaps in sglang-torchtpu at `SGLANG_TPU_COMMIT` (the release's libtpu and jax), replaces torch_tpu with the wheel from the `TORCH_TPU_WHEELS_TAG` release of `WHEELS_REPO` (torch_tpu main plus `docker/tpu_patch/torch_tpu_job_local_worlds.patch`), moves the one Python environment to `TORCH_VERSION` (the CUDA image's torch; the torchtitan backend needs >= 2.14), and adds torchtitan at `TORCHTITAN_COMMIT` with `docker/tpu_patch/torchtitan_torch_tpu.patch` and miles. `docker/tpu_patch/` feeds only this image, so it stays out of `docker/image_inputs.py`. The automatic `docker-build.yml` runs build it next to `cu13` as `radixark/miles:dev-tpu` (see Rolling Docker build); it has no versioned release. A PR that changes its inputs gets `radixark/miles:pr-<number>-tpu` from `pr-test-tpu.yml` ([`.github/tpu-ci/README.md`](../../../.github/tpu-ci/README.md)), deleted with `pr-<number>` when the PR closes. `scripts/tpu/README.md` runs it.

## PR build check (in `pr-test.yml`)

A PR that changes the image inputs build-tests its image before merge, with or without selected CUDA tests. The build does not itself request GPU execution; select CUDA tests, for example with `run-ci-image`, to also run suites inside the new image.

`pr-test.yml` calls `_build-pr-ci-image.yml` once `stage-a-cpu` satisfies its success/bypass gate; the call is a hosted-runner no-op when the inputs match the base. `resolve-ci-image` runs only when the final stage selection contains CUDA tests, so CPU-only PRs build an image only when their inputs changed and never resolve one. Both CPU stages run without waiting for images. An eligible PR keeps **one** image tag, `radixark/miles:pr-<num>`, for its whole life, and rebuilds it only when the content that feeds it changes:

| Job | What it does |
| --- | --- |
| `docker-decide` | hashes the build inputs (`docker/image_inputs.py`) and compares. Inputs equal to the base branch → no PR image, suites run on `dev`. Otherwise it reads the `miles.image-inputs` label off the published `pr-<num>` tag: same hash → reuse it, different or absent → rebuild. Forks inspect the public tag without repository secrets |
| `docker-build` | builds `cu13` for `linux/amd64` and `linux/arm64` and pushes `pr-<num>`, stamped with the inputs hash. Same-repository rebuilds use a `docker-build` node. Fork rebuilds wait on a hosted runner for `Build Fork CI Image` to return success; otherwise this job is a hosted-runner no-op |
| `resolve-ci-image` | resolves the CI image to `pr-<num>` whenever that tag is current — whether this run built it or an earlier one did — so **every GPU suite runs inside the PR's image**; a failed build stops the matrix instead of testing a stale image. A PR image outranks a `ci-image-tag:` PR-body directive, which applies only when the PR has no image (PRs whose image inputs match the base) |
| `delete-pr-tag` (`docker-pr-tag-cleanup.yml`) | removes the `pr-<num>` tag when the PR closes; the tag stays available for re-runs while the PR is open |

So a rerun, or a push that touches only source files, reuses the image the PR already has instead of rebuilding an identical one. Non-docker PRs are untouched: no PR image, matrix on `dev`, as before.

`docker/image_inputs.py` is the single source of truth for what counts as an input. Its default input set serves CUDA; `--variant rocm10-mi35x-pr` selects the separate ROCm input set. ROCm-only inputs do not change the CUDA hash.

To rebuild an eligible PR image when its inputs did not change — a moved base image, a floating dependency, a corrupt push — add the **`rebuild-ci-image`** label. The label does not select tests or make a PR whose build inputs match the base eligible. Once consumed by a build, it is removed so later runs can reuse the image.

### ROCm PR images

When ROCm stages are selected, `pr-test-rocm.yml` calls the same image workflow
with `rocm10-mi35x-pr`. It compares the ROCm inputs against the merge's first
parent, reuses an image with the current inputs label, and builds changed inputs
on Miles' Spur compute nodes through the trusted publisher. Dockerfile RUN steps use no AMD
devices; downstream shards pull the published image after verification. The tag is
`radixark/miles:pr-<number>-rocm10-mi35x`; its recipe derives from
`rocm10-mi35x`, so the existing rolling ROCm repository stays unchanged.

`tests/ci/rocm.py` owns image resolution: the reusable call's `image_tag`, the
dispatch `ci_image_tag`, a current PR image, the PR body's
`ci-rocm-image-tag:`, then the released default. Only the PR tag family uses
`radixark/miles`; existing ROCm tags use `rocm/sgl-dev`. `ci-image-tag:` continues
to select CUDA images. A failed build or fork publication blocks image resolution
and the AMD suites. The `rebuild-ci-image` label applies to eligible ROCm builds
as it does to CUDA.
The label is shared: a faster build may consume it before the other hardware's
image decision, so one label does not guarantee that both variants are forced.

Both same-repository and fork ROCm rebuilds use the trusted publisher below, with the source workflow bound
to `pr-test-rocm.yml` and the requested recipe restricted to the ROCm PR variant.
ROCm retains its existing lack of a CPU A gate; CUDA's CPU gate is unchanged.
The publisher verifies the ROCm image's linux/amd64 config and inputs hash.
CUDA and ROCm builds for one PR have separate publication concurrency groups.
Automatic changed-file ROCm jobs also wait for this image and use the resolver's
output, including when no regular stage is selected. Trusted fork demand checks
recompute those file jobs from the frozen diff.
Comment ROCm file reruns check the public tag anonymously through Docker Hub's
HEAD tag API. Only HTTP 404 selects the released default; authentication, rate
limit, network and service errors fail the lookup. Hosted Docker login remains
on the CUDA path.

Closing a same-repository PR removes its CUDA, TPU and ROCm tags. For a fork,
`docker-pr-tag-cleanup.yml` uses a closed-only `pull_request_target` event to
delete just its fixed ROCm tag, using the default-branch workflow with no checkout
or execution of PR code. This trusted cleanup and the trusted publisher changes
must reach the default branch before fork ROCm publication can work.

### Fork PR publication

Fork `pull_request` runs do not receive Docker Hub secrets. Their image job uses the same CPU A success/bypass gate, hash decision, and output contract as same-repository PRs. When a rebuild is needed, it submits an attempt-bound request and waits for trusted publication. Success continues to any selected CUDA tests in the same CI attempt; a publication failure fails the image job. CPU tests are not rerun.

`build-fork-ci-image.yml` starts on `PR Test` or `PR Test (ROCm)`'s `workflow_run.in_progress` event and waits for the request artifact, which is readable before the source run finishes. GitHub can fire that event several times for one attempt; only the earliest triggered run publishes, and later ones exit without a request. Only a request allocates a builder.

ROCm requests from same-repository PRs use this handoff too; CUDA uses it for forks only. The caller waits for that earliest publisher run and its build job to succeed, so an existing tag cannot satisfy a forced rebuild prematurely. The hosted handoff has a 360-minute limit covering queue and build time; the builder retains its 180-minute limit.

The trusted workflow binds the request to its source run and attempt, the open PR and current head, the latest matching PR Test run, and the original merge SHA recorded by GitHub for `_build-pr-ci-image.yml`. It resolves fast-fail policy and ROCm test selection from the request's original PR-event labels, matching the caller even after labels are removed. These labels are policy input from the source workflow, not proof of publishing authority; credential isolation is described below.

It checks CUDA's CPU gate, including successful jobs retained by partial reruns, and recomputes that the variant's build inputs differ from the base. ROCm additionally uses trusted code to parse the PR's test registrations without importing them and checks stage or changed-file demand; CUDA builds do not require selected GPU tests. A later main advance does not change the frozen merge.

Host orchestration comes from the default branch. Only the PR Dockerfile and build context enter an isolated BuildKit instance, with no secret forwarding or insecure entitlements. CUDA exports both architectures as a local OCI layout; ROCm exports linux/amd64 through the Spur build job described below. BuildKit is removed before Docker Hub login; the publisher copies the layout to its fixed PR tag without running the image and verifies the required platforms' inputs hashes.

CUDA publication uses a digest-pinned skopeo container; ROCm uses skopeo installed on the Spur login runner. Fork edits to Python orchestration, the hash reader, or workflow take effect after merge.

For ROCm version upgrades, the trusted helper parses the fork's literal `VARIANTS["rocm10-mi35x"]["build_args"]` without executing its Python, checks string keys and values, and passes them as final `--build-arg` overrides. Fork ROCm images allow literal build_args edits in any variant in docker/build.py, so a single PR can upgrade all ROCm variants; only rocm10-mi35x arguments feed the ROCm PR image. Any other code change relative to the frozen merge's first parent is refused. Formatting and comments do not change the compared AST. Computed arguments, malformed values, or removal of a trusted driver's default argument also fail explicitly.

A new request cancels an older build for the same PR. During building, the trusted helper checks every 30 seconds that the source attempt remains active and stops its build process group if it ends. A newer publisher cancels the older one; live-PR checks also reject a superseded request. For ROCm, stopping the launcher cancels its exact Slurm job and confirms it has left the queue. The publisher rechecks the live PR and source before publication. Builder and local-output cleanup also run on failure or cancellation. A forced build consumes its original one-shot label, with the same removal-failure warning as same-repository builds.

This path requires `build-fork-ci-image.yml` and its trusted helpers on the default branch. It cannot complete while introduced only on a PR branch. Existing `Approve Trusted CI` handles GitHub's contributor approval hold independently.

### Spur ROCm build and publication boundary

For ROCm, the trusted publisher runs on `linux-miles-mi35x-spur`, a login-node
runner. `spur/build_image.py` submits one exclusive compute-node job through
`sbatch`. Builds and ROCm tests read `spur/other-project-nodes.json` to exclude
the other projects' CI nodes.
The allocation occupies a whole Miles node, including its GPUs, but Dockerfile
RUN steps receive no GPU devices. This trades one node's availability for isolation from
concurrent tests; it does not make the login runner build the image.

The build launcher uses the runner's `SLURM_PARTITION`, `SLURM_ACCOUNT`,
`SLURM_QOS`, `CI_NODE_SCRATCH`, and optional `CI_SHARED_DIR` settings. Login and
compute nodes must share the checked-out workspace and output directory. The
compute node clones the frozen source into private scratch and copies only the
trusted driver/hash reader there. A job-local Python venv installs `typer`.
Docker and its Buildx plugin must already be installed on the compute node;
Python with venv support and Git are also required. The login runner needs
Python, GitHub CLI, Slurm commands, Docker/Buildx for manifest inspection, and
`skopeo`. The workflow checks these publishing tools before allocating a node.

Slurm receives `--export=ALL` with a newly constructed environment containing
only a fixed tool PATH and optional `SLURM_CONF`. It does not inherit runner
secrets, proxy credentials, `BASH_ENV`, or `BUILD_GITHUB_TOKEN`. This avoids
`--export=NONE`, which can load the user's login environment ([Slurm sbatch
reference](https://slurm.schedmd.com/sbatch.html)). The private manifest contains
source paths, hashes and literal arguments, with no authentication data.
Build subprocesses get an isolated HOME and empty Docker config. BuildKit uses
its own `docker-container` instance with bridge networking and no insecure
entitlements. Cleanup removes only that instance, its volume and private
scratch, and must succeed before the job writes a successful exit marker.
There is no global prune or test-container cleanup.

The compute node exports its OCI layout to the private shared run directory.
The login runner waits for the job to leave Slurm and for its successful exit
marker, then rechecks the live PR before publication. `DOCKERHUB_USERNAME` and
`DOCKERHUB_TOKEN` arrive on the login runner from Miles' repository Actions
secrets, only in the Docker login action after the build. Its config lives in a
mode-0700 node-local `/tmp/miles-image-auth.*` directory, outside the shared
workspace/output. `skopeo` copies the layout to the fixed PR tag without running
it; the publisher verifies the inputs hash before unblocking shards. Cleanup
removes the auth directory even if cluster cancellation fails. An interrupted
build is cancelled by its recorded numeric job ID. Cancellation sends
`scancel` before checking the queue and requires two consecutive empty replies
before shared files can be removed; one transient empty reply does not confirm
stoppage. The same two-read rule applies when recovering an ID by unique job
name. Query errors retain the shared files and fail cleanup. Missing completion
or failed cleanup blocks publication.

Publication uses the Docker Hub secret on the Spur login runner. Closed-PR tag
deletion is the explicit exception: the existing trusted hosted
`docker-pr-tag-cleanup.yml` uses the same secret, as it does for TPU. Anonymous
ROCm file lookups do not receive this secret.

A Docker Hub token that can write `radixark/miles` can generally overwrite other
tags in that repository; the fixed PR tag is enforced by trusted code, not a
per-tag registry permission. Use a dedicated repository-scoped write token.
The existing ROCm test launcher exports the runner's `.env` Docker Hub
credentials to test jobs. Keep those credentials pull-only; provision this
publishing write token as a repository Actions secret. An existing write token
in the runner's `.env` would remain exposed through that test path.
The login runner and its administrator remain trusted: root or another process
with the same Unix identity can read step environment and temporary auth files.
Mode-0700 shared directories do not isolate jobs running as that same identity,
and a rootful BuildKit daemon is not a hostile multi-tenant security boundary.
Repository secrets are withheld from fork source workflows, but a compromised
Spur host or shared runner account is outside this credential boundary.

Provision enough free Docker storage on the compute node and shared space for
the OCI output before enabling this path. A prior GPU-free cold build used
68.385 GiB of sampled BuildKit storage and produced a 48.790 GiB unpacked image;
the same-recipe published image was about 18.055 GiB compressed. These are
capacity references, not a bound on future PRs. The job has a two-hour Slurm
limit and a 150-minute queue/build wait within the publisher's 180-minute limit;
held jobs or missing completion fail instead of being treated as a usable image.
Spur submission and publication require a deployment smoke run after these
trusted helpers reach the default branch.

## Rolling Docker build (`docker-build.yml`)

This workflow owns the rolling `dev` / `latest` (CUDA) and `dev-tpu` / `latest-tpu` (TPU) image families. It has four jobs:

- **`check-upstream`** (schedule / `simulate_schedule` only) — polls the inputs the image bakes: the HEAD SHA of sglang `sglang-miles` (`sgl-project/sglang`) and Megatron-LM `miles-main` (`radixark/Megatron-LM`) — the source branches it builds — plus a fingerprint of the selected `radixark/miles-wheels` rolling release, so a rebuilt sgl-router or other wheel also triggers a build (re-uploads to the same tag are caught by fingerprint, not commit SHA). It compares against the values cached from the last build and sets `should_build=true` if any moved. `miles` itself is intentionally not polled — that would rebuild far too often. The cron polls every 10 minutes (off the top of the hour), so an upstream move starts a build within about 30 minutes, and this check is what stops those polls from rebuilding an unchanged image, with one staleness bound: because the image also bakes a `miles` checkout, `should_build` is forced to `true` once the last triggered build is **12h** old, so `dev` never drifts more than half a day behind the `miles` repo even when sglang / Megatron / wheels are quiet. A lookup that still fails after three attempts, or returns a malformed value, fails that poll (the run goes red) without building or saving state, and the next poll retries, so a lookup that keeps failing shows up as red runs rather than silently stopping rebuilds. (The cache file's last line records the epoch of the last triggered build; it is only re-saved when a build fires.)
- **`push-variants`** (push only) — diffs the push against its `before` commit and picks the variants whose inputs it touched: `cu13` for `docker/Dockerfile` and `docker/verify_transformer_engine.py`, `tpu` for `docker/Dockerfile.tpu` and `docker/tpu_patch/**`, both for `docker/install-kube-tools.sh` and `requirements.txt`. A TPU-only change therefore never rebuilds the multi-arch cu13 image.
- **`build-and-push`** (self-hosted `docker-build` runner) — one matrix job per variant (`cu13` and `tpu` on a schedule, the `push-variants` pick on a push, the picked one on a manual dispatch) calls `docker/build.py` to build + push, then conditionally points that variant's `latest` at its new `dev` and prunes its old timestamped tags. The TPU image pins its sglang-torchtpu, torch_tpu and torchtitan sources, so its only moving inputs are the baked miles checkout (bounded by the 12h rebuild) and the x86 wheels release (fingerprinted by `check-upstream`); it shares the cu13 gate.
- **`notify-build-failure`** — when an automatic build (schedule or push) fails, posts one card to the Miles CI Lark group with the failed job and step (`lark_notify.py docker-build-failure`). A failed manual dispatch is not reported.

`build-and-push` runs when `check-upstream` was skipped, or ran and reported `should_build=true`, and on a push only after `push-variants` succeeded. Automatic builds (schedule, push) share one concurrency group per variant, so each variant builds one at a time and at most one waits; a newer waiting build replaces an older one, and it still builds the newest heads because the Dockerfile clones them when the build starts. Manual dispatches stay outside that group.

### Triggers: automatic vs manual

- **Automatic** (no human) — the **schedule** (10-minute poll, gated by `check-upstream`) and any **push to `main` that touches `docker/Dockerfile`, `docker/Dockerfile.tpu`, `docker/tpu_patch/**`, `docker/install-kube-tools.sh`, `docker/verify_transformer_engine.py`, or `requirements.txt`**. The schedule builds `cu13` → `radixark/miles:dev` (multi-arch) and `tpu` → `radixark/miles:dev-tpu` (amd64); a push builds only the variants whose inputs it touched (see `push-variants` above).
- **Manual** — `workflow_dispatch` (pick one variant — see Trigger a build yourself below) or running `docker/build.py` locally. The rolling `rubin` and ROCm images (`rocm*-mi35x`, `rocm10-mi30x`) have **no automatic rolling build in this repo** (ROCm nightlies live in sgl-project/sglang); ROCm PR builds are described above (`cu13-x86` / `cu13-aarch64` just rebuild the same `dev` image single-arch).

`docker/build.py` and `docker/patch/**` participate in PR image validation but are not `main`-push triggers; [Release a Version](/developer/ci/04-release) treats them as manual preflight cases.


| Trigger                                     | `check-upstream`                   | builds                | `latest` move     | prune      |
| ------------------------------------------- | ---------------------------------- | --------------------- | ----------------- | ---------- |
| schedule (10-minute poll)                   | runs; build if upstream moved or last build ≥ 12h ago | `cu13`, `tpu` | yes        | yes |
| push to `main` touching `docker/Dockerfile`, `docker/Dockerfile.tpu`, `docker/tpu_patch/**`, `docker/install-kube-tools.sh`, `docker/verify_transformer_engine.py`, or `requirements.txt` | skipped                            | the touched variants (`push-variants`) | no                | no         |
| `workflow_dispatch`                         | skipped                            | the one input variant | no                | no         |
| `workflow_dispatch` + `simulate_schedule`   | runs                               | the one input variant | no                | no         |


### Tags & where it pushes

All images push to **Docker Hub**. CUDA, TPU and the ROCm PR variant use `radixark/miles`; rolling ROCm variants use `rocm/sgl-dev`. The tag a build writes depends only on `--image-tag` and the variant's postfix:

| variant(s) | `--image-tag dev` writes | `latest` mode writes |
| --- | --- | --- |
| `cu13` / `cu13-x86` / `cu13-aarch64` | `radixark/miles:dev` + `radixark/miles:dev-<YYYYMMDDHHMM>` | `radixark/miles:latest` |
| `rubin` | `radixark/miles:dev-rubin` (+ timestamped sibling) | `radixark/miles:latest-rubin` |
| `tpu` | `radixark/miles:dev-tpu` + `radixark/miles:dev-tpu-<YYYYMMDDHHMM>` | `radixark/miles:latest-tpu` |
| `rocm724-mi35x` / `rocm10-mi35x` | `rocm/sgl-dev:miles-rocm*-mi35x` (+ timestamped sibling) | `rocm/sgl-dev:latest-rocm*-mi35x` |
| `rocm10-mi30x` | `rocm/sgl-dev:miles-rocm10-mi30x` (+ timestamped sibling) | `rocm/sgl-dev:latest-rocm10-mi30x` |

What **moves a shared tag**: `--image-tag dev` overwrites `:dev` and adds a timestamped sibling; on a **scheduled** run `latest`→`dev` and `latest-tpu`→`dev-tpu` advance; pruning likewise runs **only on schedule**, keeping the newest 20 `dev-<ts>` and, separately, `dev-tpu-<ts>` tags and never deleting a tag younger than 14 days. Any `workflow_dispatch` — **including** `simulate_schedule` — writes its own tag(s) but never moves `latest` or prunes; only the real cron mutates published tags. See the trigger table above.

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
| `variant` | yes | `cu13` / `cu13-x86` / `cu13-aarch64` / `rubin` / `tpu` / `rocm724-mi35x` / `rocm10-mi35x` / `rocm10-mi30x` |
| `image_tag` | yes | `dev` / `latest` / `custom` |
| `custom_tag` | no | tag name; required when `image_tag=custom` |
| `dockerfile` | no | path to Dockerfile (default `docker/Dockerfile`) |
| `simulate_schedule` | no | `true` runs the `check-upstream` poll first (default `false`) |

### Steps (`build-and-push`)

1. checkout → Buildx → install Python + typer → Docker Hub login.
2. **Build + push** via `build.py` — a schedule builds `cu13` and `tpu` as two matrix jobs, a push only the variants it touched, and a manual dispatch builds only the one variant you picked.
3. **schedule only** — point `latest`→`dev` (`latest-tpu`→`dev-tpu` for `tpu`).
4. **schedule only** — prune that variant's `dev-<timestamp>` or `dev-tpu-<timestamp>` series to the newest 20, keeping every tag younger than 14 days.

### Push auth & permissions

Pushes use a Docker Hub credential, not your identity:

- **Remote (CI)** — the workflow logs in with repo secrets `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`, so you don't hold the key — you just trigger the run, which needs repo **Write** access. No approval gate, but `build-and-push` runs on a `self-hosted` runner, so it only fires when one is online.
- **Local** — `build.py --push` uses your own `docker login`; you need push rights to the target namespace (`radixark/miles`, or `rocm/sgl-dev` for ROCm).

## Wheels build (`build-wheels.yml`)

The independent [Wheels rebuild workflow](/developer/ci/06-build-wheels) publishes the prebuilt assets this image consumes. Its scheduling, manual inputs, build stages, and publishing credentials are documented separately.

## Versioned release build (`release-docker.yml`)

`release-docker.yml` checks out the supplied release ref, requires its committed `release-lock.json`, and passes the locked SGLang and Megatron-LM commits plus the checked-out Miles SHA as final `--build-arg` overrides. Rolling `docker-build.yml` keeps the branch defaults. Published tags and the guarded dispatch and retry procedure are documented in [Release a Version](/developer/ci/04-release).

## Image retention (open)

`docker-build.yml` prunes `dev-<timestamp>` and `dev-tpu-<timestamp>`, keeping the newest 20 of each plus every tag younger than 14 days; `dev` / `latest` and `dev-tpu` / `latest-tpu` move forward. Ordinary PR, nightly, and weekly CI therefore has no durable image record. A release branch cut instead retags the newest timestamped `dev` image, or mutable `dev` when none exists, as prune-exempt `release-vX.Y.Z-ci` and records that tag in `release-lock.json`. The guarded preflight is documented in [Release a Version](/developer/ci/04-release).
