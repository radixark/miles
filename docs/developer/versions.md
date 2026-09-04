---
title: Versions and Images
description: How the Miles, SGLang, and Megatron-LM trees fit together, which artifact owns each release identity, and what the Docker images pin.
---
A Miles run is three Python source trees on one `PYTHONPATH`. Understanding which tree owns
a given behavior, and which file pins its version, is most of what you need to debug a
version problem or land a bump.

## The three trees

| Tree | Comes from | Installed as | Lives at |
|---|---|---|---|
| Miles | this repository | `pip install -e . --no-deps` | `/root/miles` |
| SGLang | the `sglang-miles` branch of [`sgl-project/sglang`](https://github.com/sgl-project/sglang), on top of the `lmsysorg/sglang:<tag>` base image | `pip install -e "python[all]" --no-deps` | `/sgl-workspace/sglang` |
| Megatron-LM | the `miles-main` branch of [`radixark/Megatron-LM`](https://github.com/radixark/Megatron-LM) | `pip install -e .` | `/root/Megatron-LM` |

Two things follow from that table.

**The dependency branches are moving development sources.** Both carry patches Miles needs before they land upstream, which is why installing stock `sglang` or upstream Megatron-LM next to Miles does not work. Ordinary `dev` images follow those branches; a versioned release records their exact commits in `release-lock.json`.

**All three are editable installs.** Nothing is copied into `site-packages`, so changing a
file in any of the three trees changes the next run. That is also what lets CI move the
SGLang and Megatron-LM checkouts to a different ref without reinstalling anything.

The `version` in `setup.py` owns the base Miles release version `X.Y.Z`. Release candidates and post releases keep that base in `setup.py` and add `rcN` or `.postN` only to the exact Git and Docker tags.

## Where each pin is written

```text
docker/Dockerfile     the image recipe and its default build-args
docker/build.py       the variant table: which build-args each variant overrides
requirements.txt      Miles' own Python dependencies
release-lock.json     generated on a release branch: dependency commits, CI image tag, and audit fingerprints
```

The default build-args are the version surface:

| Build-arg | Default | What it selects |
|---|---|---|
| `SGLANG_IMAGE_TAG` | `v0.5.16` | The `lmsysorg/sglang` base image, which brings torch, CUDA and Transformer Engine |
| `SGLANG_BRANCH` | `sglang-miles` | The branch fetched into the base image's SGLang checkout |
| `SGLANG_COMMIT` | empty | Empty means the branch HEAD at build time; set it to freeze one commit |
| `MEGATRON_REPO` / `MEGATRON_BRANCH` / `MEGATRON_COMMIT` | `radixark/Megatron-LM` / `miles-main` / empty | The Megatron-LM checkout; an empty commit follows branch HEAD, while a release build supplies the locked commit |
| `MILES_COMMIT` | `main` | The Miles checkout baked into the image |
| `ENABLE_CUDA_13` | `1` | CUDA 13 plus the Mooncake structured-object-store wheel; `0` selects the CUDA 12.9 path |
| `WHEELS_REPO` | `yueming-yuan/miles-wheels` | The prebuilt-wheels repository |
| `WHEELS_TAG_X86` / `WHEELS_TAG_ARM64` | `cu130-torch213-x86_64` / `cu130-torch213-aarch64` | Two complete wheels releases, selected by `TARGETARCH` and installed verbatim |

Two design choices are worth naming. The Dockerfile holds the defaults and `build.py` owns
the per-variant deltas, so there is exactly one place to look for what a variant changes.
And a wheels release is installed as a whole release, with no tag assembled at build time,
so the set of kernels in an image is one auditable name rather than a computed string.

Everything else is pinned inline where it is installed: `mbridge` and `Megatron-Bridge` and
`torch_memory_saver` at explicit commits, `flash-linear-attention`, `tilelang` and friends at
explicit versions. Transformer Engine is special: `docker/verify_transformer_engine.py`
asserts the installed triplet is `2.17.0`, and the patches under `docker/patch/cu13/` are
applied to it with a build failure if any patch does not apply cleanly, so an image can
never ship silently unpatched TE.

`requirements.txt` is Miles' own dependency list, and the convention there is that **a pin
carries its reason inline**: `transformers==5.12.1` names the HF-native weight conversion and
an SGLang collision, `blake3` / `xxhash` / `zstandard` name the disk-delta weight sync,
`onnxscript` records that TE 2.17 imports it at import time. Follow that when you add one.

## Release identities and source ownership

Release values represent different facts. Each row below names the source used by the supported automatic path.

| Fact | Authoritative source | Derived or consumed values |
|---|---|---|
| Base Miles version | `setup.py` | Release branch `release/vX.Y.Z`; base-version check before tagging |
| Exact published version | `version` input to `release-tag.yml`, persisted as annotated Git tag `v<exact-version>` | CUDA image tags `v<exact-version>` and `v<exact-version>-cu12` |
| Frozen dependency selection | `release-lock.json` committed on the release branch | SGLang commit, Megatron-LM commit, and the CUDA image tag used by release CI |

For example, base version `0.3.0` owns branch `release/v0.3.0`; that branch can produce exact tags `v0.3.0rc0`, `v0.3.0`, and `v0.3.0.post1` without changing `setup.py` between tags.

`release-lock.json` freezes the two dependency source commits and names the prune-exempt CUDA image used by release CI. Miles itself is frozen by the release commit and final Git tag. Wheels asset fingerprints are audit records, not content-addressed inputs.

Use [Release a Version](/ci/04-release) for the maintainer procedure. That runbook owns workflow order, inputs, success signals, and recovery; this page owns the identity and pinning model.

## The images

CUDA variants are published to [`radixark/miles`](https://hub.docker.com/r/radixark/miles)
on Docker Hub, ROCm variants to `rocm/sgl-dev`. The
[`dev` tag](https://hub.docker.com/r/radixark/miles?tag=dev) is what CI runs on unless a job
says otherwise, so its last-pushed timestamp is the fastest way to see how current the
fleet's image is.

| Variant | Tag written by `--image-tag dev` | Platforms |
|---|---|---|
| `cu13` | `radixark/miles:dev` | `linux/amd64` + `linux/arm64`, one manifest. This is the daily image |
| `cu13-x86` / `cu13-aarch64` | `radixark/miles:dev` | Single-arch rebuilds of the same image |
| `cu12-x86` | `radixark/miles:dev-cu12` | `linux/amd64`, CUDA 12.9 legacy |
| `rocm720-mi35x` / `rocm10-mi35x` | `rocm/sgl-dev:miles-rocm*-mi3xx` | Native |

`--image-tag dev` also publishes a timestamped sibling. Scheduled retention and manual tag behavior are documented in [Docker build](/ci/02-docker-build).

Build one yourself with `docker/build.py`:

```bash
python docker/build.py --variant cu13-x86 --image-tag custom --custom-tag my-experiment --push
```

[Docker build](/ci/02-docker-build) is the full reference for the build script, the
workflow and the tag rules.

Official versioned releases add `radixark/miles:v<exact-version>` for the CUDA 13 multi-arch image and `radixark/miles:v<exact-version>-cu12` for the CUDA 12.9 image. Publishing them does not move the rolling `dev` or `latest` families.

## What CI moves, and what it does not

This is the part that decides whether your change needs a new image. A CUDA CI job starts
from `radixark/miles:<tag>` and then:

1. Runs `pip install -r requirements.txt`, then restores the image's own cuDNN pin. The
   restore is not cosmetic: TE's fused attention needs a newer cuDNN than torch pins, a
   plain resolve drags it back down, and the symptom is a fused-attention backward failing
   with `CUDNN_STATUS_BAD_PARAM`.
2. Resets both dependency checkouts and fetches the selected refs. Explicit dispatch or PR-body overrides win first, `release-lock.json` commits win when no override exists, and the moving `sglang-miles` / `miles-main` heads are the final defaults.
3. Sets `PYTHONPATH` to the Miles workspace plus both source roots.

It never reinstalls the three source trees, because they are editable installs. So:

| Your change | Needs a new image? |
|---|---|
| `requirements.txt` | No. The next CI run installs it. |
| A Dockerfile layer: a pinned wheel, an inline commit, a TE patch, the base image | Yes |
| SGLang or Megatron-LM code | No. Point CI at a ref instead |
| Miles code | No |

The ROCm stage is the exception: it takes SGLang and Megatron-LM from `rocm/sgl-dev` and exposes no dependency-ref inputs, so the only way to move them there is a new ROCm image. A release call can select the Miles ref, but its baked dependencies still make the run a smoke signal rather than a lock-accurate check.

## Bumping principle

**Bump where the pin lives, exactly once.** A Python dependency moves in
`requirements.txt`; an image layer moves in `docker/Dockerfile`; a variant-only difference
moves in `docker/build.py`. If a bump needs edits in two of the three, one of them is in the
wrong place.

**Prefer moving the branch to pinning a commit during rolling development.** `SGLANG_COMMIT` and `MEGATRON_COMMIT` are empty by default, so ordinary images follow `sglang-miles` and `miles-main` together. A versioned release is the deliberate exception: its lockfile supplies both exact commits to CI and the final image build.

**Validate an image-affecting bump on the PR that makes it.** A PR touching
`docker/Dockerfile`, `docker/build.py`, `docker/verify_transformer_engine.py`,
`docker/patch/**` or `requirements.txt` builds a `pr-<number>` image first and runs every
GPU suite inside it, and that fresh tag outranks a `ci-image-tag:` directive so the PR
cannot accidentally test the old image. Add the `run-ci-image` label to widen the selection
to every tag except the long-running and fault-tolerance ones.

**Validate a dependency-ref bump without building anything.** Put the ref in the PR
description and CI checks out that instead:

```text
ci-sglang-pr: #12345
ci-megatron-pr: my-fix-branch
```

`#12345` resolves to `refs/pull/12345/head`, so an unmerged upstream fix can be tested
before it lands on `sglang-miles`. See
[Contributing](/developer/contributor-guide#pr-description-ci-tags) for all three
directives.

**Expect `dev` to move on its own, within a bound.** The scheduled build (00:00 and 12:00
UTC) polls the SGLang and Megatron-LM branch heads plus a fingerprint of the wheels release,
and rebuilds when any of them moved. It deliberately does not poll Miles, which would
rebuild constantly, and instead forces a build once the last one is 24 hours old. So `dev`
follows its dependencies immediately and trails Miles `main` by at most a day. When you need
that to stop moving underneath you, pin `ci-image-tag:` to a timestamped tag.

**Bump the ROCm images by hand.** They have no automatic path, so a ROCm bump is a
`workflow_dispatch` on the variant you want.

## After a bump, the usual suspects

| Symptom | Look at |
|---|---|
| `CUDNN_STATUS_BAD_PARAM` in a fused-attention backward | Something re-resolved cuDNN below the image's pin |
| Build fails with "TE patch did not apply cleanly" | A `docker/patch/cu13/*.patch` no longer matches the new TE; rebase the patch or drop it if upstream fixed it |
| TE triplet assertion at build time | The base image moved TE off `2.17.0`; update `docker/verify_transformer_engine.py` together with whatever depends on it |
| `mooncake.structured_object_store` import fails | A CUDA 12 image; that wheel is only installed on the cu13 path |
| A test passes locally but fails in CI, or the reverse | Compare the image tag and the two dependency refs or commits CI resolved. Every job logs all three |

## Related

- [Installation](/getting-started/installation) for pulling and running the image.
- [Release a Version](/ci/04-release) for the maintainer release runbook.
- [Contributing](/developer/contributor-guide) for the CI labels and PR-description
  directives.
- [Docker build](/ci/02-docker-build) for the build script, workflow triggers and tag
  mechanics.
