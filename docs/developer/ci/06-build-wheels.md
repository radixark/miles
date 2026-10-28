---
title: Wheels rebuild
description: Rebuild and publish router, int4_qat, Transformer Engine, and sglang wheels with the independent build-wheels workflow.
---
`build-wheels.yml` builds prebuilt wheel releases independently of the [Docker image workflow](/developer/ci/02-docker-build). The image workflow detects release changes and rebuilds the consuming image.

## Wheel sets

Four `radixark/miles-wheels` asset sets follow a moving source. Each build writes a `<dist>-source.json` release asset recording the commit it came from, and `build-wheels.yml` rebuilds a set only when its source has moved past that commit:

| set | source | x86_64 | aarch64 |
| --- | --- | --- | --- |
| `sgl-router` (wheel + `sgl-model-gateway` binary) | `radixark/sgl-router-for-miles` `main` | GitHub-hosted `ubuntu-24.04` | GitHub-hosted `ubuntu-24.04-arm` |
| `int4_qat` (`fake_int4_quant_cuda`) | last `main` commit touching `miles/kernels/quant/int4_fake` | `docker-build` runner, in the SGLang base image | by hand |
| `te` (Transformer Engine triplet, `<release>+miles`) | `radixark/TransformerEngine` `miles-main`: NVIDIA's release commit plus the fixes Miles carries | `docker-build` runner | by hand |
| `sglang` (`<tag>.post<n>+miles.g<commit>`) | `sgl-project/sglang` `sglang-miles` | GitHub-hosted `ubuntu-22.04` | GitHub-hosted `ubuntu-22.04-arm` |

`sglang` is the wheel a pip-installed Miles depends on, not part of the image, so it goes to its own `sglang-miles-<ISO week>-<arch>` releases (e.g. `sglang-miles-2026w43-x86_64`; one per week keeps each under GitHub's 1000-asset cap). A wheel is built once per commit, arch and Python version and never replaced, because a Miles wheel pins it (with its hash); the rolling releases the image installs never see it. So `sglang` is rebuilt only for a (commit, arch, Python) that no `sglang-miles-*` release holds yet, not by its manifest: a new week does not rebuild unchanged code, a Python version that failed is retried next hour, and `force` does not apply to it. It is built for every Python version Miles installs on (`SGLANG_PYTHONS`, 3.11 and 3.12), since its Rust extensions make the wheel per-interpreter. The glibc 2.35 runners cap the wheel's tag at `manylinux_2_35`, so it installs on Ubuntu 22.04 and newer.

## Publish setup

Use the shared `radixark-miles-ci` GitHub App installed on `radixark/miles` and `radixark/miles-wheels`. Grant `Contents: write` for wheel publishing and `Issues: write` plus `Pull requests: write` for [label commands](/developer/ci/01-label). Store its client ID in the `radixark/miles` repository variable `CI_APP_CLIENT_ID` and its private key in the repository secret `CI_APP_PRIVATE_KEY`; both workflows use the same credentials.

The publish job requests only `Contents: write` on `radixark/miles-wheels`. `CI_APP_ENABLED` controls label commands and does not gate wheel publishing. When migrating existing configuration, add the new names before deploying the workflows and retain the old entries until workflows on the default branch and active runs no longer reference them.

## Schedule and manual runs

It runs hourly and by `workflow_dispatch` (`wheels`, `force`, `te_ref`, `keep_superseded`). `docker/Dockerfile` installs the `+miles` TE triplet, so scheduled runs rebuild TE like the other sets, and an upload replaces superseded wheels unless `keep_superseded` is true. There is no self-hosted ARM64 runner, so aarch64 `int4_qat` and `te` are rebuilt by hand with the miles-wheels README commands.

```bash
# Rebuild stale TE wheels.
gh workflow run build-wheels.yml --repo radixark/miles -f wheels=te

# Rebuild the router even when its release manifest already matches.
gh workflow run build-wheels.yml --repo radixark/miles -f wheels=sgl-router -f force=true
```

The hourly schedule runs at minute 17. Manual dispatch defaults to `wheels=all`, `force=false`, `te_ref=miles-main`, `sglang_ref=sglang-miles`, and `keep_superseded=false`. `wheels` also accepts `sgl-router`, `int4_qat`, `te`, or `sglang` to select one set. `te_ref` and `sglang_ref` select the Transformer Engine or SGLang branch or commit; dispatch with a release branch's `release-lock.json` `sglang_commit` when the hourly run did not build that commit. Runs are serialized and GitHub keeps only one pending run, so a dispatch queued behind a long build can be replaced by the next hourly run; check that it ran.

The `check` job summary lists each selected source commit and which architectures need rebuilding. A matching release manifest leaves the build unselected unless `force=true`.

## Build and publish flow

- **`check`** — resolves only the selected sources and reads the commit their manifests record in the `WHEELS_TAG_X86` / `WHEELS_TAG_ARM64` release. A lookup failure for a selected source or release fails the job; for `sglang` it only warns under `wheels=all`, so the image's sets still build. For `sglang` it lists which of the commit's wheels any `sglang-miles-*` release already holds. It takes the repository, tags and `SGLANG_IMAGE_TAG` from `docker/Dockerfile`, so assets land in the releases the image installs and compile against its torch.
- **`build-router`**, **`build-int4-x86`**, **`build-te-x86`** — build only the stale sets. The router job also checks that the wheel and the binary forward a `/generate` request. `te` runs NVIDIA's manylinux recipe on the host, then compiles `transformer_engine_torch` inside the SGLang base image (`lmsysorg/sglang:<SGLANG_IMAGE_TAG>`).
- **`build-sglang`** (GitHub-hosted) — builds the sglang wheel with the `sglang` step of `miles-wheels/build_wheels.py`, which follows upstream's own wheel recipe and smoke-tests the Rust extension. **`publish-sglang`** uploads it into the week `check` picked using the same CI App token as `publish`, keeping any wheel the release already holds. Each arch publishes on its own, with every Python version that built, so one failed build does not hold back the others.
- **`publish`** (GitHub-hosted) — when no selected build failed, syncs each arch's assets and manifests into its release with a CI App token (`vars.CI_APP_CLIENT_ID` / `secrets.CI_APP_PRIVATE_KEY`) minted with only `Contents: write` on `radixark/miles-wheels`. The release fingerprint in `check-upstream` then rebuilds the image.

- **`notify-build-failure`** (GitHub-hosted) — after the check, build, and publish jobs finish, posts one failure card to the Miles CI Lark group if any failed. Scheduled and manual runs in `radixark/miles` use the existing repository secret `LARK_WEBHOOK`; the card names the failed jobs and steps and links to the run. Successful, skipped, or cancelled jobs alone do not trigger a notification.

Runs are serialized across the whole workflow to prevent overlapping source snapshots from replacing newer assets. A long manual TE build can therefore delay the hourly router/int4_qat runs. A failed selected build blocks publishing for both architectures.
