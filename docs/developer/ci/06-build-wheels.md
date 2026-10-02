---
title: Wheels rebuild
description: Rebuild and publish router, int4_qat, and Transformer Engine wheels with the independent build-wheels workflow.
---
`build-wheels.yml` builds prebuilt wheel releases independently of the [Docker image workflow](/developer/ci/02-docker-build). The image workflow detects release changes and rebuilds the consuming image.

## Wheel sets

Three `radixark/miles-wheels` asset sets follow a moving source. Each build writes a `<dist>-source.json` release asset recording the commit it came from, and `build-wheels.yml` rebuilds a set only when its source has moved past that commit:

| set | source | x86_64 | aarch64 |
| --- | --- | --- | --- |
| `sgl-router` (wheel + `sgl-model-gateway` binary) | `radixark/sgl-router-for-miles` `main` | GitHub-hosted `ubuntu-24.04` | GitHub-hosted `ubuntu-24.04-arm` |
| `int4_qat` (`fake_int4_quant_cuda`) | last `main` commit touching `miles/backends/megatron_utils/kernels/int4_qat` | `docker-build` runner, in the SGLang base image | by hand |
| `te` (Transformer Engine triplet, `<release>+miles`) | `radixark/TransformerEngine` `miles-main`: NVIDIA's release commit plus the fixes Miles carries | `docker-build` runner | by hand |

## Publish setup

Use the shared `radixark-miles-ci` GitHub App installed on `radixark/miles` and `radixark/miles-wheels`. Grant `Contents: write` for wheel publishing and `Issues: write` plus `Pull requests: write` for [label commands](/developer/ci/01-label). Store its client ID in the `radixark/miles` repository variable `CI_APP_CLIENT_ID` and its private key in the repository secret `CI_APP_PRIVATE_KEY`; both workflows use the same credentials.

The publish job requests only `Contents: write` on `radixark/miles-wheels`. `CI_APP_ENABLED` controls label commands and does not gate wheel publishing. When migrating existing configuration, add the new names before deploying the workflows and retain the old entries until workflows on the default branch and active runs no longer reference them.

## Schedule and manual runs

It runs hourly and by `workflow_dispatch` (`wheels`, `force`, `te_ref`, `keep_superseded`). While `TE_ON_SCHEDULE` is false, scheduled runs skip `te` and every TE upload keeps older TE wheels, even when `keep_superseded` is false, because `docker/Dockerfile` still installs NVIDIA's `2.17.0`. Enable scheduled TE builds only with the Dockerfile change that installs the `+miles` wheels. `keep_superseded` remains an explicit override for other wheel sets. There is no self-hosted ARM64 runner, so aarch64 `int4_qat` and `te` are rebuilt by hand with the miles-wheels README commands.

```bash
# Rebuild stale TE wheels, keeping the version the Dockerfile still installs.
gh workflow run build-wheels.yml --repo radixark/miles -f wheels=te

# Rebuild the router even when its release manifest already matches.
gh workflow run build-wheels.yml --repo radixark/miles -f wheels=sgl-router -f force=true
```

The hourly schedule runs at minute 17. Manual dispatch defaults to `wheels=all`, `force=false`, `te_ref=miles-main`, and `keep_superseded=false`; unlike the schedule, `all` includes TE. `wheels` also accepts `sgl-router`, `int4_qat`, or `te` to select one set. `te_ref` selects the Transformer Engine branch or commit.

The `check` job summary lists each selected source commit and which architectures need rebuilding. A matching release manifest leaves the build unselected unless `force=true`.

## Build and publish flow

- **`check`** — resolves only the selected sources and reads the commit their manifests record in the `WHEELS_TAG_X86` / `WHEELS_TAG_ARM64` release. Scheduled runs also skip resolving TE while `TE_ON_SCHEDULE` is false. A lookup failure for a selected source or release fails the job. It takes the repository, tags and `SGLANG_IMAGE_TAG` from `docker/Dockerfile`, so assets land in the releases the image installs and compile against its torch.
- **`build-router`**, **`build-int4-x86`**, **`build-te-x86`** — build only the stale sets. The router job also checks that the wheel and the binary forward a `/generate` request. `te` runs NVIDIA's manylinux recipe on the host, then compiles `transformer_engine_torch` inside the SGLang base image (`lmsysorg/sglang:<SGLANG_IMAGE_TAG>`).
- **`publish`** (GitHub-hosted) — when no selected build failed, syncs each arch's assets and manifests into its release with a CI App token (`vars.CI_APP_CLIENT_ID` / `secrets.CI_APP_PRIVATE_KEY`) minted with only `Contents: write` on `radixark/miles-wheels`. Each wheel set is uploaded separately so TE can retain its pinned version without retaining old router or int4_qat wheels. The release fingerprint in `check-upstream` then rebuilds the image.

- **`notify-build-failure`** (GitHub-hosted) — after the check, build, and publish jobs finish, posts one failure card to the Miles CI Lark group if any failed. Scheduled and manual runs in `radixark/miles` use the existing repository secret `LARK_WEBHOOK`; the card names the failed jobs and steps and links to the run. Successful, skipped, or cancelled jobs alone do not trigger a notification.

Runs are serialized across the whole workflow to prevent overlapping source snapshots from replacing newer assets. A long manual TE build can therefore delay the hourly router/int4_qat runs. A failed selected build blocks publishing for both architectures.
