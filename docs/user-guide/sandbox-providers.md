---
title: Sandbox Providers
description: Credentials and endpoints for the cloud sandbox providers a recipe can run its tasks on.
---

Recipes that give each task its own container get it from a sandbox provider.
Rows are providers, columns the connectors that can use them; a filled cell has
had a real rollout run on it. Each column header links to the connector's
recipe, and a check carries its own link where that pair runs on a different
one.

| Sandbox provider | [Harbor](https://github.com/radixark/miles/tree/main/examples/experimental/harbor) | [HUD](https://github.com/radixark/miles/tree/main/examples/experimental/hud) | [NeMo Gym](https://github.com/radixark/miles/tree/main/examples/experimental/nemo-gym) | [OpenEnv](https://github.com/radixark/miles/tree/main/examples/experimental/openenv) |
|---|:---:|:---:|:---:|:---:|
| [AgentENV](https://github.com/kvcache-ai/AgentENV) | ✓ | | | [✓](https://github.com/radixark/miles/tree/main/examples/experimental/agentenv) |
| [Daytona](https://www.daytona.io/) | [✓](https://github.com/radixark/miles/tree/main/examples/experimental/swe-agent-harbor-daytona) | ✓ | ✓ | ✓ |
| [E2B](https://e2b.dev/) | ✓ | | | ✓ |
| [Modal](https://modal.com/) | | | | ✓ |

The rest of this page is how to set each provider up. Its SDK is installed per
recipe; the recipe README has that line.

Every provider takes its credential the same two ways: exported in the
environment, or in a key file. On a multi-host cluster the file has to be
readable where the rollout workers run — a shared filesystem, or the same path
on every node — because the launcher hands the workers the file's *path*, never
its contents.

## E2B

E2B Cloud, or any server that speaks the E2B API.

```bash
mkdir -p ~/.config/e2b && echo e2b_... > ~/.config/e2b/api_key   # or export E2B_API_KEY
```

With nothing else set, the SDK talks to E2B Cloud. For a server of your own,
point both planes at it:

```bash
export E2B_API_URL=http://<server>:8000        # control plane
export E2B_SANDBOX_URL=http://<server>:8000    # data plane
```

[AgentENV](https://github.com/kvcache-ai/AgentENV) is one such server —
Firecracker microVMs behind the E2B API — and deploying one is
[its own guide](https://github.com/radixark/miles/tree/main/examples/experimental/agentenv).

E2B Cloud builds templates as a non-root user, so a task Dockerfile whose `RUN`
layers need root fails to build there. Terminal-Bench-2 tasks ship prebuilt
images and are unaffected.

## Daytona

```bash
mkdir -p ~/.config/daytona && echo dtn_... > ~/.config/daytona/api_key   # or export DAYTONA_API_KEY
```

Accounts carry a total-disk quota: keep concurrent sandboxes × per-sandbox disk
under it.

## Modal

The credential is a token pair, kept in the config file Modal's CLI writes:

```bash
uv tool install modal && modal token new     # writes ~/.modal.toml
```

`MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` in the environment work as well; one
half without the other is treated as missing. `MODAL_PROFILE` and
`MODAL_ENVIRONMENT` pick the workspace when the profile's default is not the
one you want.
