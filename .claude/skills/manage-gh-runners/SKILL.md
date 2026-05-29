---
name: manage-gh-runners
description: Add, remove, list, or swap GitHub Actions self-hosted runners on a CI host that uses the env-var-driven `gh-runner` docker image (raw `docker run`, no compose). Triggers, "add gh runner", "remove gh runner", "list ci runners", or similar. Run `setup-ci-host` first if /data/miles_ci is not yet provisioned. Boundary vs the `actions-runner` + docker-compose flow documented in `tests/ci/README.md`, only use this skill when `docker ps` shows the `gh-runner` image. Does NOT prepare the host filesystem layout — that is `setup-ci-host`'s job.
user_invocable: true
---

# Manage gh-runner Containers

Add, remove, or swap GitHub Actions self-hosted runners on a CI host that
uses the `gh-runner` image (raw `docker run`, env-var driven). This is
distinct from the `actions-runner` + `docker-compose` flow documented in
[`tests/ci/README.md`](../../../tests/ci/README.md).

If you don't already know which flow your host uses, check:

```shell
docker ps --format '{{.Names}}\t{{.Image}}'
```

* `gh-runner` image → use this skill
* image starting with `ghcr.io/actions/actions-runner` → use the README's compose flow

Run [Step 0 (`setup-ci-host`)](../setup-ci-host/SKILL.md) before any of the
operations below if `/data/miles_ci` is not yet provisioned on the host.

## Usage

```shell
cd /root/miles/.claude/skills/manage-gh-runners
export REPO_URL=https://github.com/<org>/<repo>

./manage-runners.sh list
./manage-runners.sh add    --name <host>-4gpu-0 --cvd 0,1,2,3 --hwtype h200 --ngpu 4 --token <REG_TOKEN>
./manage-runners.sh remove --name <host>-4gpu-0 --token <REG_TOKEN>
```

Get `<REG_TOKEN>` from the repo's settings UI:
`https://github.com/<org>/<repo>/settings/actions/runners/new?arch=x64&os=linux`.
The same token is accepted by both `config.sh --token` (register) and
`config.sh remove --token` (deregister); no separate removal token is
needed. Tokens expire ~1 hour after issuance.

## Label and CVD convention

Every runner gets:

```
self-hosted, Linux, X64, gpu-runner, <name>, <hwtype>, <Ngpu>gpu
```

* `<name>` — typically `<host>-<Ngpu>gpu-<i>`.
* `<hwtype>` — `h200`, `h100`, ... (matches workflow `runs_on` predicates).
* `<Ngpu>gpu` — `2gpu`, `4gpu`, `8gpu` (meta-label required by jobs that
  pin GPU count, e.g. `pr-test.yml`'s `stage-c-4-gpu-h200`).

`--ngpu` must equal the number of devices in `--cvd`; the script enforces
this so a 4-GPU runner can't accidentally claim the `2gpu` label.

CVD partitioning rule: contiguous, non-overlapping slices summing to the
host's physical GPU count. Example 4+2+2 split on an 8-GPU H200 host:

| Runner | CVD |
|---|---|
| `<host>-4gpu-0` | `0,1,2,3` |
| `<host>-2gpu-0` | `4,5` |
| `<host>-2gpu-1` | `6,7` |

## Why the script does what it does

A few invariants the script encodes that are easy to get wrong by hand:

* **Identity bind** `/data/runner-setup/data/<name>:/data/runner-setup/data/<name>`
  — the runner spawns sibling job containers through the host docker socket
  and forwards its in-container `--work` path verbatim to the daemon; if
  host and container paths differ, the daemon resolves a directory the
  runner never wrote to.
* **Cache mount** `/data/runner-cache/<name>/.cache → /root/.cache` keeps
  pip / HF / torch caches on the big disk instead of the root partition.
* **No `/data/miles_ci` mount on the runner container itself** — only the
  per-job containers it spawns need that path, and that bind is declared
  in [`_run-ci.yml`](../../../../.github/workflows/_run-ci.yml) (resolved
  against the HOST filesystem by the daemon).
* **Deregister BEFORE `docker rm -f`**: `config.sh remove --token <TOKEN>`
  inside the container clears the runner from GitHub's UI. Skipping it
  leaves an offline-zombie entry that confuses future operators.
* **Per-runner host dirs (`/data/runner-setup/data/<name>`,
  `/data/runner-cache/<name>`) are not reusable across runner names** —
  `.runner` registration state is name-bound; the script deletes these on
  `remove` for that reason.
