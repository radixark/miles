---
name: manage-gh-runners
description: Add, remove, list, or swap GitHub Actions self-hosted runners on a CI host that uses the env-var-driven `gh-runner` docker image (raw `docker run`, no compose). Triggers, "add gh runner", "remove gh runner", "list ci runners", or similar. Run `setup-ci-host` first if /data/miles_ci is not yet provisioned. Boundary vs the `actions-runner` + docker-compose flow documented in `tests/ci/README.md`, only use this skill when `docker ps` shows the `gh-runner` image. Does NOT prepare the host filesystem layout — that is `setup-ci-host`'s job. Host tags (`--tags aliyun`) give job containers RDMA on RoCE hosts.
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

## Runner image

`manage-runners.sh` launches `gh-runner:latest`, built on each host from
[`gh-runner/`](gh-runner/): Ubuntu with the actions-runner distribution baked
in, the docker CLI for sibling job containers, and an entrypoint that seeds the
identity-mounted `RUNNER_DIR`, registers once, and otherwise runs the existing
registration.

```shell
docker build -t gh-runner:latest /root/miles/.claude/skills/manage-gh-runners/gh-runner
```

Existing runner containers keep their old image until they are recreated.

## Usage

```shell
cd /root/miles/.claude/skills/manage-gh-runners
export REPO_URL=https://github.com/<org>/<repo>

./manage-runners.sh list
./manage-runners.sh add    --name <host>-4gpu-0 --cvd 0,1,2,3 --hwtype h200 --ngpu 4 --token <REG_TOKEN>
./manage-runners.sh add    --name <host>-4gpu-0 --cvd 0,1,2,3 --hwtype h200 --ngpu 4 --token <REG_TOKEN> --tags aliyun
./manage-runners.sh remove --name <host>-4gpu-0 --token <REG_TOKEN>
```

Get `<REG_TOKEN>` from the repo's settings UI:
`https://github.com/<org>/<repo>/settings/actions/runners/new?arch=x64&os=linux`.
The same token is accepted by both `config.sh --token` (register) and
`config.sh remove --token` (deregister); no separate removal token is
needed. Tokens expire ~1 hour after issuance.

## Label and CVD convention

Every GPU runner gets:

```
self-hosted, Linux, X64, gpu-runner, <name>, <hwtype>, <Ngpu>gpu
```

* `<name>` — typically `<host>-<Ngpu>gpu-<i>`.
* `<hwtype>` — `h200`, `h100`, ... (matches workflow `runs_on` predicates).
* `<Ngpu>gpu` — `2gpu`, `4gpu`, `8gpu` (meta-label required by jobs that
  pin GPU count, e.g. `pr-test.yml`'s `stage-c-4-gpu-h200`).

CPU-only runners, even on GPU hosts, MUST NOT carry GPU-model, GPU-count, or `gpu-runner` labels; Docker builders retain `cpu` and `docker-build`.

`--ngpu` must equal the number of devices in `--cvd`; the script enforces
this so a 4-GPU runner can't accidentally claim the `2gpu` label.

CVD partitioning rule: contiguous, non-overlapping slices summing to the
host's physical GPU count. Example 4+2+2 split on an 8-GPU H200 host:

| Runner | CVD |
|---|---|
| `<host>-4gpu-0` | `0,1,2,3` |
| `<host>-2gpu-0` | `4,5` |
| `<host>-2gpu-1` | `6,7` |

## Host tags

`--tags` adds comma-separated host tags as runner labels and enables per-tag
runner setup. Changing a runner's tags means `remove` + `add`: GitHub keeps the
labels a runner registered with.

### `aliyun`: RDMA in job containers on RoCE hosts

aliyun hosts are RoCE: eight `mlx5_bond_N` LAG devices on `bond0`-`bond7`, each
rail its own IPv6 `/64` with the host at `::1`, and cross-rail or cross-host
traffic routed by a `/46` route via a per-rail gateway. GitHub attaches each job
container to a runner-created bridge network (`container.options` cannot set
`--network`). The HCAs are visible there, but their GIDs belong to netdevs
outside the container's network namespace, so `ibv_modify_qp` to RTR fails with
`ENODEV`: Mooncake, NVSHMEM and plain verbs all fail. IB hosts do not need this
tag.

With `--tags aliyun`, `add` also:

* creates one `rdma-railN` docker network per bond, once per host: ipvlan L2
  (it keeps the bond's MAC; the fabric drops macvlan MACs), IPv6 only in the
  rail's `/64` with containers in `<prefix>:ffff::/80`, no gateway (one would
  install an IPv6 default route through a nonexistent next hop), and the host's
  non-connected routes on that bond in the `rdma.routes` label;
* installs [`hooks/aliyun/docker`](hooks/aliyun/docker) under
  `/data/runner-setup/hooks/aliyun`, mounts it read-only first in the runner
  container's `PATH`, and checks that the runner resolves `docker` to it.

The runner resolves `docker` through `PATH` and runs `docker create`, then
`docker start <id>`, before any job step. The hook passes every command through
except `docker start`: after a successful start of a container on
`github_network_*`, it connects every `rdma-railN` and installs the routes,
running the host's `ip` in the job's network namespace through a short-lived
`--pid host --privileged` helper from the runner's own image. Any failure fails
the start, so a job never runs without RDMA silently. The job's IPv4 networking
(default route, Ray and SGLang node IPs) stays on the GitHub bridge. A runner
started as a service instead of this container reads `PATH` from
`RUNNER_DIR/.path`.

Check a tagged runner: `docker exec <name> sh -c 'command -v docker'` prints
`/opt/runner-hooks/aliyun/docker`, and a job container started through it lists
eight `rdma-railN` networks and eight `/46` routes. On aliyun-2 a cross-rail
`ib_write_bw` inside such a container (8 MiB messages, 4 QPs) reached 388 Gb/s,
and Mooncake CPU-to-CPU reached 92-106 GB/s, matching host networking;
`docker start` takes about 2.4 s longer. To undo, recreate the runner without
the tag; the networks are inert without attached containers.

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
* **Token only at registration**: `add` creates the runner container, registers
  it from a one-shot container that reads the token from a 0600 temporary file
  (`RUNNER_TOKEN_FILE`, `RUNNER_REGISTER_ONLY`), then starts it, so `docker
  inspect` never shows a token. An EXIT trap set right after `docker create`
  deletes the container on any exit until it starts, and a failed start first
  tries to deregister the runner (a warning names a runner it could not
  deregister). `add` refuses a runner directory that still holds a
  registration (run `remove` first), so an image built before these variables
  fails registration instead of starting a listener.
* **Deregister BEFORE `docker rm -f`**: `config.sh remove --token <TOKEN>`
  inside the container clears the runner from GitHub's UI. Skipping it
  leaves an offline-zombie entry that confuses future operators.
* **Per-runner host dirs (`/data/runner-setup/data/<name>`,
  `/data/runner-cache/<name>`) are not reusable across runner names** —
  `.runner` registration state is name-bound; the script deletes these on
  `remove` for that reason.
