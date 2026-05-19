# Doc about CI

## Configure GitHub secrets

https://github.com/radixark/miles/settings/secrets/actions

* `WANDB_API_KEY`: get from https://wandb.ai/authorize

## Setup new GitHub runners

### Step 1: Env

Write `.env` mimicking `.env.example`.
The token can be found at https://github.com/radixark/miles/settings/actions/runners/new?arch=x64&os=linux.

WARN: The `GITHUB_RUNNER_TOKEN` changes after a while.

Required variables:

* `GITHUB_RUNNER_URL`, `GITHUB_RUNNER_TOKEN` — registration credentials.
* `DATA_ROOT` — host-side path to the CI data tree. `/data` on scitix-72 (default); `/mnt/nvme0n1` on novita. If unset, compose falls back to `/data`.

### Step 2: Prepare `/home/runner/externals`

```shell
docker run --rm -it --privileged --pid=host -v /:/host_root ubuntu /bin/bash -c 'rm -rf /host_root/home/runner/externals && mkdir -p /host_root/home/runner/externals && chmod -R 777 /host_root/home/runner/externals'
docker run -d --name temp-runner ghcr.io/actions/actions-runner:2.334.0 tail -f /dev/null
docker cp temp-runner:/home/runner/externals/. /home/runner/externals
docker rm -f temp-runner
ls -alh /home/runner/externals
```

### Step 3: Run

```shell
cd /root/miles/tests/ci/github_runner
docker compose up -d
```

The base `docker-compose.yml` defines a template `runner` service; the per-host
concrete runner services (e.g. `scitix-72-0`, `scitix-72-1`) live in
`docker-compose.override.yml`. The base service is set to `replicas: 1` so a
stray `docker compose up` without the override never silently spins up many
runners.

## GPU partitioning convention

GPU exposure to inner CI job containers happens in two layers:

* **Driver layer**: the workflow `.github/workflows/_run-ci.yml` keeps
  `--gpus all` in `container.options`, so every job's inner container sees
  every physical GPU on the host via `nvidia-smi -L`. Removing `--gpus all`
  would make `nvidia-container-runtime` default to `NVIDIA_VISIBLE_DEVICES=void`
  and produce a zero-GPU container, so this flag is intentional.

* **CUDA layer**: each runner container sets `CUDA_VISIBLE_DEVICES` in its
  compose `environment:` block. `_run-ci.yml`'s `container.options` includes
  a bare `--env CUDA_VISIBLE_DEVICES` flag that propagates that env into the
  inner job container. miles tests respect CVD (e.g. `miles/ray/train_actor.py`,
  `miles/backends/sglang_utils/sglang_engine.py`), so CUDA-level partitioning
  is sufficient.

Host conventions:

* **scitix-72** (H200, 2 runners): `scitix-72-0` pins `CUDA_VISIBLE_DEVICES=0,1,2,3`;
  `scitix-72-1` pins `4,5,6,7`. Per-runner CVD is defined in
  `docker-compose.override.yml`.
* **scitix-73** (H200, 2 runners, externally managed): already injects CVD via
  `actions-runner/.env` per the historical H200 split-worker convention; no
  change needed in this repo.
* **novita-host2 / novita-host4** (H100, 1 runner each, externally managed):
  CVD intentionally left unset so jobs see all 8 GPUs. The bare
  `--env CUDA_VISIBLE_DEVICES` forwards the "unset" state, and CUDA defaults
  to seeing every visible device.

## DATA_ROOT path identity rule

`docker-compose.yml` mounts `${DATA_ROOT}/miles_ci:${DATA_ROOT}/miles_ci` —
the SAME variable on both sides of the colon. This is mandatory: the runner
process spawns sibling job containers through the mounted `/var/run/docker.sock`,
and the runner's in-container `--work` path is forwarded verbatim to the host
daemon. If the host source and container target differ, the daemon resolves a
host path the runner never wrote to, and the sibling container mounts an
unrelated (or empty) tree.

The workflow `.github/workflows/_run-ci.yml`'s four data mounts
(`/root/models`, `/root/datasets`, `/root/.cache/huggingface`, plus an
extra `/data/miles_ci` aggregator) are NOT subject to the identity rule
because they are pure data mounts, not work-dir forwarding. Only the host
side is parameterized via `${{ inputs.data_root }}`; the container targets
stay fixed because miles tests hardcode them (e.g. `/root/models` is
referenced by 50+ test files).

## Restarting runners after compose changes

* **env-only changes** (e.g. adjusting `CUDA_VISIBLE_DEVICES` or `DATA_ROOT`):
  recreate the container so it picks up the new compose env.

  ```shell
  docker compose down && docker compose up -d
  ```

  A bare `docker restart <container>` does NOT pick up compose env changes —
  it restarts the existing container with its original env.

* **label changes** (e.g. adding a new runner label): the entrypoint guards
  `config.sh` with `if [ ! -f /home/runner/.runner ]`, so a recreated
  container does NOT re-register. To force re-registration, delete
  `/home/runner/.runner` inside the container before recreate, OR run
  `config.sh --replace` manually.

## Debugging

Logs

```shell
# All containers
docker compose logs -f

# One container
docker logs -f scitix-72-0
```

Exec

```shell
docker exec -it scitix-72-0 /bin/bash
```

Verify a runner's CUDA partition

```shell
docker exec scitix-72-0 printenv CUDA_VISIBLE_DEVICES   # -> 0,1,2,3
docker exec scitix-72-1 printenv CUDA_VISIBLE_DEVICES   # -> 4,5,6,7
```

Quickly iterate

```shell
docker compose down -v && docker compose up -d && docker logs -f scitix-72-0
```
