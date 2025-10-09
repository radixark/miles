# How to setup runner

### Step 1: Env

Write `.env` mimicking `.env.example`.
The token can be found at https://github.com/radixark/miles/settings/actions/runners/new?arch=x64&os=linux.

WARN: The `GITHUB_RUNNER_TOKEN` changes after a while.

### Step 2: Run

```shell
cd /data/tom/primary_synced/miles/tests/ci/github_runner
docker compose up -d
```

### Debugging

Logs

```shell
docker compose logs -f
```

Exec

```shell
docker exec -it github_runner-runner-1 /bin/bash
```
