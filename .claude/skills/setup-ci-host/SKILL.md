---
name: setup-ci-host
description: Prepare a new physical CI host so /data/miles_ci is canonical — a real directory on the biggest disk, or a symlink to it — before installing miles GitHub Actions runners. Triggers, "setup ci host", "/data/miles_ci not ready", "miles ci host bootstrap", or similar. Idempotent; safe to re-run. Run once per host before either the README's docker-compose runner flow or `manage-gh-runners`. Boundary vs `manage-gh-runners`, this skill only prepares the host filesystem layout (big-disk detection, mkdir, symlink); it does NOT add, remove, or talk to GitHub runners.
user_invocable: true
---

# Setup CI Host

Prepares a new physical host to run miles GitHub Actions CI runners by making
`/data/miles_ci` canonical on the host.

The miles CI workflow (`.github/workflows/_run-ci.yml`) bind-mounts
`/data/miles_ci` and its subdirectories (`models`, `datasets`, `hf_cache`)
from the host into every CI job container, using literal paths everywhere.
For that to work, every CI host must provide `/data/miles_ci` either as a
real directory on its biggest disk OR as a symlink to wherever the real big
disk is. This script automates that.

Run this once on each new CI host, before the runner-setup steps in
[`tests/ci/README.md`](../../../tests/ci/README.md) or the externally-managed
flow in [`manage-gh-runners`](../manage-gh-runners/).

## Usage

```shell
cd /root/miles/.claude/skills/setup-ci-host
./setup-host.sh
```

Re-runs are safe: the script is idempotent and a no-op on hosts that already
have `/data/miles_ci` correctly set up.

## What it does

1. Probes mounted filesystems with `df` and picks the mount point with the
   most total bytes, excluding `tmpfs`/`devtmpfs`/`overlay`/`squashfs` and
   system mounts (`/`, `/boot`, `/dev`, `/proc`, `/run`, `/sys`).
2. **If the biggest mount is `/data`** (the canonical case, e.g. scitix-72):
   ensures `/data/miles_ci` exists as a real directory. Done.
3. **If the biggest mount is something else** (e.g. `/mnt/nvme0n1` on novita):
   ensures `<big_disk>/miles_ci` exists as a real directory, then atomically
   replaces `/data/miles_ci` with a symlink pointing to `<big_disk>/miles_ci`
   (creates `/data` first if missing).
4. Creates `/data/miles_ci/models`, `/data/miles_ci/datasets`,
   `/data/miles_ci/hf_cache` if they do not exist.
5. Prints a summary: chosen mount, whether `/data/miles_ci` is a real dir or
   a symlink, free space on the backing disk.

## Verification

After running, the operator can confirm the result with:

```shell
ls -la /data/miles_ci
df -h /data/miles_ci
readlink /data/miles_ci || echo "(real directory, not a symlink)"
```

The first command should list the four subdirs (`models`, `datasets`,
`hf_cache`, plus any existing `runner_<hostname>` work dirs). `df -h` should
report on the host's big disk regardless of whether `/data/miles_ci` is real
or a symlink. `readlink` succeeds only when `/data/miles_ci` is a symlink.

## When to re-run

- After adding a larger disk to the host (the script will re-detect; if
  `/data/miles_ci` already exists pointing at a smaller disk, the script
  prompts before moving anything — see [Safety](#safety)).
- After noticing CI mount errors that suggest `/data/miles_ci` is wrong.

## Safety

The script never silently overwrites a `/data/miles_ci` that already exists
and points somewhere unexpected. When run from a terminal, it prompts the
operator interactively; in non-interactive contexts (cron, CI, scripts) it
refuses and exits non-zero so a human always makes the destructive call.

Two prompts can appear:

1. **Existing real directory on the wrong disk** — `/data/miles_ci` has data
   but the biggest disk is elsewhere. Options:
   - `[m] migrate`: `rsync` the data to the big disk, then `rm -rf` the
     original and symlink. Preserves data.
   - `[w] wipe`: `rm -rf` the original (data lost), then symlink. Requires
     typing `yes` to confirm.
   - `[a] abort` (default): leave everything as-is, exit non-zero.
2. **Existing symlink pointing at the wrong target** — `/data/miles_ci`
   already symlinks somewhere, but not to the current biggest disk. The
   prompt asks `[y/N]` to re-point. Re-pointing does NOT migrate data from
   the old target; only the symlink changes.

The script requires root (or write access to `/data`). It will detect a
non-root invocation and suggest `sudo ./setup-host.sh`.

## Inside the container

Once the host side is prepared, the runner container's compose mount
`/data/miles_ci:/data/miles_ci` works as an identity bind (per the docker-out-
of-docker requirement explained in `docker-compose.yml`). The inner job
container sees `/data/miles_ci` as a real bind from the host's canonical
path, whether or not the host side is itself a symlink — Docker resolves the
host symlink at bind time, so the container always sees the underlying
directory. This is the desired behavior.
