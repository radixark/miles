# TMax prod runs: how they hang and fail

Every way a 4-node TMax run on Beaker has hung or failed so far (2026-10-15 to 2026-10-19), and where each one
stands. This page is the index:

- the evidence and the options considered are in [INVESTIGATIONS.md](INVESTIGATIONS.md) (linked per row);
- what to do when a symptom shows up mid-run is in the
  [Beaker runbook](beaker/README.md#failures-and-what-they-mean);
- open work is in [ROADMAP.md](ROADMAP.md).

When a run hits a new failure, add a row here and its evidence to INVESTIGATIONS.md.

Status: **open** (nothing addresses it), **mitigated** (made rarer or survivable, cause still there), **fixed**,
**workaround** (avoided by hand, for example by excluding a node).

Hardening was paused on 2026-10-19; work moved to closing the training gap with open-instruct. The open rows below
are where to resume.

## Open, by impact

1. **The podman head starves at the first training step after resuming from `iter_0000224`.** It happened on every
   attempt: resume 4, resume 5 attempts 1 and 2, and resume 6. The podman run cannot get past step 225.
   ([A1](#a-the-run-stalls-or-hangs))
2. **Nodes starve in general.** Sandboxes without CPU or memory limits share nodes with the trainers, the engines,
   and (on the head) all 512 agent processes. A1, A2, B2 and B5 are probably all this. Options A, C and E in
   INVESTIGATIONS.md are untried.
3. **A replica stays `scheduled` after a slow teardown on its node.** In one case Beaker never retried, and the run
   sat idle for 43 h. ([C1](#c-beaker-and-the-cluster))
4. **The gateway fleet answers unreliably.** Steady `503`s, about 3% of containers lost, and one throughput
   collapse (Oct 17 v2) that has not recurred. ([D](#d-gateway-backend-and-the-podman-fleet))
5. **A hung head is no longer bounded.** Since `8aaf766777` nothing ends a run whose head hangs without exiting.
   ([E1](#e-latent-risks))

## A. The run stalls or hangs

| # | What | Signature in the head log | Seen | Status |
|---|---|---|---|---|
| A1 | Head starves at the first step after a resume | About 1–3 min after `ft op=train phase=start rollout=225`: session-server `404`s; `benching sandbox daemon … ReadTimeout`; router `HTTP health check failed` for engines on other nodes; `ActorUnavailableError … Connection reset by peer`; the dashboard answers `500`; `WebSocket closed for job …`; the step never finishes | Resume 4 (Oct 17), resume 5 attempts 1–2 and resume 6 (Oct 19) | **Open.** Unverified hypothesis: the data order is fixed, so every resume runs the same tasks at this point, and one loads the head node (no sandbox limits). Next: a per-node sampler (option A) and the tasks in flight. [Resume 5/6](INVESTIGATIONS.md#2026-10-19-resume-5-dies-at-the-first-step-after-iter_0000224-three-times) |
| A2 | A rollout node's engines crawl | `Decode batch … gen throughput` of a few tokens/s on one node; 30–47 min steps; weight syncs of minutes | Oct 16: step 15, steps 56–59, after step 31 | **Mitigated:** sandboxes run at nice 19 (`f5ef1f8273`). Suspected CPU starvation by sandboxes. [47-min step](INVESTIGATIONS.md#2026-10-16-47-minute-step-after-step-15), [slowdown](INVESTIGATIONS.md#2026-10-16-slowdown-from-0920-to-1041-retry-2-steps-5659), [320 s sync](INVESTIGATIONS.md#2026-10-16-320-s-weight-sync-retry-1-after-step-31) |
| A3 | A cross-node collective stalls | A gradient reduce-scatter that never returned; a 318 s weight sync | Oct 16 | **Mitigated:** `--distributed-timeout-minutes 120`, `--sglang-watchdog-timeout 1800`. Megatron's 10 min and SGLang's 300 s defaults had killed two runs |
| A4 | Fully-async rollout stops producing groups | `No completed rollout groups for 30s`, repeated; nothing exits | Smokes | **Open** (ROADMAP P1 "fail fast on a stalled rollout") |
| A5 | Gateway episodes slow until none finish | `gateway request failed (503)` about 60 a minute; sandbox time per episode 60 s → 750 s within 15 min; 0 steps in 8.8 h | Gateway v2, Oct 17 | **Open, not recurred.** The Oct 19 runs on a smaller, redeployed fleet held at about 93 episodes a minute. [Gateway at prod scale](INVESTIGATIONS.md#2026-10-17-the-gateway-backend-at-prod-scale-two-4-node-runs) |
| A6 | A failed replica takes minutes to stop | After a cancel, our replicas take 2–8.5 min to exit, some with SIGKILL (exit 137); the head kept running 6 min after `follow_ray_job` reported `FAILED` | Oct 17, Oct 19 | **Open.** Probably tearing down nested podman (sandboxes, storage mounts). Fix idea: stop sandboxes and podman on SIGTERM. Feeds C1 |

## B. The job fails

| # | What | Signature | Seen | Status |
|---|---|---|---|---|
| B1 | `ray job submit` returns 0 while the job still runs, so the head treats training as finished | `WebSocket closed for job … close code 1000`, then `Status for job …: RUNNING` | Resume 4 (Oct 17), resume 5 and 6 (Oct 19) | **Fixed:** `follow_ray_job.py` (`2316f39868`) follows the job and exits 0 only on `SUCCEEDED`. Upstream Miles has the same behavior. [Resume 4](INVESTIGATIONS.md#2026-10-17-ray-job-submit-returns-0-mid-run-resume-4) |
| B2 | Ray declares a busy node dead | `raylet has lagging heartbeats …`, `The actor died because its node has died`, trainer `all alive cells failed` | Oct 16 retries 1 and 4; Oct 17 retries 2 and 3 | **Mitigated:** the node health check tolerates about 20 min (`RAY_health_check_*`, `f5ef1f8273`). [Retry 1](INVESTIGATIONS.md#2026-10-16-rollout-node-declared-dead-retry-1-failure), [retry 4](INVESTIGATIONS.md#2026-10-16-rollout-node-declared-dead-again-retry-4-failure), [retries 2–4](INVESTIGATIONS.md#2026-10-17-the-podman-resume-keeps-losing-nodes-retries-24) |
| B3 | Raylets exit when the head's GCS is slow for 60 s | `(raylet) Failed to connect to GCS within 60 seconds`, `Raylet is terminated` | Resume 5 attempt 1 (Oct 19) | **Mitigated:** `RAY_gcs_rpc_server_reconnect_timeout_s=1200` (`0f9404b0d4`); resume 6 survived it. [Resume 5/6](INVESTIGATIONS.md#2026-10-19-resume-5-dies-at-the-first-step-after-iter_0000224-three-times) |
| B4 | Our worker head check ends a run whose head is only slow | `Ray head … unreachable 6 times in a row; leaving` | Oct 17: three attempts (10 s per-check timeout); Oct 19: resume 6 (open-instruct's exact check) | **Fixed:** workers no longer check the head (`8aaf766777`, option M (1)). See E1. [Worker gives up](INVESTIGATIONS.md#2026-10-17-a-worker-replica-gives-up-on-a-starved-head-resume-retry-1) |
| B5 | The Ray job fails while the head starves | Engine health checks fail, then `ended FAILED: Job entrypoint command failed with exit code 1` | Resume 5 attempt 2 (Oct 19) | **Open:** the same cause as A1 |
| B6 | Rollout executor segfault | `!!!!!!! Segfault encountered !!!!!!!`, then `EOFError: Ran out of input` in agent processes | Oct 16, first attempt, step 24 | **Open, not recurred.** `PYTHONFAULTHANDLER=1` now leaves a stack. [Segfault](INVESTIGATIONS.md#2026-10-16-rollout-executor-segfault-first-attempt-failure) |
| B7 | Trainer rendezvous lost | `Failed to recv, got 0 bytes` from the c10d store while building a `reduce_scatter` communicator | Oct 16, retry 5 | **Open, not recurred.** The store's host had a 95%-full disk (C4). [Rendezvous](INVESTIGATIONS.md#2026-10-16-trainer-rendezvous-lost-in-retry-5-a-node-disk-95-full-all-run) |
| B8 | Weight-sync reply dropped | `httpx.ReadError` from `update_weights_from_distributed` (`wait_futures index=… failed`) | Oct 15, old run | **Open, not recurred.** Option K: treat it as "outcome unknown" and check the engines' weight version. [ReadError](INVESTIGATIONS.md#2026-10-15-weight-sync-readerror-old-4-node-run) |
| B9 | Shared Weka volume full | `OSError: No space left on device` writing traces or checkpoint events | Oct 16, retries 2 and 3 | **Workaround:** the pruner keeps 3 checkpoints (about 118 GB each); `oe-adapt` is filled by other users. [Weka full](INVESTIGATIONS.md#2026-10-16-oe-adapt-weka-volume-full-retry-2-and-3-failures) |
| B10 | Retries run out | Every retry fails on the same cause, then the experiment ends | Oct 16 (`retries: 5`) | **Mitigated:** `retries: 10`. A deterministic failure such as A1 still uses them all up |

## C. Beaker and the cluster

| # | What | Signature | Seen | Status |
|---|---|---|---|---|
| C1 | A replica stays `scheduled` and never gets ready, right after one of our replicas was torn down slowly on that node | Status has only `created` and `scheduled`, no log. The others are cancelled with `timed out after waiting 1h0m0s for synchronized replica start` | `holmes-cs-aus-510` (Oct 17: no retry, 43 h idle); `-508` (Oct 19: retried after 1 h) | **Open.** Excluded both; infra issue drafted (scratchpad, not filed). Possibly fixed by A6's teardown |
| C2 | Replicas cancelled for starting late, then retries never get all nodes | `timed out after waiting 10m0s for synchronized replica start`, then `only 16/32 GPUs joined the Ray cluster in 30 min` | Oct 16 | **Fixed:** `synchronized_start_timeout: 1h` |
| C3 | Preempted | `preempted by job … allocated workloads are scheduled ahead of unallocated ones`; or after `min_runtime`, workspace-group balancing | Gateway v2 (Oct 17, after 8 h); gateway run with `min_runtime: 0` (Oct 19, twice in 2 h, lost steps 5–8 both times) | **Mitigated:** `min_runtime: 8h` (`f020639f98`). Only steps since the last checkpoint are lost (`save_interval: 5`) |
| C4 | Node disk full before the job starts | Ray's `file_system_monitor … over 95% full` from the first minute | `-556`, `-500`, `-490` | **Workaround:** `--exclude-host` |
| C5 | `apt-get update` fails on a node | `At least one invalid signature was encountered` for every repo, exit 100 within 6 s, crash loop | `-498`, `-527` (Oct 17) | **Workaround:** excluded |
| C6 | A worker cannot join the head's Ray cluster | `Failed to get node info b'RPC error: Deadline Exceeded'`, `The current node timed out during startup` | `-540`, twice (Oct 19) | **Workaround:** excluded |
| C7 | Beaker's own healthcheck fails before start | `healthcheck for job … failed` | `-550` (Oct 17) | Beaker cordoned it |

Excluded on every submission: `holmes-cs-aus-490, -498, -500, -508, -510, -527, -540, -556`.

## D. Gateway backend and the podman fleet

| # | What | Signature | Seen | Status |
|---|---|---|---|---|
| D1 | The fleet's Redis is down, so the head's gateway lists no replicas | `Redis endpoint … unavailable: Timeout`; `-1 podman replicas (need 64)` after the 30 min wait | Oct 17, 03:50–05:06 | Fleet-side. The Beaker retry recovered. [Redis outage](INVESTIGATIONS.md#2026-10-17-fleet-redis-outage-delays-the-gateway-prod-run) |
| D2 | Burst of lost containers | `410 sandbox_lost`, reason `unknown`, 309 in 4 min, with every replica job up | Oct 17 v1 | **Open, fleet-side.** Lost containers now drop their group instead of scoring 0 (`2445076320`) |
| D3 | Containers lost when a replica leaves the registry | `gateway container lost (410)`, `affinity_owner_lost`: "strict affinity server is no longer registered" | About 3% of episodes on Oct 19 | Group dropped. Fleet-side; worth sending to the owners |
| D4 | Steady `503`s | `affinity_owner_unavailable` (most); `Podman session capacity exhausted (8/8)`; a few `408 command timed out`, `502 affinity_upstream_disconnected` | About 150 a minute on Oct 19, nearly all failing in under 0.1 s and succeeding on retry | Harmless so far. Logged with codes since `40463ef3ef` |
| D5 | Throughput collapse | See A5 | Oct 17 v2 | Open, not recurred |
| D6 | After a restart, containers fail to start: the preempted attempt's containers still hold fleet capacity | `Podman session capacity exhausted (8/8)` `503`s, 14–18 thousand per 10 min, and `sandbox never started` for about 80 min after the resume; batches with 20–37% `start_failed` episodes trained as one-token zeros, so response length and reward dipped for 12 steps | Gateway resume 1 after its 06:27 preemption (Oct 20); probably also after the 22:08 relaunch (Oct 19) | **Fixed** (`3abf79b12a`): the head closes its containers on exit (TERM trap, `tmax_gateway.py close-all`). A SIGKILL without the grace period (node failure, OOM) still leaves them until the 2 h idle timeout, and failed starts no longer train (`start_failure: drop` in `prod_4node_gateway.yaml`): their groups are dropped |

## E. Latent risks

Not seen yet, but possible with the current code.

| # | Risk | Why | Status |
|---|---|---|---|
| E1 | A hung head is never ended | Workers no longer check the head (B4), and `follow_ray_job.py` waits without a limit while the job is running or the dashboard is unreachable. A head that hangs without exiting holds 4 nodes until someone stops it | **Accepted** (option M (1), 2026-10-19). Watch runs externally |
| E2 | A run ends early recorded as "succeeded" | Only if Ray itself reports `SUCCEEDED` early. B1's path is closed | Low |
| E3 | Upstream's new weight-sync timeouts | Rebasing past `8fa34c05c1` / `e991cbc48a` adds `--update-weights-timeout` (600 s) and `--update-weight-engine-request-timeout` (300 s). Each fails a run whose sync is slow; we have seen 1,286 s and 318 s syncs | Set both high (for example 7200 s) when rebasing |
| E4 | Sandbox memory limits do nothing | podman runs with `cgroups="disabled"`, so the 4 GB limit and OOM detection may not apply (ROADMAP "Not verified") | Unverified; ties into A1 |
| E5 | `git gc` on the weka checkout breaks a later retry | The job clones the weka checkout with `--shared` at a pinned commit | Open (ROADMAP P2) |
