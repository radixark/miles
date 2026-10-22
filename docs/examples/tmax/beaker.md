---
title: "TMax on Beaker"
description: "Running the TMax experiments on AI2's Beaker: submissions, resume, images, mirrors, monitoring and failure triage."
# Generated from examples/tmax/beaker/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
How the [TMax Terminal-RL](/examples/tmax) experiments run on AI2's Beaker: what a submission does, the config's
`beaker:` section, run names and resume, images and sandbox pulls, how to watch a run, and what its failures look
like. `submit.py` plays the role open-instruct's `mason.py` plays for its launch scripts; the
[open-instruct comparison](/deep-dive/open-instruct-comparison) lists where the two differ.

## Commands

```bash
# Production: open-instruct's 4-node recipe (2 training + 2 rollout nodes, 64k context)
python examples/tmax/beaker/submit.py examples/tmax/configs/prod_4node.yaml

# Smoke tests: 4 GPUs on one node, or 2 replicas x 2 GPUs
python examples/tmax/beaker/submit.py examples/tmax/configs/smoke_async.yaml
python examples/tmax/beaker/submit.py examples/tmax/configs/smoke_2node.yaml

# Print the Beaker spec instead of submitting
python examples/tmax/beaker/submit.py examples/tmax/configs/prod_4node.yaml --dry-run

# Launcher flags after `--` override the config
python examples/tmax/beaker/submit.py examples/tmax/configs/smoke_2node.yaml -- \
    --trace-dir /weka/oe-adapt-default/allennlp/deletable_rollouts/miles --dump-training-data

# Continue an existing run: its checkpoints, traces and W&B run. The experiment needs a name of its own (--name)
# when the old one still exists; --exclude-host (repeatable) keeps it off a bad node
python examples/tmax/beaker/submit.py examples/tmax/configs/prod_4node.yaml \
    --exp-name tmax_prod4node_qwen35_9b_20261016_001037 --name tmax_prod4node_qwen35_9b_20261016_001037_resume1 \
    --exclude-host holmes-cs-aus-556
```

Before submitting:

- **Commit `examples/tmax`.** The job runs the committed code at the current `HEAD`, not the checkout, and
  `submit.py` refuses uncommitted changes under `examples/tmax` unless you pass `--allow-dirty`.
- **Log in to the `beaker` CLI**, and check that the workspace holds the secrets the job reads.
- **Start the registry mirrors** if the config says `mirror_url: auto` (see [Sandbox images](#sandbox-images-credentials-and-mirrors)).
  Without them the job still runs, pulling from docker.io, and `submit.py` warns.

## What a submission does

1. **`submit.py`** reads the config. It names the run `<exp_name>_YYYYMMDD_HHMMSS` (UTC) unless `--exp-name` is
   given, and resolves `mirror_url: auto` to the mirrors running now. It then writes a Beaker v2 spec: image,
   GPUs, shared memory, weka mounts, secrets, `minRuntime` + `autoResume` + task retries, and replicas when
   `nodes > 1`. It creates an experiment named after the run.
2. **Every replica** clones the repository at the pinned commit (`git clone --shared` of the weka checkout into
   `/tmp/miles_snapshot`) and runs `tmax_beaker_entry.sh <config>`.
3. **Podman** comes up on every replica, the way open-instruct's Terminal-RL jobs bring it up. The entry script
   apt-installs podman 4.9, copies the vendored `podman/` config, writes `storage.conf`, and runs `docker_login.sh`:
   `sandbox_shards` daemons with their own storage, the mirror config, the image janitor, and the Docker Hub login.
   An `alpine` container proves the first daemon works.
4. **Worker replicas** (rank > 0) forward each podman shard socket on TCP port `23750 + shard`, accepting only the
   head's address. They wait up to 2 h for the Ray head (the head runs `prepare` first), join it, and stay until the
   head marks the run done (exit 0); a failed or preempted head stops them through Beaker.
5. **The head** (rank 0):
   - starts a checkpoint pruner (every 10 min it keeps the newest `keep_last_n_checkpoints`, never the one
     `latest_checkpointed_iteration.txt` names, and of the HF snapshots only every `hf_snapshot_every`-th and the
     newest finished one);
   - runs `run_tmax_qwen3_5.py prepare` (download and conversion, cached on weka);
   - starts the Ray head and waits up to 30 min for every GPU to join;
   - adds the workers' forwarded shards to `TMAX_DOCKER_HOST`;
   - runs `execute` against the running cluster (`MILES_SCRIPT_EXTERNAL_RAY=1`);
   - waits for the Ray job itself (`follow_ray_job.py`). `ray job submit` also returns 0 when its log stream drops
     mid-run, so the job's own status decides, and anything but `SUCCEEDED` fails the replica;
   - marks every node done.

## The config's `beaker:` section

| Key | Default | Meaning |
|---|---|---|
| `cluster`, `workspace`, `budget` | `ai2/holmes`, `ai2/oe-agents-holmes`, `ai2/oe-omai` | Where it runs and who pays |
| `priority` | `normal` | The configs use `urgent` |
| `min_runtime` | unset | `minRuntime` + `autoResume`: preemptible only after this long, then resumed automatically (mason's `--min_runtime --auto_resume`) |
| `preemptible` | `false` | Only used without `min_runtime`; Beaker rejects the two together |
| `retries` | `0` | Task retries; each resumes the run from its checkpoints |
| `nodes` | `1` | Replicas, one Ray node each; set `num_nodes` to the same value at the top level |
| `gpus` | required | GPUs per replica |
| `synchronized_start_timeout` | `1h` | Multi-node: how long replicas wait for each other to start |
| `shared_memory` | `256GiB` | `/dev/shm` |
| `image` | `docker.io/radixark/miles:latest` | Pin a dated image instead; see [Images](#images) |
| `sandbox_shards` | `1` | Podman daemons per node (`SWERL_PODMAN_SERVICE_COUNT`) |
| `mirror_url` | unset | docker.io pull-through mirrors, comma-separated, or `auto` |
| `keep_last_n_checkpoints` | `0` (keep all) | The pruner's retention |
| `weka` | `[nora-default, oe-adapt-default]` | Buckets mounted at `/weka/<bucket>` |
| `dockerhub_username` | `$DOCKERHUB_USERNAME` | Login for sandbox image pulls |
| `secrets` | `DOCKER_PAT`, `HF_TOKEN`, `WANDB_API_KEY` from `<user>_DOCKER_PAT` etc. | Env var → Beaker secret |
| `env` | none | Extra env vars, e.g. `NCCL_DEBUG: WARN` |
| `gateway_registry` | unset | With `sandbox_backend: gateway`: the podman fleet's head registry (`head+sqlite:///weka/…/head.sqlite3`); the head starts a gateway from it. See [Sandboxes on a podman fleet](#sandboxes-on-a-podman-fleet-gateway-backend) |
| `gateway_url` | unset | With `sandbox_backend: gateway`: a running gateway to use instead |
| `gateway_min_replicas` | `1` | Podman replicas the gateway must list before training starts |
| `literegistry_version` | `1.0.57` | The literegistry release the head's gateway runs (the fleet's) |
| `literegistry_podman_client_version` | `0.1.4` | The `literegistry-podman-client` release the agents use |

`submit.py` also sets `NUM_NODES`, `BEAKER_ALLOW_SUBCONTAINERS=1`, `BEAKER_SKIP_DOCKER_SOCKET=1`,
`PYTORCH_ALLOC_CONF=expandable_segments:True`, `SANDBOX_SHARDS`, `MIRROR_URL`, `KEEP_LAST_N_CHECKPOINTS`,
`EXTRA_ARGS` (the overrides), `MILES_SCRIPT_EXP_NAME` (the run name, which wins over the config's `exp_name`), and
on multi-node jobs `NCCL_SOCKET_IFNAME=ib` and `NCCL_IB_HCA=^=mlx5_bond_0` (NCCL over InfiniBand, not the bonded
Ethernet).

## Runs, names and resume

- **The run name keys everything.** It names the checkpoints (`<output_dir>/<run>/checkpoints`), the traces and
  dumps (`<trace_dir>/<run>/`), the W&B group and run id, and the Beaker experiment.
- **Retries resume.** A Beaker retry (preemption after `min_runtime`, or a failure with `retries` left) reruns the
  same command with the same run name. Megatron loads the latest checkpoint, the dataset resumes at its saved
  position, and W&B continues the run. In-flight rollouts are lost; their prompts are skipped, not regenerated.
- **A new submission starts fresh**, even from the same config. `--exp-name <run>` continues an existing run.
- **Checkpoints are big.** One Qwen3.5-9B checkpoint with Adam state is about 117 GB, so `keep_last_n_checkpoints: 3`
  holds about 350 GB. With `hf_snapshot_every: 20` (the prod configs) the policy is also exported in HF format at
  every save, to `<checkpoints>/hf/step_<rollout_id>` (the weights after `rollout_id + 1` steps, so `step_19` is
  after 20), about 18 GB each; the snapshots after every 20 steps are kept, about 450 GB over 500 steps. They are
  the language model's weights under the base model's `Qwen3_5ForConditionalGeneration` config, without the vision
  tower or the MTP layers (Megatron trains neither), like an offline conversion: graft those back from the base model
  (`convert_qwen35_causallm_to_cg.py`) before evaluating one.
- **Dumps are smaller.** `dump_training_data` writes about 200 MB per step (`rollout_data/<step>.pt`,
  `train_data/<step>_<rank>.pt`, plus Miles' `dashboard_columns/`), about 100 GB over 500 steps. The per-episode
  trace JSONL grows by about 70 KB per episode.

## Images

- **Pin a dated official image per config**: `docker.io/radixark/miles:dev-YYYYMMDDHHMM`. The configs use
  `dev-202610151741` (SGLang 0.5.22.dev46, torch 2.13 on CUDA 13.0, Megatron `9df207fd1`).
- **Session server v2 needs SGLang's prefill weight versions** (`--enable-prefill-weight-versions`). Images from
  2026-10-14 on have them; with an older one (e.g. `dev-202610131159`) the job dies at start with an
  `AttributeError` on `enable_prefill_weight_versions`.
- **CUDA 13 only.** Current images need a CUDA 13 driver: `ai2/holmes` (B300) has one, `ai2/jupiter` (H100) does
  not, and upstream retired the CUDA 12 images.
- **Check an image before a long run.** Submit a short one-GPU job on it that prints the versions and the flags you
  rely on (`python -c "import sglang, torch; print(sglang.__version__, torch.__version__)"`, `python -m
  sglang.launch_server --help | grep <flag>`).
- **Podman is not in the image.** The entry script apt-installs 4.9 at every start and
  `pip install docker`. open-instruct builds podman 5.6.2 and crun 1.14.3 into its image; a thin image derived from
  the Miles one with those baked in, launched with beaker-gantry, is the planned replacement.

## Sandbox images: credentials and mirrors

Every task has its own image (`hamishi740/swerl-tmax-v3:<tag>`, 14,490 of them, about 300 MB each), pulled by each
podman shard that runs it.

- **Pull credentials.** Pulls carry the Docker Hub login from the `<user>_DOCKER_PAT` Beaker secret. podman takes
  pull credentials from the API request alone, so the entry script exports them as `TMAX_REGISTRY_USERNAME` /
  `TMAX_REGISTRY_PASSWORD` and `tmax_sandbox` sends them with every pull. An empty one pulls anonymously, at 100
  pulls per hour per IP. Each pull logs `pulling sandbox image <image> on <daemon> as <user>`.
- **Tokens stay out of the log.** The entry script turns `set -x` off around the token, and the launcher keeps
  `--wandb-key` off the logged train command. When scanning logs for leaks, grep the secret values, not the
  variable names.

`registry_mirror/registry_mirror.py` runs open-instruct's Docker Hub pull-through caches (`registry:3`,
host-networked on `ai2/neptune`, logged in with the same secret):

```bash
python examples/tmax/beaker/registry_mirror/registry_mirror.py launch   # 3 mirrors, ports 5000 / 5010 / 5020
python examples/tmax/beaker/registry_mirror/registry_mirror.py urls     # what mirror_url: auto resolves to
python examples/tmax/beaker/registry_mirror/registry_mirror.py probe    # /v2/ and a real task manifest per mirror
python examples/tmax/beaker/registry_mirror/registry_mirror.py test     # pull-through and fallback, on podman 4.9
python examples/tmax/beaker/registry_mirror/registry_mirror.py warm     # open-instruct's warmer, one job per mirror
```

- **How a job uses them.** `mirror_url: auto` resolves the running mirrors at submit time. The entry script shuffles
  them per node; podman tries them in order and then docker.io, so a dead mirror costs one refused connection per
  pull, not the job.
- **Mirrors move.** Whenever Beaker reschedules one, its address changes and its cache starts empty. Relaunch the
  mirrors, then resubmit (with `--exp-name` to continue a run) to pick up the new addresses.
- **Warming is optional and large.** A full warm is about 4.4 TB per mirror. The image janitor deletes images unused
  for 10 minutes, so most pulls are cache misses on the node anyway.
- **After rotating the Docker Hub token**, update the Beaker secret, then relaunch the mirrors, which read it at
  start.

## Sandboxes on a podman fleet (gateway backend)

With `sandbox_backend: gateway` in the config, the job runs no podman: every episode's container lives on a
LiteRegistry podman fleet, a separate deployment of rootless podman replicas and Docker Hub mirrors that heartbeat
into a Redis registry ([literegistry](https://github.com/goncalorafaria/literegistry),
`literegistry_podman_beaker`). This is open-instruct's `literegistry_podman` setup. It takes the sandboxes, and the
CPU they burn, off the nodes that train and run SGLang ([INVESTIGATIONS.md](https://github.com/radixark/miles/blob/main/examples/tmax/INVESTIGATIONS.md)).

```yaml
sandbox_backend: gateway
beaker:
  gateway_registry: head+sqlite:///weka/oe-adapt-default/gfaria/primebeaker/registries/podman-terminal-jupiter-head.sqlite3
```

- **What the head does.** It installs `literegistry-podman-client` (the agents' client) into the training
  environment, and literegistry into a venv of its own, since literegistry pins starlette and pulls streamlit, which
  the training environment cannot take. It then starts `python -m literegistry.gateway
  --registry=<gateway_registry> --register=False` on `127.0.0.1:24750`, restarting it if it exits; the gateway's log
  goes to `<output_dir>/<run>/literegistry-gateway-<UTC time>.log` on weka, one file per attempt. Before training starts, the head waits up to 30 min for the gateway to
  list `gateway_min_replicas` podman replicas. The gateway finds the fleet's Redis through the head registry, so it
  follows a Redis move, and prints the first replica's `session_limits` once enough are listed. `gateway_url` skips
  all this and uses a running gateway. However the job ends, including Beaker's SIGTERM on preemption or cancel
  (SIGKILL follows 5 min later), the head closes the containers it still holds on the fleet
  (`tmax_gateway.py close-all`, at most 60 s); otherwise they keep fleet capacity for the replicas' 2 h idle timeout
  and the next attempt's containers cannot start.
- **What a container costs.** Started from the dev box, a container took 7.5–9.4 s to start, including the pull
  through the fleet's mirrors and setup, and each command took 0.3–0.6 s. In the Beaker smoke, from holmes, the
  median command took 1.2 s (p90 2.3 s), its own run time included.
- **Episode behavior** stays as with local podman, with these differences:
  - Output is read up to 512 KB of stdout and 128 KB of stderr, rather than 1 MB each, so the tail of a huge output
    comes from an earlier position.
  - Replicas enforce the limits local podman only requests: 4 GB of RSS and 2,048 processes per container,
    checked by a userspace watchdog every 5 s. They also allow 8 containers per replica and a 2 h idle timeout
    (`session_limits` in the roster).
  - A container the fleet reports lost ends the episode (`end=sandbox_lost`): `410 sandbox_lost` (a watchdog
    breach, a stopped container, or `unknown`), `410 affinity_owner_lost` (its replica left the roster), or a `404`
    for its binding (the gateway no longer knows the container; the keepalive normally prevents this). The trace and
    the episode metrics record why, as `sandbox_lost_reason`. A loss the policy caused (`memory_limit`,
    `pids_limit`, `container_stopped`) scores 0, like an OOM kill; a loss the fleet caused (`unknown`,
    `affinity_owner_lost`, `binding_lost`) leaves the reward missing, so the missing-reward filter drops the group
    rather than training on a false 0. As literegistry prescribes, the episode never continues in a replacement
    container; local podman instead recreates a vanished container once.
  - Transient gateway answers (`408`, `5xx`, a dropped connection) are retried; every request is safe to repeat.
- **The fleet** is shared and managed outside this job. Its roster is at `<gateway>/v1/models`: on 2026-10-16, 172
  podman replicas, 8 mirrors and the fleet's own gateway.
- **Check it from a dev box.** Start a gateway from the same registry with `python -m literegistry.gateway
  --registry=<gateway_registry> --host=127.0.0.1 --port=24711 --register=False`. Then `curl
  127.0.0.1:24711/v1/models` lists the replicas.

## Multi-node

- **Starved nodes.** Sandboxes run without CPU limits next to Ray, SGLang and the trainers. The entry script runs
  the podman services, and so every sandbox, under `nice -n 19` (`SANDBOX_NICE`, 0 turns it off), and gives Ray's
  node health check about 20 minutes (`RAY_health_check_timeout_ms` 20000, `RAY_health_check_failure_threshold`
  60) before it declares a node dead, and the nodes' Ray processes the same before they give up on an unresponsive head (`RAY_gcs_rpc_server_reconnect_timeout_s` 1200, Ray's default 60); the log shows both (`19 /usr/bin/podman system service`, `ray pid …:
  RAY_health_check_…`). Worker replicas do not watch the head (open-instruct's `ray_node_setup.sh` leaves after 6
  failed `ray status` checks, which ended a run whose starved head was only slow): they leave once the head marks
  the run done, and Beaker's failure and preemption propagation stops them otherwise. See
  [INVESTIGATIONS.md](https://github.com/radixark/miles/blob/main/examples/tmax/INVESTIGATIONS.md) for the node deaths behind these, and option M.
- **Layout.** With `nodes: N`, replica 0 is the Ray head and runs the launcher; training takes the first nodes and
  SGLang the rest (`async_rollout_gpus` must be whole nodes). The rollout executor, and with it every agent process,
  is pinned to the head (`--pin-rollout-manager-to-head`). The sandboxes still spread: the head drives the other
  nodes' podman through the forwarded TCP ports, and `sandboxes_per_node` caps each node (128 in prod, 512 in
  total).
- **Replicas start together.** Beaker cancels replicas that have not started within `synchronized_start_timeout`
  and does not retry the cancelled ones. At Beaker's 10-minute default, one slow placement left a retried job with
  2 of its 4 nodes, and it failed every retry with `only 16/32 GPUs joined the Ray cluster in 30 min`. The default
  here is 1 h.
- **Failure propagates.** One replica failing or being preempted stops all of them (`propagateFailure`,
  `propagatePreemption`), so a retry restarts the whole cluster from the last checkpoint.
- **Timeouts.** `prod_4node.yaml` passes `--distributed-timeout-minutes 120 --sglang-watchdog-timeout 1800` and
  `NCCL_DEBUG=WARN`. Cross-node collectives on this layout have stalled for minutes (a gradient reduce-scatter
  that never returned; a 318 s weight sync), and the defaults (10 min, 300 s) killed two runs.

## Watching a run

```bash
beaker experiment get <experiment>                     # jobs, replica ranks, status
beaker job logs <head job> --since 10m                 # replica 0 carries the training log

# One line per event until the head exits: every 10th step, syncs over 20 s, failures, exited replicas
python examples/tmax/tools/watch_run.py <experiment> --every-step 10 --slow-sync 20
```

Log lines worth grepping in the head job's log:

| Line | Meaning |
|---|---|
| `Ray cluster has 32 GPUs; workers [...]` | Every replica joined |
| `rank00000] model.py:… - step N: {'train/loss': …}` | Step N trained |
| `trainer_controller_actor] … fn=update_weights ok=true elapsed_s=…` | A weight sync finished |
| `rollout_executor] metrics.py:… - perf N: {…}` | Rollout N's metrics (reward, staleness, filters) |
| `episode done: reward=… end=… turns=…` | One episode finished |
| `Failed collecting samples` | A session could not be collected; its group is dropped |
| `phase=fail`, `ok=false`, `Killing server`, `RayTaskError`, `Traceback` from `MegatronTrainRayActor` | Something is going wrong |

Timeline of the 4-node run `tmax_prod4node_qwen35_9b_20261016_001037`, from replicas starting:

| Time | Event |
|---|---|
| 1 min | All 32 GPUs in the Ray cluster (data and checkpoints already cached) |
| 15 min | First weight sync done |
| 29 min | Rollout 0 complete |
| 41 min | Step 0 trained |
| After that | A step about every 5 min |

Weight syncs take 1.5-8 s, with an occasional 30-50 s one.

In W&B, a run and its group are named after the experiment name, in the config's `wandb_project` (entity
`ai2-llm`): `oe-general-agents` for `prod_4node`, `miles-run_tmax_qwen3_5` for the smokes. The useful signals are:

- `rollout/raw_reward_unfiltered`;
- `rollout/fully_async/avg_staleness` and `stale_groups_filtered`;
- `rollout/dynamic_filter/drop_zero_std_*`;
- `train/train_rollout_logprob_abs_diff` (about 0.007 when healthy) and `train/grad_norm`;
- `train/kl_loss`, the KL to the base model (open-instruct's `objective/kl2_avg`), and `train/entropy_loss`;
- `rollout/agent/end_reason/*`, `rollout/agent/non_submitting_fraction`, `rollout/tools/{bash,env_reset}/*`;
- `rollout/unfiltered/frac_solve_rate_*` (the groups' solve rates before the filter);
- `perf/update_weights_time`.

Ignore `perf/tokens_per_gpu_per_sec` and `perf/rollout_time` in fully-async runs. Every episode is also in the trace
files, `<trace_dir>/<run>/traces-<host>.jsonl`: one JSON record per episode with `reward`, `metrics.end_reason`,
`metrics.turns`, `metrics.tool_errors`, `metrics.model_query_time_sum`, `metrics.env_execution_time_sum`,
`metrics.sandbox_start_time`, `metrics.total_time`, and on a gateway-backend loss `metrics.sandbox_lost_reason`.

## Failures and what they mean

What to do when a symptom shows up. Every failure mode seen so far, with its status, is indexed in
[FAILURE_MODES.md](https://github.com/radixark/miles/blob/main/examples/tmax/FAILURE_MODES.md).

| Symptom | Cause | Action |
|---|---|---|
| `timed out after waiting 10m0s for synchronized replica start`; retries then fail with `only 16/32 GPUs joined the Ray cluster in 30 min` | Beaker cancelled replicas that started late and never retried them | Fixed by `synchronized_start_timeout: 1h`. If it recurs, stop the experiment and resubmit with `--exp-name` |
| NCCL watchdog timeout inside a gradient reduce-scatter; or the SGLang scheduler watchdog killing engines during a weight sync | A cross-node collective stalled or crawled | Timeouts are raised in `prod_4node.yaml`; the retry resumes from the last checkpoint |
| `httpx.ReadError` from `update_weights_from_distributed` in `MegatronTrainRayActor` (`wait_futures index=… failed`), during a sync far slower than usual | The connection to some engines dropped mid-sync. Not a timeout: the client has no read timeout. Cause unknown | The retry resumes. See [INVESTIGATIONS.md](https://github.com/radixark/miles/blob/main/examples/tmax/INVESTIGATIONS.md) |
| One rollout node's engines decoding at a few tokens/s per request (`Decode batch … gen throughput`) while the other node's are normal; a step taking 30+ min; a weight sync taking minutes | That node is CPU-starved; the leading suspect is CPU-heavy sandbox tasks next to the engines | Nothing to do mid-run; see [INVESTIGATIONS.md](https://github.com/radixark/miles/blob/main/examples/tmax/INVESTIGATIONS.md) for the options |
| `the actor's node was terminated unexpectedly: health check failed due to missing too many heartbeats`, then `Killing server cell_id=…` | A node too busy for Ray's heartbeat; Ray declares it dead and its engines are disposed. The job then fails | The Beaker retry resumes from the last checkpoint |
| `benching sandbox daemon tcp://<node>:2375x … ReadTimeout` | That node's podman daemons too slow to answer | Episodes move to other daemons for about 5 min; repeated benching of one node goes with the starvation above |
| `!!!!!!! Segfault encountered !!!!!!!` from `RolloutExecutor`, then `EOFError: Ran out of input` from agent processes | A native crash in the rollout executor; no stack in the job log | The retry resumes. Set `PYTHONFAULTHANDLER: "1"` in the config's `env` to capture a stack next time |
| `gateway http://127.0.0.1:24750: N podman replicas (need M)` and the head exits | Gateway backend: the fleet's registry is unreachable or lists too few replicas | Check the registry path and the fleet's roster (`/v1/models`); see [Sandboxes on a podman fleet](#sandboxes-on-a-podman-fleet-gateway-backend) |
| `end=sandbox_lost` in episode logs | Gateway backend: the replica removed or stopped the container (its resource watchdog, or the agent killed PID 1), its replica left, or its binding expired | `sandbox_lost_reason` in the trace says which: the policy's limits and a stopped container score 0, the fleet's losses drop the group; a burst of `unknown` is the fleet's, see [INVESTIGATIONS.md](https://github.com/radixark/miles/blob/main/examples/tmax/INVESTIGATIONS.md) |
| `Failed collecting samples: ReadError('')` from `agentic_tool_call.py` | Fetching an episode from the session server failed; the whole group is dropped | 2.5-3% of episodes on the pre-rebase code; none so far on the rebased code (upstream's session-server and transport fixes) |
| `http_utils.py … Error: , retrying... (attempt 1/60, url=…/sessions` with `starlette.requests.ClientDisconnect` on the session server | A session-create request dropped mid-body | Harmless: retried and succeeded |
| `AttributeError` mentioning `enable_prefill_weight_versions` at start | Image older than session server v2 needs | Pin `dev-202610151741` or newer |
| `NewConnectionError … Connection refused` to engine ports, `retry_until_deadline … wait_init_expected_num_cells` for the first few minutes | Engines still starting | Normal |
| `Step 'bash' failed: 97 is not a valid stream` (or similar docker-py `ValueError`) in traces | podman 4.9's exec stream framing breaking under load | Counted as a tool error, the episode continues; podman 5.6.2 should remove it |
| `pulling sandbox image … as anonymous`, HTTP 429 from Docker Hub | No pull credentials | Check `dockerhub_username` and the `DOCKER_PAT` secret |
| `runroot must be set` from podman | `storage.conf` without `runroot` / `graphroot` | The entry script writes one; keep it if you change the bring-up |
| CUDA error 304 in a nested container | GPU containers cannot be nested in a Beaker task | Run the Miles image as the task; nest only CPU sandboxes |
| Ray `file_system_monitor … over 95% full` | Holmes nodes start with 10-25 GB free of 878 GB; sandbox images share the disk | The image janitor usually holds it; watch for `ENOSPC` |
| `follow_ray_job: Ray job … is still RUNNING although its submit returned; following it` | The job's log stream dropped mid-run, usually on a starved head, and `ray job submit` returned 0 | Nothing: the head keeps following the job and printing its log |
| One replica `scheduled` for hours and never ready; the others cancelled with `timed out after waiting 1h0m0s for synchronized replica start`; no further retry | A node that never starts the job (`holmes-cs-aus-510`, 2026-10-17) | Stop the experiment; resubmit with `--exp-name` and `--exclude-host` for that node |
| `No completed rollout groups for 30s`, repeated | Rollout stalled; fully async only warns | Nothing exits on its own yet; stop the job and resume it |
| Synchronous step stuck at 0/N with no errors | Zero-std filter with small groups drops every group | `filter_zero_std: false` for smoke tests |
| Every episode `end=no_tool_call turns=1` | SGLang's reasoning and tool-call parsers not set | The launcher passes them; keep them when editing the command |

## Cluster notes

- **`ai2/holmes`.** 8× B300 (288 GB) per node, CUDA 13, about 878 GB of local disk with little free. Weka buckets
  `nora-default` and `oe-adapt-default` are mounted at `/weka/<bucket>`.
- **Paths the configs use.**
  - Models and datasets (shared by runs, converted once):
    `/weka/nora-default/shashankg/miles_tmax_v2/{models,datasets}`.
  - Checkpoints: `/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/shashankg/miles/<run>`.
  - Traces and dumps: `/weka/oe-adapt-default/allennlp/deletable_rollouts/miles/<run>`.
- **Qwen3.5 `torch_dist` checkpoints** converted before upstream's head-sharded linear-attention layout (#3610) are
  rejected at load. Convert from HF into a new model directory; never delete a converted checkpoint another job may
  be reading.
