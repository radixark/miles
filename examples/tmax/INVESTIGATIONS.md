# TMax port investigations

Investigations from running the port, newest first. Each records what was observed, the evidence, the options
considered and whether one was chosen. Open work lives in [ROADMAP.md](ROADMAP.md), which links back here; failure
signatures and what to do about them are in the [Beaker runbook](beaker/README.md#failures-and-what-they-mean), and
every failure mode with its current status is indexed in [FAILURE_MODES.md](FAILURE_MODES.md).

Runs referred to:

- **Old 4-node run:** `tmax_prod4node_qwen35_9b`, pre-rebase code `1dfdb12cb`.
- **Current 4-node run:** `tmax_prod4node_qwen35_9b_20261016_001037`, Beaker experiment `01M510QWZ4HQMBBN1EW3ZQX6JA`,
  commit `edf2c7ae3e`, 4× 8 B300.
  - **First attempt:** 00:23 to 04:21 UTC on 2026-10-16, steps 0–24.
  - **Retry 1:** 04:33 to 06:15, steps 25–34.
  - **Retry 2:** 06:45 to 15:19, steps 35–97; died with `ENOSPC`.
  - **Retry 3:** 15:24 to 15:26; died with `ENOSPC` at startup.
  - **Retry 4:** 15:30 to 22:40, steps 95–149; rollout node declared dead.
  - **Retry 5** (the last of `retries: 5`): 22:45 to 23:46; failed in its first step. Retries exhausted.
- **Resume:** submitted on the podman backend with `--exp-name tmax_prod4node_qwen35_9b_20261016_001037`, continuing
  from `iter_0000149`, off `holmes-cs-aus-556` and `holmes-cs-aus-500` (`--exclude-host`), with `retries: 10`. A first
  submission (`01M53JRRQ310SG0GDCF249TW1A`, 5 retries) was stopped before it started.
- **Gateway runs:** `tmax_prod4node_gateway_qwen35_9b_20261017_035830` (v1) and `..._20261017_074752` (v2), stopped.
- **Resume 3/4 (hardened, commit `6fb5cb00d8`):** resume 3 (`01M55HZB5SVPMEV5W9C0RBRX2S`) crash-looped from 20:25 on
  2026-10-17: `apt-get update` failed with `At least one invalid signature was encountered` for every repository on
  `holmes-cs-aus-498` and `holmes-cs-aus-527` (exit 100 within 6 s, three retries in 12 minutes), stopped. Resume 4
  (`01M55SE04SXJGSEM0WQEQA79GD`) excludes those two as well as `-556` and `-500`. It trained rollout 225 from
  `iter_0000224`, failed at 21:33, and its retry never started; stopped on 2026-10-19
  ([below](#2026-10-17-ray-job-submit-returns-0-mid-run-resume-4)).

## Open decisions

The options below are the ones presented during the investigations; none is implemented yet.

| # | Option | Addresses | Cost | Recommendation |
|---|---|---|---|---|
| A | Per-node host sampler in the entry script: load, CPU, memory, disk, TCP reset/abort counters and top processes every 15 s | Confirms or rules out rollout-node CPU starvation (2026-10-16 investigations) | `examples/tmax` only; next submission | Do first |
| B | Run every sandbox command under `nice -n 19` | Engines, agents and the router win the CPU when a task goes CPU-heavy; commands still run | `examples/tmax` only | Do together with A |
| C | Keep sandboxes off the rollout nodes: per-node sandbox caps (0 on rollout nodes, more on training nodes) | Strongest isolation of the SGLang engines | Needs a per-node cap in `tmax_sandbox`; training nodes then carry 256 sandboxes each | If A confirms starvation |
| D | Cap thread counts inside sandboxes (`OMP_NUM_THREADS` and friends) | Stops one numpy/BLAS task from using every core | Changes task behavior relative to open-instruct | Avoid |
| E | Run agent processes on their sandbox's node instead of all on the head | Spreads agent CPU over the nodes, as open-instruct's engine actors do | Larger change to the rollout layout | Later |
| F | `--sglang-attention-backend trtllm_mha` | 1.75× engine throughput at 32k context ([engine probe](#2026-10-16-engine-settings-probe-attention-backend-and-fp32-lm-head)) | One flag; validate in a 4-GPU Miles smoke (multi-turn sessions, 64-token radix pages, in-place updates) | Yes, after the smoke |
| G | fp32 LM head on both sides: `--sglang-enable-fp32-lm-head` plus an fp32-accumulating Megatron output layer | Train/rollout log-prob gap −32% (0.0065 → 0.0044) | Miles core change on the trainer side (check upstream first); ~1% engine throughput | Yes, matches open-instruct |
| H | `PYTHONFAULTHANDLER=1` in the Beaker env | Captures a Python stack if the rollout executor segfaults again | One env var | Bundle with the next restart |
| I | More episodes in flight (1024, 256 sandboxes per node) | Uses idle engine capacity during long-tail phases | Doubles sandbox load per node, which is the suspected cause of the stalls | Not before A–C |
| J | Episode wall-clock cap (`episode_timeout_s`) | Bounds long-tail episodes | open-instruct has no cap | Not now |
| L | Sandboxes on a LiteRegistry podman fleet: `sandbox_backend: gateway`, open-instruct's `literegistry_podman` setup | Removes the sandboxes, and their CPU load, from the training and SGLang nodes altogether ([2026-10-16 entry](#2026-10-16-gateway-backend-sandboxes-on-a-podman-fleet)) | Implemented (`255d93f374` .. `1e9265bd13`, merged in `e961a935aa`), 4-GPU smoke passed; depends on a fleet run outside the job | The 4-node run (fleet capacity first) |
| M | Worker replicas' watch on the Ray head. (1) None: workers leave when the head marks the run done, or through Beaker's `propagateFailure` when the head's job fails. (2) open-instruct's: 6 failed `ray status` checks 5 s apart, with no timeout on a check. (3) A ~20-minute tolerance (`HEAD_MISS_LIMIT` 80, `aeb4f9ecd3`) | The ported check (6 misses with a 10 s timeout each, about 90 s) failed three podman attempts on 2026-10-17 while a starved head was slow to answer | (1) relies on Beaker noticing a failed head; a head that hangs without exiting would leave the workers waiting | **(1), taken 2026-10-19 by the user, after (2) ended resume 6 on a starved head that had recovered from 5 misses minutes before; (2) was used 2026-10-17 to 19** |
| K | Make a dropped weight-sync reply non-fatal: treat it as "outcome unknown" after a completed broadcast and verify the engines' weight version | Survives the `ReadError` seen once in the old run | Miles core change; check upstream first | Only if it recurs |
| N | Keep the 9B OPD run's recipe (lr 1e-6, OPD term from the rollout log-probs) and decide at about step 30 ([2026-10-27 entry](#2026-10-27-the-9b-pure-opd-runs-early-transient)) | open-instruct's identical run settled by itself by steps 25 to 30 | None | Current choice |
| O | OPD term from the trainer's recompute: drop `--use-rollout-logprobs` | Removes the stale student term behind the transient | One flag, but the rollout log-probs then stop anchoring DPPO's ratio, so stale samples lose their correction | No: P keeps the correction |
| P | Prime-RL's split: OPD term from the trainer's log-probs, DPPO still anchored on the rollout log-probs ([OPD across frameworks](../../docs/deep-dive/opd-across-frameworks.md)) | The stale student term, keeping the staleness correction | Small Miles core change (`apply_opd_kl_to_advantages` / `loss/objective.py`); check upstream first | First choice if N fails |
| Q | Per-token clip on the OPD advantage (open-instruct `mopd`'s `opd_adv_clip`) | Bounds the -36 advantages of teacher-rejected tokens and the KL-to-base spikes | Miles core change; changes the objective relative to open-instruct's `opd` runs | Low value: the outliers are a format preamble the student drops by step 10, so a clip only changes steps 0–10 |
| R | lr 5e-7 (open-instruct tried it for a 4B student, run `zjfcsx10`) | Slows everything, the transient included | A config | Last resort: it does not target the cause |

Suggested next restart: A + B + F + H (and G once the trainer side exists), validated on a 4-GPU smoke, then
resubmitted with `--exp-name tmax_prod4node_qwen35_9b_20261016_001037` to continue from the latest checkpoint.

## Ideas to explore

- **Remote sandbox fleets** (from the user). open-instruct has two alternatives to colocated podman:
  1. A LiteRegistry podman fleet behind a local gateway (branch `literegistry_podman`). Now option L, ported.
  2. A hosted OpenSandbox service on GCP (branch `pd_open_sandbox_cuda13`, paid). Not explored yet.
- **GRPO after OPD on tasks the start does not always solve.** GRPO from the 9B OPD run's step 80 gained nothing by
  step 120 ([README](README.md#grpo-from-the-9b-opd-runs-step-80-4-nodes)): 37-45% of its sampled groups are all-pass
  and about 20% all-fail, so about 65% of the rollouts are dropped by the zero-variance filter. The filter already keeps
  those groups out of the gradient, so dropping the tasks up front mainly buys throughput (about 40% fewer rollouts per
  step without the all-pass tasks, about 2.5x fewer without both kinds) and a shift toward harder tasks. A run sees
  under one epoch, so it needs a pass-rate sweep with the start checkpoint (about 116k episodes at k=8 over all
  14.5k tasks, or a subset), or screening each task with a few samples before the full 32 (SPEED-RL's idea), which
  needs a two-phase group in the rollout wrapper.

Larger directions to evaluate against the evidence below. Add yours here.

## 2026-10-29: `oe-adapt` full again; checkpoint saves fail

- **Runs:** GRPO from OPD step 80 (Beaker `01M60M1V4QBQDE3XMG8ZQ0KR3R`, commit `286b0c0563`), last checkpoint 169, and
  the 4B RLVR run (`01M60M1X1A5M1D3HXNY123BYYG`, commit `3ec4925326`), last checkpoint 134.
- **What happened:** `/weka/oe-adapt-default` (440 TB) reached 100% at about 02:00 UTC, 34 MB free. At 02:08 both runs
  failed in `save_model` (`CheckpointException`, then `NonRetryableError: All cells failed during
  execute_first_alive#save_model`), and Beaker retried both.
- **GRPO's retries failed at startup.** Attempt 3 failed within 2 minutes. Attempt 4 failed after 30: the entry
  script could not create the gateway log on weka (`tmax_beaker_entry.sh: line 162: …literegistry-gateway-…`), so the
  gateway never started and the 30-minute replica wait ran out. Its commit predates `320cd35167`, which keeps the
  gateway restarting and logs to `/tmp` while weka refuses (the 2026-10-23 incident). The 4B run has that fix: its
  attempt 4 logged the gateway to `/tmp`, resumed from step 134 and saved `iter_0000139` and `iter_0000144` once
  space returned.
- **Recovery:** the user freed space (4.2 TB free at 03:46, 18 TB at 04:16). GRPO's attempt 5 saved `iter_0000174` at
  04:38; four of its ten retries were used. Weka reads stayed slow for a while afterwards: a gateway-log read and a
  `gantry` start each blocked in `D` state for minutes.
- **Our share:** about 4.7 TB on 2026-10-29. Each 9B run keeps 3 training states of about 117 GB plus HF snapshots of
  17 GB, and the debug dumps (`--save-debug-rollout-data`, `--save-debug-train-data`) add 40-175 GB per run under
  `deletable_rollouts/`.
- **Options:** in [ROADMAP.md](ROADMAP.md) (P1, checkpoint storage).

## 2026-10-28: the fleets' registry mirror fails, then the private fleet's Redis

- **Fleets:** `podman-parallel64-20261027` (parallel64; the 4B RLVR run) and `podman-private-reasoning-20261017`
  (private; GRPO from OPD step 80 and the cross-size OPD run). The older shared fleet,
  `podman-terminal-jupiter-head`, is off.
- **Symptom:** `sandbox start failed for image=hamishi740/swerl-tmax-v3:<tag>: PodmanGatewayError('gateway returned
  HTTP 503 for affinity/handshake: … Resolving "hamishi740/swerl-tmax-v3" using unqualified-search registries …
  Trying to pull docker.io/…')`, then `sandbox never started, dropping the group`.
  - On parallel64 from the 4B run's first attempt (`01M5ZZ4SG85JNP62W1XRTSCRNC`, 00:39), stopped at 02:44 without a
    checkpoint.
  - On private from 04:43: 10-15 groups dropped per run every 7 minutes, against 8 groups per step. Neither run
    trained after 04:36 (GRPO) and 04:41 (OPD), and by 05:50 handshakes hung until the client's 310 s timeout.
- **Cause:** every replica was started with one fixed `registry_mirror`, the fleet's own gateway (parallel64:
  `jupiter-cs-aus-154:29040`, private: `jupiter-cs-aus-190:28940`). When a gateway moved to another host after a node
  failure, the replicas kept the dead address; with no Docker Hub credentials, pulls of images a replica had not cached
  fell back to anonymous docker.io and its rate limit. The fleets' registered mirror services stayed healthy
  throughout. At 05:41 the private fleet's Redis head also stopped (`jupiter-cs-aus-125` refused connections), so no
  gateway could route at all.
- **Our gateway could not help.** The head runs its own gateway, as open-instruct's `start_colocated_gateway.sh`
  does, so the fleet's gateway moving never touches our routing, and it follows Redis moves through the head registry.
  But the pull happens on the replica, and routing needs Redis.
- **Fix, by the fleet owners the same night:** both fleets relaunched (128 replicas each, literegistry 1.0.61) under
  a wrapper that every 30 s writes podman's `registries.conf` from the registry: every registered mirror (shuffled per
  replica), then the fleet's gateway, then Docker Hub with credentials from the user's Beaker secrets. Private's
  Redis was recovered from its snapshot and then replaced. After the repair, 16 of 16 starts of images neither fleet
  had cached bound in 4-21 s, through a local literegistry 1.0.63 gateway started as the head starts it.
- **What we did:** stopped both private-fleet runs at 05:53 (checkpoints 59 and 94) and resumed them with their
  original configs and `--exp-name` at 06:44, on the repaired private fleet; the 4B run started over on parallel64.
  About 2 h of 8 nodes lost.
- **Later the same day:** a failed node (`jupiter-cs-aus-164`) took 3-4 replicas from both fleets: about 250
  `affinity_owner_unavailable` per run in 10 minutes and 3-4 lost containers. Training continued.

## 2026-10-27: the 9B pure-OPD run's early transient

- **Run:** `tmax_prod4node_gateway_opd_9b_20261027_022952` (Beaker `01M5XK2NCJAPFZ5M8G01659VZ5`, commit `b131645fa6`,
  `configs/prod_4node_gateway_opd_9b.yaml`): pure OPD of `allenai/tmax-9b` into `hamishivi/Qwen3.5-9B`, Megatron-mode
  teacher, 4× 8 B300, from 03:26 UTC on 2026-10-27. Compared with open-instruct's identical recipe,
  `swerl_qwen35_9b_dppo_opd_4node_64k` (W&B `6kyg0v64`; lr 1e-6, grad clip 1, DPPO TV 0.1, 8 x 32, async 4).
- **What happened:** the teacher KL fell (0.102 at step 0, 0.037 at step 13) and the raw task reward rose (0.42 to
  0.81), but from step 9 the train/rollout gap, the DPPO mask and the training entropy rose:

  | Step | Gap, Miles / open-instruct | DPPO-masked tokens, Miles / OI | Teacher log-prob on student samples, Miles / OI | Entropy, Miles | Grad norm, Miles / OI |
  |---|---|---|---|---|---|
  | 4 | 0.022 / 0.012 | 0.7% / 0.1% | -0.38 / -0.42 | 0.30 | 2.2 / 0.75 |
  | 10 | 0.078 / 0.035 | 5.9% / 2.1% | -0.44 / -0.45 | 0.48 | 2.7 / 0.56 |
  | 14 | 0.033 / 0.078 | 0.1% / 4.9% | -0.81 / -0.59 | 0.76 | 7.3 / 0.66 |
  | 17 | 0.184 / 0.154 | 11.0% / 9.5% | -0.90 / -0.65 | 0.92 | 2.6 / 0.97 |
  | 25–30 | – / 0.04 | – / 2–4% | – / -0.3 to -0.5 | – | – / 0.5–0.7 |

  open-instruct logs no entropy. Its run peaked around steps 16 to 22 (gap 0.16, 10% masked, teacher log-prob -0.96)
  and settled by steps 25 to 30; its teacher KL reached 0.018 at step 30 and 0.005 at step 100.
- **Evidence it is not a bug:** the dumped batches' reference and rollout log-probs agree on generated tokens (mean
  |difference| 0.007, k3 0.0003 at steps 0 to 2), so the teacher weights never leaked into the reference. The
  episodes at step 14 read coherently and solve tasks. The KL-to-base spikes (22,571 at step 3) are k3 outliers from
  about 100 tokens per step (0.01%) whose OPD advantage is below -20, down to -36.
- **Grad norm:** Miles' is 3 to 5 times open-instruct's from step 0. Both report it before clipping at 1.0, so Miles
  clips every step and open-instruct rarely; under Adam a constant gradient scale barely changes the update, and the
  learning rate is the same. The spikes relative to each run's baseline match (7.3 on about 2, 1.9 on about 0.6). Not
  a reason to lower the learning rate.
- **Likely driver:** the OPD term uses the rollout's student log-prob, so with up to 4 versions of staleness a
  teacher-rejected token keeps its full negative advantage after earlier updates have lowered it. Prime-RL computes
  the term from the trainer's live log-prob instead ([OPD across frameworks](../../docs/deep-dive/opd-across-frameworks.md)).
- **Dump analysis (steps 0 to 89, every 5th; the CP shards rejoined per sample):**
  - The extreme OPD advantages are a format difference. At step 0, 412 of the 501 tokens below -10 open a visible
    preamble after `</think>` ("THOUGHT: Let me", "Now", "I", "The"), which 36% of the base student's turns have and
    which tmax-9b never writes (log-prob about -36; the student samples them at a median of -0.8). The preamble fell to
    6% of turns by step 5 and 0% from step 10; tokens below -10 fell from 501 (step 0) to 17 (step 10) and 0 (step 60).
    A clip at ±5 would touch 0.22% of tokens at step 0 and none after step 20.
  - Solved and failed trajectories carry the same per-token signal (mean reverse KL 0.011 vs 0.009 at step 89, 0.019
    vs 0.012 at step 40), so gating OPD on success would scale down 60–78% of the signal without selecting anything.
  - The signal is in the reasoning: `<think>` holds 60–78% of the |advantage| mass on 35–55% of the tokens, tool calls
    20–37% on 42–64%.
  - The grad-norm and masking bursts (steps 47, 69–72, 75) are not outliers (none below -10), not staler batches
    (mean lag 1.3–2.4 versions vs 1.8–3.0 at quiet steps) and not one trajectory (the largest carries 1–2% of the
    signal). Unexplained.
  - Scripts: `opd_dump_analysis/analyze.py` in the session scratchpad.
- **Options:** N to R in [Open decisions](#open-decisions). Open: whether the run settles by step 30 as
  open-instruct's did.

## 2026-10-19: resume 5 dies at the first step after `iter_0000224`, three times

- **Run:** resume 5 (`01M5AM2F0G62GACSWDT9H1J4W0`, commit `2316f39868`), from `iter_0000224`, off seven bad nodes.
- **What happened:** attempts 1 and 2 each died about 3 minutes into training rollout 225, as resume 4 had.
  - Attempt 1: the head's GCS stopped answering for over 60 s from about 18:31. The raylets logged `Failed to connect
    to GCS within 60 seconds` and exited, and the job failed. A worker's `ray status` reached the head again at
    18:32:55, 25 s after the first raylet gave up, so the head was starved, not dead.
  - Attempt 2: the router's health checks of the engines on two nodes failed from 19:24, and the job failed.
  - `follow_ray_job.py` recorded both as failures (no mid-run "success").
- **Stuck starts:** attempt 3's replica on `holmes-cs-aus-508` stayed `scheduled` for an hour, as on `-510` on
  2026-10-17. Both nodes had just torn down one of our replicas, which can take up to 8 minutes after a cancel (some
  end in a SIGKILL, exit 137).
- **Change:** `RAY_gcs_rpc_server_reconnect_timeout_s` 1200 (Ray's default is 60) in the entry script, so the nodes
  wait about 20 minutes for an unresponsive head, as the node health check already does the other way.
- **Resume 6** (`01M5B3TNZA8KSK1W74EJ22Y06Y`, `0f9404b0d4`, the 1200 s setting): no raylet gave up and
  `follow_ray_job.py` kept following the job, but step 225 (from 22:57) did not finish in 40 minutes on a head
  starved from about 22:56 (its dashboard answered `500` by 23:23). At 23:37:40 a worker's head check (6 failed
  `ray status` checks of about 47 s each) ended the run, after recovering from 5 misses at 23:31:53. Stopped; the
  workers' head check is removed (option M (1)).
- **Open:** why the head starves at this step every time. The data order is fixed, so each resume runs the same
  tasks; one may load the head node, whose sandboxes have no memory or process limits. Resume 5 stopped on 2026-10-19
  during attempt 4.

## 2026-10-17: `ray job submit` returns 0 mid-run (resume 4)

- **What happened:** resume 4 trained from 21:04 (first weight sync, 43.8 s, as after every resume) and began
  rollout 225 at 21:20. From about 21:23 the head was starved again:
  - its router's health checks of the engines on `10.93.1.249` failed;
  - the inference controller got `ActorUnavailableError ... Connection reset by peer`;
  - log forwarding to the head stopped at about 21:26.
- **The failure chain:**
  - At 21:27:20 the dashboard closed the job's log stream (`WebSocket closed ... close code 1000`).
  - `ray job submit` printed `Status for job ...: RUNNING` and exited 0, because it exits non-zero only for `FAILED`.
  - The entry script took training as finished and went on to mark every node done.
  - That step's `ray.init` gave up at 21:33:25 with `No node info found matching attributes`, and the head exited 1.
  - Meanwhile the workers' `ray status` checks of the head failed 2–4 times in a row between 21:29 and 21:31 and then
    recovered, so the head check did not end anything.
- **Reading:** the head starving was the trigger; Ray recovered from it on its own within about 9 minutes. The retry
  happened only because the marking step failed. Had it succeeded, every replica would have exited 0 and Beaker
  would have recorded the run as succeeded in the middle of training, with no retry. Upstream Miles's `run_ray_job`
  has the same behavior.
- **Fix (on our side, not in Miles):** after the execute step returns 0, the entry script runs
  `beaker/follow_ray_job.py`. It asks Ray for the job's own status. While the job is pending or running, it prints
  the job's driver log from about where the stream stopped and waits for the end. It exits 0 only for `SUCCEEDED`,
  and only then are nodes marked done. It adds no time limit while the dashboard is unreachable.
- **The retry never started:** retry attempt 2 scheduled all four replicas at 21:42. Replica 2, on
  `holmes-cs-aus-510`, stayed `scheduled` without ever becoming ready or writing a log. Beaker cancelled the other
  three at 22:45 (`timed out after waiting 1h0m0s for synchronized replica start`) and never started attempt 3,
  although retries were left. The experiment sat idle until it was stopped on 2026-10-19 at 17:19. The checkpoint is
  still `iter_0000224`.
- **More unhealthy nodes:** `holmes-cs-aus-490`'s `/tmp` was over 95% full when the job started (10.2 GB free of
  878 GB, from Ray's own disk monitor). `holmes-cs-aus-550` failed Beaker's own healthcheck before start and is
  cordoned.

## 2026-10-17: the gateway backend at prod scale (two 4-node runs)

Two runs of `prod_4node_gateway.yaml` (the gateway backend's first version, before its rework into `255d93f374` .. `1e9265bd13`) alongside the podman resume, to see whether
the backend can be trusted at prod scale.

- **v1** (`01M54060ECZQ0CN4XF1A5FZPZ6`, from 03:58): the fleet's Redis outage cost a retry (entry below). It then
  trained steps 0 to about 12 at about 2.6 minutes a step, with 1–1.3 s weight syncs (podman: about 5 minutes and
  2–8 s). But from 06:08 to 06:11 bursts lost 309 containers (153, 75, 81 a minute; reason `unknown`) with every
  fleet replica job up, 7–17% of the episodes, each scored 0. Stopped, and the scoring changed (`a0e7785296`: the
  fleet's losses drop the group).
- **v2** (`01M54D9TWBNSMVE29GCV6T4G7H`, from 07:48, `a0e7785296`): trained no step in 8 h 47 m. Its first rollout
  began at 08:06; within 15 minutes the sandbox time per episode went from 60 s to 750 s, against a steady ~60 `503`s
  a minute from the fleet; the episode rate fell from about 25 to under 1 a minute (1,983 in all), later with
  containers failing to start. Beaker preempted it at 16:39 to rebalance workspace groups (`ai2/omai-agents` 23% over
  target), after its 8 h minimum runtime; stopped at 17:30.
- **The roster looked healthy throughout** (253–255 replicas, at least 1,669 free sessions). So the slowdown is in
  the request path, the head's own gateway, the fleet's replicas or the network from holmes, and our logs cannot
  tell which: the client records only the status code, and the head's gateway log stays in the job.
- **Reading:** the backend cannot currently sustain a prod run on this fleet. Before another attempt: log the `503`s'
  code and `execution_outcome`, per-request latency and the head gateway's log; and check the fleet with its owner.

## 2026-10-17: the podman resume keeps losing nodes (retries 2–4)

- **Retry 2** (08:24–12:13): Ray marked node `10.93.1.246` dead (missed heartbeats); the trainer controller gave up.
- **Retry 3** (12:32–14:04): Ray marked node `10.93.2.18` dead.
- **Retry 4** (14:57–15:30): a worker replica left, `Ray head 10.93.1.234:6379 unreachable 6 times in a row`.
- **Retry 5** from 17:21; the latest checkpoint is `iter_0000224`, so the run went from step 160 to 224 in 17 hours.
- Each is what `f5ef1f8273` (Ray's health check) and `aeb4f9ecd3` (the worker watchdog) address; the user chose to
  keep retrying on the old code.

## 2026-10-17: a worker replica gives up on a starved head (resume, retry 1)

- **What happened:** the podman resume's retry 1 (from 02:23, steps 160–~184, head on `10.93.2.35`) failed at 08:00:
  the worker replica on `10.93.1.243` (`holmes-cs-aus-504`) logged `Ray head 10.93.2.35:6379 unreachable 6 times in
  a row; leaving` and exited 1. Before it, the head's own podman socket timed out (`ReadTimeout`, benched at 07:50)
  and session requests to its session servers got `404`s: the head node was starved again. The weight syncs before
  were 1–3.5 s, then 53 s at 07:55. A 21-minute sync (1,286 s) after step 170 at 04:03 had not killed it.
- **Reading:** the entry script's own watchdog, not Ray, ended this attempt: about 90 s of failed `ray status` on a
  starved head. The check was ported from open-instruct's `ray_node_setup.sh`, but with a 10 s timeout on each check
  that open-instruct does not have, so a slow head counted as a miss. Now as open-instruct has it (option M);
  `aeb4f9ecd3` had first raised it to about 20 minutes.

## 2026-10-17: fleet Redis outage delays the gateway prod run

- **Run:** `tmax_prod4node_gateway_qwen35_9b_20261017_035830` (Beaker `01M54060ECZQ0CN4XF1A5FZPZ6`,
  `prod_4node_gateway.yaml`, the gateway backend's first version), from step 0, alongside the podman run.
- **What happened:** the fleet's Redis (`jupiter-cs-aus-159:29088`) stopped answering between 03:50 and 04:13 UTC and
  the head registry emptied at 04:23. The head's gateway listed no replicas, the 30-minute readiness wait ran out, and
  the first attempt exited 1 at 04:43. The fleet's owner (goncalof) recovered Redis (`redis-local-diagnose`,
  `redis-*-recovery` experiments in `ai2/oe-agents`, 04:54–05:06) on `jupiter-cs-aus-188:29088`; the 253 replicas,
  still running throughout, re-registered with it. Beaker's retry 1 (from 04:48) found the new Redis through the head
  registry and saw 250 replicas at 05:06.
- **Reading:** a Redis outage before training costs a retry and its wait; during training it would cost more, since
  the gateway resolves every container's binding through Redis (`binding_lost` ends those episodes).

## 2026-10-17: step-149 checkpoint evaluates at base-model level on TBlite

- **Setup:** the open-instruct eval harness (tmax repo, `beaker_configs/launch_eval.sh`, Vanillux2Agent, vLLM 0.19.1,
  `qwen3_xml` tool parser, no reasoning parser), greedy (`temperature=0`, `top_p=1.0`), k=5, context 67584, 2 GPUs as
  DP=2 with 16 concurrent trials, Jupiter. `iter_0000149` was converted with Miles' `tools/convert_torch_dist_to_hf.py`
  through a wrapper (the checkpoint has no `common.pt`; its args are in the DCP `common_state` entry, and the
  linear-attention `weight_layout_version` buffers are dropped), then given `hamishivi/Qwen3.5-9B`'s vision tower
  with the tmax repo's `convert_qwen35_causallm_to_cg.py`. Scripts: `/weka/nora-default/shashankg/miles_tmax_v2/evals/`.
- **Results (TBlite, 100 tasks x 5):**

  | Model | pass@1 | pass@5 | Agent timeouts | Infra errors |
  |---|---|---|---|---|
  | Miles step 149 (greedy) | 0.412 | 0.620 | 20 | 24 |
  | `allenai/tmax-9b` (greedy, same config plus `--language-model-only`) | 0.538 | 0.710 | 24 | 20 |
  | `allenai/tmax-9b`, eval sheet (sampled T=0.7, 1 GPU, 65536) | 0.534 | 0.730 | | |
  | `hamishivi/Qwen3.5-9B` base, eval sheet (sampled) | 0.418 | 0.660 | | |
  | open-instruct `swerl-qwen35-9b-dppo-4n64k` step 120, eval sheet | 0.524 | 0.760 | | |

- **TB2.1 (89 tasks x 5), same config:** Miles step 149 pass@1 0.2045, pass@5 0.337; 25 agent timeouts and 55 infra
  errors (54 `RuntimeError`, 1 `ProcessLookupError`), so 0.233 with the infra-failed trials left out. Eval sheet: base
  0.191, open-instruct `dppo-4n64k` step 120 0.229 and step 140 0.240, `allenai/tmax-9b` 0.276. On TB2.1 the
  checkpoint is a little above base and in line with open-instruct's run at the same steps; the gap is on TBlite.
- **The conversion is right:** on 146,662 trained tokens of `rollout_data/150.pt` generated by weight version 149,
  the converted checkpoint's log-probs (HF transformers) differ from the SGLang rollout log-probs by 0.0086 on average,
  the run's own train/rollout gap; the base model differs by 0.061 (`ckpt_vs_rollout.py`).
- **The eval matches training:** the same system prompt and task wrapper as the training traces; 98% of replies
  carry thinking, 97% a tool call; median 19 turns.
- **Reading:** the step-149 policy has not improved on TBlite, although its training solve rate rose from about 41% to
  60%, and open-instruct's run of the same recipe was at 0.524 by step 120. The gap is in training. Candidates:
  PPO clip instead of DPPO, Adam beta2 0.98 instead of 0.999, the wider train/rollout gap (0.009–0.011 vs 0.0065),
  and the rising sampling entropy (mean sampled log-prob −0.30 at the start, −0.44 by step 135).

## 2026-10-16: trainer rendezvous lost in retry 5; a node disk 95% full all run

- **What happened:** retry 5 resumed from `iter_0000149` and started training step 150 at 23:36. At 23:45, 531 s
  in, trainer ranks setting up an NCCL communicator for a `reduce_scatter` failed to read its id from the c10d store
  on `10.93.1.224:20000` (`Failed to recv, got 0 bytes. Connection was likely closed`); the trainer controller gave
  up (`all alive cells failed`) and the job exited at 23:46. With `retries: 5` used, the experiment is finished.
- **A node disk full all run:** Ray's file-system monitor reported `/tmp/ray/session_*` over 95% full (9.5–12.6 GB
  free of 878 GB) on `10.93.1.239` during the first attempt and on `10.93.1.224` (`holmes-cs-aus-556`) through
  retries 2, 4 and 5, already one minute into retry 4. So it was not filled by the job's own sandbox image pulls,
  which also land on that disk (podman's `graphroot` is in the job container), and nothing prunes them.
- **Reading:** the store's host ran with almost no disk the whole time; whether that dropped the store is not shown.
- **Options:** report the node per the CI-failure rules (runner name, the log line); resubmit with
  `--exp-name tmax_prod4node_qwen35_9b_20261016_001037` (resumes `iter_0000149`), ideally on the gateway backend (L),
  which also takes the sandbox images off the nodes' disks.

## 2026-10-16: rollout node declared dead again (retry 4 failure)

- **What happened:** the weight sync after step 149 (22:08:43 to 22:17:04) took 498 s, its 34 chunks slowing from
  about 5 s to 20–30 s each, against 2–3 s for the whole sync otherwise. No step trained after it. At 22:39 the
  podman daemons on rollout node `10.93.2.35` dropped connections (`tmax_sandbox` benched them), the router could no
  longer reach that node's engines, Ray reported `raylet has lagging heartbeats due to slow network or busy
  workload` and then `The actor died because its node has died`, and the job failed at 22:40.
- **Before it:** the node's engines decoded at 70–106 tokens/s per request up to the sync, with no failed session
  collections, so the slowdown came on fast.
- **Reading:** the same sequence as retry 1 (a 320 s sync, then the node declared dead 20 minutes later): a rollout
  node too busy for Ray's heartbeats. No step was lost; `iter_0000149` was saved at 22:08.
- **Options:** L (the gateway backend) removes the sandboxes from the rollout nodes; A–C otherwise.

## 2026-10-16: `oe-adapt` Weka volume full (retry 2 and 3 failures)

- **What happened:** retry 2 died at 15:19 with `OSError: No space left on device` from the rollout executor, writing
  `dashboard_columns/rollout_98.tmp` into the trace directory. Retry 3 died 2 minutes after starting, writing the
  checkpoint directory's `events`. Retry 4 resumed from `iter_0000094`, so steps 95–97 are trained again.
- **Evidence:** at 16:10 the shared `weka-aus.beaker.org/oe-adapt` volume (`/weka/oe-adapt-default`) showed 438 of
  440 TB used, 2.1 TB free. `nora` was at 93% (3.5 TB free) and `oe-training` at 100% (13 TB free).
- **Our share:**
  - this run's checkpoints, 353 GB: 3 iterations of about 118 GB each, one written every 5 steps;
  - its traces and dumps, 28 GB;
  - two older runs' checkpoints, 351 GB each: `tmax_prod1node_qwen35_9b`, and the pre-rebase
    `tmax_prod4node_qwen35_9b`.
- **Reading:** the volume is filled by other users; every checkpoint we write needs about 118 GB of the little left.
- **Options:** free our old runs' checkpoints (about 700 GB), keep fewer checkpoints per run, or put checkpoints on
  a volume with room. Any of them still leaves the run exposed to the shared volume filling up.

## 2026-10-16: slowdown from 09:20 to 10:41 (retry 2, steps 56–59)

- **What happened:** steps 56–59 took 25, 32, 13 and 11 minutes (09:20 to 10:41), against 5–7 minutes per step
  since step 35; step 60 was back to 7 minutes. Rollout 56 took 19 minutes to fill and rollout 57 took 29, and
  training step 56 itself took 6 minutes instead of about 3. The weight syncs after steps 55 and 56 took 106 s and
  163 s. The run recovered by itself, as after step 16.
- **Evidence:**
  - Completed episodes per 20 minutes fell from about 1,500 (07:00–09:20) to about 600 (09:20–09:40). The
    episode's median time rose from about 200 s to 489 s.
  - Model-call time per turn doubled, from 4–6 s to 12–13 s. Yet each engine decoded at its usual 60–110 tokens/s
    per request, on both rollout nodes, with about 10 requests running and none queued. So model calls were held
    before reaching the engines, in the router or session server on the head node, rather than in SGLang.
  - Unlike step 16, decode speed itself never dropped.
  - Signs of an overloaded head node:
    - the router's health checks against four head-node engines failed from 09:41;
    - the weight sync after step 55 took 106 s;
    - 47 sessions failed `collect_samples` with a 120 s `TimeoutError`, 14 of them while still connecting: one at
      09:21, then 09:39–09:44. Before this, retry 2 had none. Between 09:47 and 10:17, 186 more failed, in bursts
      of up to 58 a minute; each dropped its group, which slowed the rollouts further.
  - Tool timeouts (a command running past 120 s) rose from 0.27 to 0.67 per episode. Most came from
    `task_005427` (107 in 20 minutes) and `task_000635` (173 over the hour).
- **Reading:** the same CPU contention as at step 16, here starving the head node's HTTP serving (router, session
  servers, agents) rather than its engines.
- **Options:** A, B, C, E (above) and L, the gateway backend (merged in `e961a935aa`), which takes the
  sandboxes off the nodes altogether.

## 2026-10-16: gateway backend (sandboxes on a podman fleet)

- **The idea:** open-instruct's answer to colocated podman. A separate Beaker deployment runs rootless podman
  replicas and Docker Hub mirrors that heartbeat into Redis; a LiteRegistry gateway routes each container's requests
  to the replica that owns it. The training job runs no podman.
- **Port:** `tmax_gateway.TmaxGatewaySandbox` (`examples/tmax/tmax_gateway.py`), behind `sandbox_backend: gateway`. It
  follows open-instruct's `GatewayBackend` but talks to the gateway through literegistry's own client,
  `literegistry-podman-client`, which owns the API and its error types. It tracks the current literegistry release:
  the user's call, since the fleet is launched from that repository, so there is no handling for older ones. On
  Beaker the head starts a gateway on localhost from the fleet's head registry. Mechanics are in the
  [Beaker runbook](beaker/README.md#sandboxes-on-a-podman-fleet-gateway-backend).
- **Fleet on 2026-10-16:** the registry
  `/weka/oe-adapt-default/gfaria/primebeaker/registries/podman-terminal-jupiter-head.sqlite3` points at Redis on
  `jupiter-cs-aus-159`. The roster listed 172 podman replicas (`podman128-cap8-*` and `tmax-source-replay512-*`, on
  neptune and jupiter hosts), 8 Docker Hub mirrors and a gateway. It is reachable from the dev box.
- **Tests:**
  - **Unit tests:** 12 against a fake gateway, a real HTTP server driven through literegistry's client
    (`tests/fast/examples/tmax/test_tmax_gateway_sandbox.py`; skipped where the client is not installed).
  - **Live loss:** a command spawning 3,000 processes ended its episode after 5.9 s with
    `SandboxLostError: pids_limit (3012 tasks > 2048)`.
  - **Live, from the dev box through a local gateway** (3 real tasks):
    - containers started in 7.5–9.4 s and commands took 0.3–0.6 s;
    - the persistent shell kept `cwd` and exports;
    - a 70 s command finished past the replica's 60 s exec cap, and a 2 MB output was truncated as usual;
    - the withheld tests ran in 1.4–2 s.
  - **Beaker:** 4-GPU smoke `smoke_gateway.yaml` (experiment `01M51R3GETB3QFWJJ7YCDCGJCV`, commit `45644f34ea`)
    passed:
    - the head's gateway listed 172 replicas within 70 s of start, the literegistry install included;
    - 3 steps trained, from 94 episodes, 38 of them solved, with no failed sandbox starts;
    - the median sandbox command from holmes took 1.2 s (p90 2.3 s), the command's own run time included;
    - the job exited 0.
  - **Beaker, on the final code** (experiment `01M51W45C2XH6KAHYA0ACQN2X5`, commit `37f1609b5e`) passed:
    - the head printed the fleet's `session_limits` (8 sessions, 4 GB, 2,048 processes) and saw 172 replicas 14 s
      after starting its gateway;
    - 3 steps trained (train/rollout log-prob gap 0.0071, 0.0085, 0.0069), every weight sync succeeded, and the job
      exited 0;
    - of 101 traced episodes, 37 were solved; 49 submitted, 49 filled the context, 2 made no tool call, and 1 was lost
      to `pids_limit` (its trace records `sandbox_lost_reason: pids_limit`; the agent ran a fuzzing script) and ended
      at once, in its own container;
    - the 11 tool errors were all commands that ran past the 120 s tool timeout; the gateway itself failed no step;
    - sandbox time per turn: median 0.9 s, p90 3.7 s, the command's own run time included.
- **Differences from local podman:**
  - output is read up to 512 KB of stdout and 128 KB of stderr instead of 1 MB each;
  - replicas enforce the limits local podman only requests: each container may use 4 GB of RSS and 2,048 processes
    (a userspace watchdog checks every 5 s), up to 8 containers per replica, and a 2 h idle timeout. A breach
    answers `410 {"code": "sandbox_lost", "reason": "memory_limit" | "pids_limit"}`; the smoke hit `pids_limit` once,
    at 4,538 processes;
  - a container the fleet reports lost ends the episode with reward 0 (`sandbox_lost`), and the episode never
    continues in a replacement container, as literegistry prescribes. The first smoke instead re-handshook a
    `sandbox_lost` answer and got 59 lost-sandbox tool errors; fixed by classifying errors with literegistry's own
    client.
- **What the port adds on top of literegistry's client.** The client sends one request and classifies its answer;
  these episode-level pieces, carried over from the first version of the port, still hold under 1.0.57:
  - a keepalive every 5 min, since an idle binding expires after 900 s while the policy thinks;
  - retries of transient answers;
  - commands longer than the replica's 60 s exec cap run detached and are polled to completion;
  - uploads over the 1 MB stdin cap go in 512 KB chunks;
  - when a polled command's status file never appears, a trivial exec checks whether the container is gone before
    the step fails as a tool error;
  - a `404` for the binding ends the episode as a loss (`binding_lost`) rather than failing every later step;
  - the loss reason goes into the episode metrics and trace (`sandbox_lost_reason`), and the head logs the fleet's
    `session_limits` at startup.
- **Contract:** literegistry 1.0.57, the release the fleet runs (gfaria's checkout
  `/weka/oe-adapt-default/gfaria/literegistry`, and on PyPI). The head's gateway is pinned to it.
  - The gateway now sends a pinned command once and never replays it.
  - Transport failures come back as `503 affinity_owner_unavailable` or `502`/`504`, with an `execution_outcome`
    (`not_sent` or `unknown`). Our requests are safe to repeat, so the port retries them.
  - A request's `command` is capped at 16 KB, its stdin at 1 MB and its timeout at 60 s.
  - `literegistry-podman-client` 0.1.4 raises `PodmanContainerLostError` for `410 sandbox_lost` (reasons
    `memory_limit`, `pids_limit`, `container_stopped`, `unknown`) and `410 affinity_owner_lost`. It treats a generic
    404 or 503 as a gateway error, not a loss; the port treats a 404 as one (see above).
- **Decided 2026-10-17, how a lost container scores:** the policy's losses (`memory_limit`, `pids_limit`,
  `container_stopped`) score 0, like an OOM kill; the fleet's (`unknown`, `affinity_owner_lost`, `binding_lost`)
  leave the reward missing, so the group is dropped. Prompted by the 4-node gateway run (Beaker
  `01M54060ECZQ0CN4XF1A5FZPZ6`, `prod_4node_gateway.yaml`): from 06:08 to 06:11 UTC on 2026-10-17, 309 of its
  containers were lost in bursts (153, 75 and 81 a minute) for reason `unknown`, with every fleet replica job still
  up, against about 0.5% of episodes outside the bursts.
- **Open questions:**
  - Fleet capacity when 512 of our episodes share it with other users' jobs.
  - Whether cross-cluster latency (holmes to jupiter/neptune) adds up over 64-step episodes.

## 2026-10-16: rollout node declared dead (retry 1 failure)

- **What happened:** at 06:10 Ray declared rollout node `10.93.1.245` dead
  (`the actor's node was terminated unexpectedly: health check failed due to missing too many heartbeats`). Miles'
  inference controller killed and disposed that node's 8 engines. The replica on that node exited at 06:13, and
  Beaker's failure propagation stopped the rest (head exit 143).
- **Before it:** from 06:08 the node's podman daemons stopped answering in time (`ReadTimeout`), and `tmax_sandbox`
  benched them for about 5 minutes. The same node's engines had dropped to 5–7 tokens/s per request during the
  05:47 weight sync.
- **Reading:** a node too busy for Ray's own heartbeat points to CPU or memory starvation. Each rollout node runs 8
  engines next to up to 128 sandboxes with no CPU limits, and with podman's `cgroups="disabled"` the 4 GB memory
  limit is probably not enforced either.
- **Options:** A, B, C (above).

## 2026-10-16: 320 s weight sync (retry 1, after step 31)

- **What happened:** the sync after step 31 (05:47:12 to 05:52:32) took 320 s, against 2–6 s otherwise, and
  succeeded. The trainer side was normal; the 34 broadcast chunks took 4–23 s each instead of fractions of a second.
- **Evidence:** a broadcast completes only when every engine has received it, so one slow receiver delays every
  chunk. The engines on `10.93.1.245` decoded at 5–7 tokens/s per request around 05:47 and 05:54, against about 90
  normally; those on `10.93.2.30` stayed at 70–105.
- **Reading:** the same CPU starvation as the node failure above, 20 minutes earlier on the same node. It also fits
  the slow syncs (27–64 s, once 318 s) of both 4-node runs.

## 2026-10-16: rollout executor segfault (first-attempt failure)

- **What happened:** at 04:21:14 the rollout executor process logged `!!!!!!! Segfault encountered !!!!!!!` and
  died. Its agent subprocesses then failed with `EOFError: Ran out of input`, and the job failed. Beaker's retry
  resumed from `iter_0000024`, so no training steps were lost.
- **Evidence:** no native or Python stack reached the job log. For about 15 minutes before it, session-server
  requests from the same process kept failing: connections dropped mid-request, and retries hit `404` because the
  first attempt had already gone through.
- **Options:** H, so a recurrence leaves a stack; until then the cause is unknown.

## 2026-10-16: engine settings probe (attention backend and fp32 LM head)

Question: are open-instruct's vLLM settings, or better SGLang ones, missing from the port? The settings are compared
in [Miles vs open-instruct](../../docs/deep-dive/open-instruct-comparison.md#inference-engines). The two that
needed measuring were the attention backend SGLang picks and the fp32 LM head.

**Attention backend.** SGLang 0.5.22.dev46 resolves `attention_backend=triton` for Qwen3.5 on B300, standalone as
well as under Miles. A 4-GPU Beaker probe (experiment `01M51DGA0R1F14373581KVEFED`) ran one-GPU servers with the
production flags and `sglang.bench_serving` on random token ids:

| Backend | 32k in / 1k out, 32 concurrent: output tok/s | Time per output token | Time to first token | 4k in / 1k out, 64 concurrent: output tok/s |
|---|---|---|---|---|
| `triton` (production) | 700 | 28 ms | 18 s | 5,470 |
| `trtllm_mha` (page size 64) | 1,224 | 16 ms | 10 s | 6,199 |
| `flashinfer` | 1,045 | 19 ms | 12 s | 6,159 |
| `fa4` | Fails at start: `SM100 forward with head_dim=256 does not support seqused_q/seqused_k` | | | |
| `triton` + fp32 LM head | 692 | 28 ms | 18 s | 5,494 |

On the policy's own tokens all backends give the same prefill log-probs (0.0065 from the Hugging Face reference
below). On tool-output tokens, any two implementations disagree by 0.4–0.6 nats on average. That is expected
rather than a bug: those tokens are improbable, so their log-probs move almost one-for-one with tiny logit
differences. They are loss-masked.

**fp32 LM head.** Measured on rollout 0 of the current run (the initial weights), with Hugging Face transformers
(bf16 weights) standing in for the trainer:

| Comparison, trained tokens | Mean \|Δ log-prob\| | p99 |
|---|---|---|
| Same hidden states: bf16-rounded logits vs fp32-accumulated logits | 0.0052 | 0.061 |
| HF bf16 head vs SGLang bf16 head (today's setup; production shows 0.007) | 0.0065 | 0.113 |
| One side fp32, the other bf16 | 0.0065–0.0066 | about 0.09 |
| HF fp32 head vs SGLang fp32 head (open-instruct's setup) | 0.0044 | 0.066 |

- **Rounding dominates.** bf16 rounding of the logits is most of the train/rollout gap.
- **One side is not enough.** Removing the rounding on one side only does nothing; both sides cut the gap by about
  a third.
- **What it takes.** The SGLang side is one flag. Megatron's output layer produces bf16 logits that Miles only
  upcasts afterwards, and Miles has no option to change that.

Options: F, G. Probe script, Beaker spec, HF scripts and results: `/weka/nora-default/shashankg/miles_tmax_v2/probes/`
(`sglang_probe.py`, `sglang_probe_beaker.yaml`, `lmhead_rounding.py`, `lmhead_vs_probe.py`,
`results/20261016_035343/`).

## 2026-10-16: 47-minute step after step 15

- **What happened:** step 16 trained 47 minutes after step 15 (02:41 to 03:28), against about 5 minutes per step
  before and after.
- **Evidence:**
  - From 02:40 to 03:10, the 8 engines on the then-head node decoded at 3–5 tokens/s per request, against about 90
    before and after. The 8 engines on the other rollout node stayed normal.
  - Episodes on the slow engines waited on model calls (81% of episode time in that window). Completions fell from
    about 690 to about 100 per 10 minutes, so no group completed and the trainer waited.
  - Once the engines recovered, 990 fresh episodes finished in 10 minutes.
  - The slowdown coincided with all 32 episodes of `task_009830` (02:20 to 03:20) and of `task_004592` (02:23 to
    03:19). `task_009830` runs numpy/scipy simulations that hit the 120 s step limit 103 times in 12 traced
    episodes. A sandbox sees every core of the host, so one such process can use the whole machine.
  - The head node also ran all 512 agent processes, the router and the session servers. The router failed health
    checks even against the other node's engines.
  - Sandbox command latency stayed normal (0.52 s median, against 0.41 s), so podman itself was not slow.
- **Ruled out:** Miles' fully-async producer frees each slot as its own episode finishes
  (`--rollout-submission-granularity sample` is the default under `--fully-async`), as open-instruct's pool does.
- **Comparison:** open-instruct's 4-node 64k runs (W&B group `9b_dppo_repro_jupiter`, steps 1–500 over 4 resumes):

  | Steps | Median step time | Slowest step |
  |---|---|---|
  | 1–50 | 6.8 min | 22 min |
  | 51–100 | 5.8 min | 13 min |
  | 101–200 | 8.4 min | 44 min |
  | 201–300 | 10.3 min | 40 min |
  | 301–400 | 11.6 min | 74 min |
  | 401–500 | 12.6 min | 92 min |

  So long steps happen there too (16 over 30 minutes), but none before step 105.
- **Options:** A–E, plus I and J for the long tail.
- **Caveat on counting.** Ray collapses repeated log lines (`[repeated Nx across cluster]`), so counting
  `episode done` lines undercounts episodes. Count them from the trace files instead.

## 2026-10-15: weight-sync `ReadError` (old 4-node run)

- **What happened:** the sync after step 118 took 64 s instead of the usual 1.5 s. Then 5 of the 16 trainer → engine
  HTTP requests failed with `httpx.ReadError` at the same moment, on both engine nodes, and the job failed.
- **Evidence:**
  - Not a timeout. The trainer's HTTP client has no read timeout, SGLang's handler waits indefinitely, and an
    expired timeout would raise `ReadTimeout`, `WriteTimeout` or `PoolTimeout`. Raising timeouts would not have
    helped; `--distributed-timeout-minutes 120` and `--sglang-watchdog-timeout 1800` were already set.
  - The engines kept answering 200 until 22:02:02.
  - The rollout side saw dropped session-server connections in the same minutes.
- **Reading:** connections dropped under load, consistent with the starvation found on 2026-10-16. A first check
  against the old run's traces found its slow syncs at ordinary sandbox counts, but that counted sandboxes, not CPU
  use, and could not see which node they ran on.
- **Options:** A and B to find and reduce the cause; K if it recurs; configurable HTTP timeouts would not help.
