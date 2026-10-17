# TMax port investigations

Investigations from running the port, newest first. Each records what was observed, the evidence, the options
considered and whether one was chosen. Open work lives in [ROADMAP.md](ROADMAP.md), which links back here; failure
signatures and what to do about them are in the [Beaker runbook](beaker/README.md#failures-and-what-they-mean).

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
| K | Make a dropped weight-sync reply non-fatal: treat it as "outcome unknown" after a completed broadcast and verify the engines' weight version | Survives the `ReadError` seen once in the old run | Miles core change; check upstream first | Only if it recurs |

Suggested next restart: A + B + F + H (and G once the trainer side exists), validated on a 4-GPU smoke, then
resubmitted with `--exp-name tmax_prod4node_qwen35_9b_20261016_001037` to continue from the latest checkpoint.

## Ideas to explore

Larger directions to evaluate against the evidence below. Add yours here.

## 2026-10-17: a worker replica gives up on a starved head (resume, retry 1)

- **What happened:** the podman resume's retry 1 (from 02:23, steps 160–~184, head on `10.93.2.35`) failed at 08:00:
  the worker replica on `10.93.1.243` (`holmes-cs-aus-504`) logged `Ray head 10.93.2.35:6379 unreachable 6 times in
  a row; leaving` and exited 1. Before it, the head's own podman socket timed out (`ReadTimeout`, benched at 07:50)
  and session requests to its session servers got `404`s: the head node was starved again. The weight syncs before
  were 1–3.5 s, then 53 s at 07:55. A 21-minute sync (1,286 s) after step 170 at 04:03 had not killed it.
- **Reading:** the entry script's own watchdog, not Ray, ended this attempt: about 90 s of failed `ray status` on a
  starved head. `aeb4f9ecd3` raises it to about 20 minutes (`HEAD_MISS_LIMIT` 80), matching the Ray health-check
  tolerance of `f5ef1f8273`; neither is in the running attempt, which the user chose to let retry on the old code.

## 2026-10-17: fleet Redis outage delays the gateway prod run

- **Run:** `tmax_prod4node_gateway_qwen35_9b_20261017_035830` (Beaker `01M54060ECZQ0CN4XF1A5FZPZ6`,
  `prod_4node_gateway.yaml` on branch `shashank/tmax-gateway`), from step 0, alongside the podman run.
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
- **Options:** A, B, C, E (above) and L, the gateway backend (on branch `shashank/tmax-gateway`), which takes the
  sandboxes off the nodes altogether.

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
