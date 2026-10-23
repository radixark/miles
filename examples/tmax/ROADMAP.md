# TMax port roadmap

Open work on the port, by priority, as of 2026-10-22. The differences from open-instruct that motivate most items
are described in [the comparison](../../docs/deep-dive/open-instruct-comparison.md); this file only tracks the
work. When an item is done, move it to [Done](#done) with its commit.

Priorities: **P0** blocks calling this a reproduction of open-instruct's run, or risks the run or a secret. **P1**
improves fidelity or visibility. **P2** is cleanup and rare cases.

## P0

| Item | Why | How |
|---|---|---|
| **Rotate the Docker Hub token and the W&B API key; scrub old W&B configs** | The Docker token was printed in Beaker job logs from 2026-09-25 until `de58153970`. The W&B key was on the logged train command until `07bf7f9d74`, and Miles uploads its arguments as the W&B config, so earlier runs' configs still hold it | Rotate both and update the Beaker secrets (`<user>_DOCKER_PAT`, `<user>_WANDB_API_KEY`); relaunch the mirrors; delete `wandb_key` from earlier runs' configs. Upstream: have Miles redact the key before uploading its config. Not confirmed done |
| **Rollout-node CPU starvation** | Evidence in [INVESTIGATIONS.md](INVESTIGATIONS.md). The engines on one rollout node slowed to a crawl twice: a 47-minute step after step 15, and a 320 s weight sync. Then Ray declared that node dead for missed heartbeats, failing retry 1. The old run's weight-sync `ReadError` fits the same pattern. The leading suspect is the up to 128 sandboxes per node, which run without CPU limits next to the engines | Option L, the gateway backend (sandboxes on a LiteRegistry podman fleet), has run production since 2026-10-20 (merged `e961a935aa`) with no starvation in 312 steps; the podman backend still has the problem. For podman, options A–C in [INVESTIGATIONS.md](INVESTIGATIONS.md#open-decisions): a per-node host sampler, `nice -n 19` sandbox commands, sandboxes off the rollout nodes. Only if the `ReadError` recurs: make a dropped sync reply non-fatal (option K) |

## P1

| Item | Notes |
|---|---|
| SGLang attention backend | SGLang picks `triton` for Qwen3.5 on B300; `trtllm_mha` is 1.75× faster at 32k context with the same log-probs. Validate in a 4-GPU Miles smoke, then pass `--sglang-attention-backend trtllm_mha` (option F) |
| fp32 LM head on both sides | Measured: the train/rollout gap drops 0.0065 → 0.0044, but only with fp32 on both sides. Pass `--sglang-enable-fp32-lm-head`, and make Megatron's output layer accumulate in fp32, which is a Miles core change (option G) |
| Token-identical first turn | SGLang's tool schema adds `"strict": false` and reorders keys: about 17 extra tokens per prompt. Render the tool from the raw dict, then diff token ids against a decoded open-instruct rollout |
| W&B and Beaker visibility | Update progress and ETA per step (needs a `BEAKER_TOKEN` secret); export `WANDB_TAGS` / `WANDB_NOTES` with commit and Beaker link |
| Fail fast on a stalled rollout | Fully async only warns every 30 s when no group completes; add a watchdog that exits so the Beaker retry resumes, or evaluate `--use-fault-tolerance` |
| Fully-async throughput metrics | `perf/rollout_time` measures the buffer drain, so `perf/tokens_per_gpu_per_sec` is meaningless and some `perf/longest_*` go negative; `rollout/zero_std/*` is computed after the filter and is always 0. Fix upstream or hide |
| open-instruct metrics worth porting | Max and std of the train/rollout log-prob gap (`debug/vllm_vs_local_logprob_diff_{max,std}`): a change in Miles's loss, where only the mean is reduced; cumulative token counter |
| Podman 5.6.2 and crun 1.14.3 in the image | apt's podman 4.9 breaks exec framing under load (44 of 70 step failures in one 4-node run). Plan: a thin image derived from the official Miles image with podman, crun and docker-py baked in, launched with beaker-gantry |
| OPD at production scale | open-instruct's terminal OPD runs are pure OPD on 4 nodes at 64k: 8 prompts x 32 samples, `--async_steps 4`, DPPO, lr 1e-6 (`scripts/general_agent/terminal/rl/opd/` on branch `opd`). Write the config, and run the 2B smoke (`smoke_opd_2b.yaml`), the only Beaker check of `scripts/models/qwen3.5-2B.py` |
| Agents near their sandboxes | Every agent process runs on the head node, which also trains; open-instruct spreads its environment actors over the nodes. Run agents on the node of their sandbox |

## P2

- **Harness fidelity:**
  - match open-instruct's tool-response framing (or keep documenting it);
  - count the pending observation in the token budget (about 1.3% of episodes end `context_full`);
  - queue a command behind a timed-out one, as open-instruct does;
  - format timeouts as floats (`120.0s`);
  - check `test.sh` inside the container;
  - choose one behavior for a vanished container;
  - add per-phase sandbox timing logs (`SWERL_SANDBOX_TIMING_LOGS`).
- **Sandboxes:**
  - a janitor for leaked containers, keyed on the `miles_tmax_sandbox` label;
  - print free disk to the job log, with an emergency prune under a threshold.
- **Beaker:**
  - `PYTHONFAULTHANDLER=1` in the env, so a native crash such as the rollout-executor segfault leaves a stack (option H);
  - find secrets by Beaker identity and check that they exist (add `BEAKER_TOKEN`);
  - a description option;
  - a `/output` result dataset;
  - make a missing dataset state fatal on resume;
  - snapshot the repository so a `git gc` on the weka checkout cannot break a later retry (bundle or full clone).
- **Training:**
  - optionally flush the KV cache after in-place updates;
  - check which FlashAttention Megatron uses on B300;
  - decide whether `max_weight_staleness` should be 3 to match `--async_steps 4` exactly (see the comparison).
- **Code hygiene:** cover the launcher with recorded-command tests.

## Done

| Item | Commit |
|---|---|
| Run watcher and trajectory checker (`tools/watch_run.py`, `tools/check_samples.py`) | the commit adding `tools/` |
| Per-submission run names; `--exp-name` continues a run; W&B run per submission | `edf2c7ae3e` |
| Multi-node replicas wait up to 1 h for each other (`synchronizedStartTimeout`) | `fd31043981` |
| Image `dev-202610151741` | `c55f57cb5b` |
| Session server v2 with the namespaced radix cache | `09053e0bc6` |
| `--distributed-timeout-minutes 120`, `--sglang-watchdog-timeout 1800`, `NCCL_DEBUG=WARN` on the 4-node config | `72033aa800` |
| Each step's trained batch dumped (`dump_training_data`), about 200 MB per step | `e1e06ce53d` |
| W&B key off the logged command line | `07bf7f9d74` |
| Head-node lookup through the GCS (Miles core) | `ef3dcd4861` |
| Adam beta2 0.999 and `--seed 42` as in open-instruct | `95bb23689a` |
| Training entropy logged (`--observe-training-entropy`) | `f882fdc500` |
| W&B entity pinned (`wandb_entity: ai2-llm`); the W&B run's URL in the Beaker description | `715206319f` |
| DPPO loss (`--policy-loss-type dppo_tv`, Miles core), the default `policy_loss` | `aecf2f49db`, `ea42bb5d4e` |
| KL to the base model logged (k3, open-instruct's `kl2`, coefficient 0) | `237a1f80d9` |
| HF snapshots kept every 20 steps (`hf_snapshot_every`: `--save-hf` at every save, pruned by the entry script) | `fecaa97c0a` |
| Per-rollout metrics in both modes (`tmax_metrics`): episodes and tools, solve rates before the filter, solved/unsolved lengths, advantages, samples so far | `735c39d953`, `a57d6f855f`, `a8fdd2bd0d` |
| Gateway backend: sandboxes on a LiteRegistry podman fleet; fleet losses and start failures drop the group; the head closes its containers when the job stops | `255d93f374` .. `1e9265bd13`, merged with the parity work in `e961a935aa` |
| Registry mirrors, `mirror_url: auto`, authenticated pulls | `0b910d7822`, `bfd6826c2b`, `dafe101cff` |
| Docker Hub token out of the job log | `de58153970` |
| Env-step exceptions become failed-step observations | `212c6ad522` |
| Multi-node layout and sandboxes spread over every node | `1eb8f5e307`, `984e017222`, `7ae3853e16`, `505ea34897` |
| Session-collect failures (2.5-3% of episodes on the old base, each dropping its group) | Gone after rebasing onto upstream's session-server and transport fixes |
| Pure on-policy distillation (`opd_pure`, open-instruct's `--opd_pure`), and recipes for `allenai/tmax-2b` and `hamishivi/Qwen3.5-2B` | `107a60581c`, `eb48263f8d`, `b7fdc6b939` |
| CUDA 13 images on Jupiter's and Ceres' 570 drivers (forward compatibility) | `f5cd9bd69e` |

## Decisions taken

- **Upstream first.** Rebase onto upstream rather than patch Miles core (2026-10-15). The session-fetch retry
  written before that rebase was dropped; it survives on `backup/tmax-terminal-rl-pre-rebase-2` (`5237ac3fac`).
- **Session server v2** with upstream's namespaced radix cache, not v1.
- **Images.** Official Miles dev images pinned per config for now; the thin derived image with beaker-gantry later.
- **Mirrors** are neither rotated nor warmed.
- **Context.** Production trains at 64k (`max_seq_len: 67584`); smoke tests use 16k.
- **Runs.** Every submission is a new run with a UTC date-time suffix; `--exp-name` continues one.
- **Loss.** DPPO binary-TV, open-instruct's (`policy_loss: dppo_tv`), since `e961a935aa`.
- **Kept on purpose:**
  - containers are removed at episode end;
  - exact global token-mean loss;
  - hard per-node sandbox caps;
  - every episode saved as text;
  - commit pinning.
- **Not ported:**
  - open-instruct's per-shard loss weighting and its dropping of packs not divisible by DP;
  - ZeRO-3 with Ulysses (Megatron TP × CP gives the same DP);
  - the tiled Liger loss (Miles' fused log-prob backend covers the memory problem);
  - metrics that are constant 0 in this recipe;
  - a reference model at KL weight 0.

## Not verified

- **Staleness origins.** Whether open-instruct's `result.model_step` is the oldest or the newest version in an
  in-flight trajectory, which decides how exact the one-version offset from Miles' oldest-token staleness is.
- **Megatron internals:** its Adam epsilon default, and whether it offers an fp32 output layer.
- **Memory limits.** Whether the 4 GB limit and OOM detection work at all with podman's `cgroups="disabled"`, on
  either side.
- **open-instruct's actor spread.** How evenly its environment actors actually spread over nodes.
- **Rollout-node CPU starvation.** Strongly suggested by timing (engines on one node crawling while CPU-heavy tasks
  ran, a node missing Ray heartbeats), but not yet measured on the hosts; option A measures it.
