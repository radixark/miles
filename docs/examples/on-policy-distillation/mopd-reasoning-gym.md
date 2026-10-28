---
title: "Multi-Teacher OPD on reasoning_gym (Qwen3.6-35B-A3B)"
description: "Qwen3.6-35B-A3B distilled from two reasoning_gym GRPO teachers on two B200 or GB300 nodes, synchronous or one-step overlap."
# Generated from examples/on_policy_distillation/mopd_reasoning_gym/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Distill Qwen3.6-35B-A3B from two frozen GRPO teachers, one per reasoning_gym task, with
sampled-token reverse-KL on-policy distillation. Each rollout is scored only by its own task's
teacher. The launcher is `scripts/run_qwen3_6_35b_a3b_mopd.py`; this directory holds the data
prep, the rollout hooks, the teacher server and optional analysis tooling.

| File | Role |
|---|---|
| `scripts/run_qwen3_6_35b_a3b_mopd.py` | Launcher. `prepare` downloads the student and both teachers, builds the prompts and converts the student; `execute` submits the job. |
| `prepare_data.py` | Builds the train pool and the held-out sets from reasoning_gym (run by `prepare`). |
| `hooks.py` | Rollout hooks: Miles' OPD reward plus timing stamps, and the held-out verifier. |
| `serve_teachers.py` | Serves both teachers with SGLang on the teacher node. |
| `analyze.py`, `make_report.py` | Optional: per-step tables, plots and an HTML report from a `--dump-dir` run. They need matplotlib. |

## Workload

| | |
|---|---|
| Student | `Qwen/Qwen3.6-35B-A3B` (bf16) |
| Teachers | `semianalysisai/Qwen3.6-35B-A3B-caesar-cipher-GRPO-20260924` @ `dfedf677` (step 125) and `semianalysisai/Qwen3.6-35B-A3B-simple-geometry-GRPO-20260925` @ `679d4b66` (step 50) |
| Data | reasoning_gym 0.1.25 default configs. Train pool: seed 20000, 10k prompts per task, shuffled together. Held-out: seed 10000, 128 prompts per task. System prompt `SYSTEM_PROMPTS["default"]`, thinking on, prompts over 2,048 tokens dropped. |
| Routing | `--opd-teacher-urls caesar_cipher=... simple_geometry=...` with `--opd-teacher-key task` |
| Objective | Sampled-token reverse-KL OPD: `A_t = log π_T(y_t) − log π_old(y_t)`. PPO clip 0.2 / 0.28, no task reward, no KL loss, no entropy bonus. Token-mean loss (`--calculate-per-token-loss`) by default. |
| Schedule | 20 updates × 128 prompts × 1 response, one optimizer step per update |
| Sampling | Temperature 1.0, top-p 1.0, 30,720 response tokens (32,768 context) |
| Optimizer | Adam, lr 1e-6 constant, β = (0.9, 0.999), weight decay 0, grad clip 1.0, bf16 moments |
| Eval | Held-out, avg@4, temperature 1.0, 30,720 tokens. Exact match (`score_answer == 1`) on the last `<answer>` after `</think>`, the teachers' own reward. |

## Two setups

The launcher ships two setups, each on two nodes with the same algorithm, batch and hyperparameters.
Node A runs the launcher (trainer, and rollout where the setup puts it there); node B serves the
teachers.

| Setup | `--layout` | When to use it |
|---|---|---|
| Setup 1: synchronous | `b200-sync`, `gb300-sync` | The matched baseline: run it **without** `--mtp-spec-decode` when comparing against another framework's synchronous colocated run. |
| Setup 2: one-step overlap | `b200-async`, `gb300-async` | The recommended Miles setup: run it **with** `--mtp-spec-decode`. |

Both setups keep token-mean loss (`--calculate-per-token-loss`) and `--log-probs-backend fused` on.

## Before either setup

On both nodes, install the data dependency and put `--model-dir`, `--data-dir` and `--dump-dir` on
storage both nodes share, or copy the files. Node B loads the teachers. In the one-step-overlap
setup, node B's rollout engines also load the student, and the rollout manager, which Ray may place
on either node, reads the prompts and writes the rollout dumps.

```bash
pip install reasoning-gym==0.1.25
```

The launcher on node A downloads the student and both teachers, builds the prompts and converts
the student before it trains (`--skip-prepare` skips this). Node B serves the teachers, so
download them there first; with shared storage one download serves both nodes.

```bash
hf download semianalysisai/Qwen3.6-35B-A3B-caesar-cipher-GRPO-20260924 \
    --revision dfedf677a5bf08a451733ec48c845c2661a6ab14 --local-dir /root/models/teacher-caesar_cipher
hf download semianalysisai/Qwen3.6-35B-A3B-simple-geometry-GRPO-20260925 \
    --revision 679d4b66018c03249a3bfd0ecca3195ed9cd0e0d --local-dir /root/models/teacher-simple_geometry
```

Do not run the launcher on node B: its cleanup kills every SGLang process on its node, teachers
included. `serve_teachers.py` staggers the two teacher servers by 20 s, because SGLang DP servers
started together race for the same internal ports.

## Setup 1: synchronous (matched baseline)

Trainer and rollout engines share node A's GPUs and take turns (`train.py --colocate`). The trainer
takes the old-policy log-probs from its own training forward (`--skip-actor-forward-only`, one
optimizer step per update); `--no-skip-actor-forward-only` restores the separate forward-only pass.
Weights move to the rollout engines over CUDA IPC on node A.

| `--layout` | Node A | Node B |
|---|---|---|
| `b200-sync` | 8 B200 shared by the trainer (EP8) and 8 TP1 rollout engines | caesar_cipher teacher DP4 on GPUs 0–3, simple_geometry DP4 on 4–7 |
| `gb300-sync` | 4 GB300 shared by the trainer (EP4) and 4 TP1 rollout engines | caesar_cipher teacher DP2 on GPUs 0–1, simple_geometry DP2 on 2–3 |

Node B, teachers:

```bash
# b200-sync
python examples/on_policy_distillation/mopd_reasoning_gym/serve_teachers.py --models-dir /root/models \
    --teacher caesar_cipher=0,1,2,3 --teacher simple_geometry=4,5,6,7 --log-dir /root/shared_data/teachers
# gb300-sync
python examples/on_policy_distillation/mopd_reasoning_gym/serve_teachers.py --models-dir /root/models \
    --teacher caesar_cipher=0,1 --teacher simple_geometry=2,3 --log-dir /root/shared_data/teachers
```

Node A, once node B prints `READY`. The launcher starts its own single-node Ray cluster:

```bash
# matched baseline
python scripts/run_qwen3_6_35b_a3b_mopd.py --layout b200-sync --teacher-host $NODE_B \
    --dump-dir /root/shared_data/mopd-dump
# the same with MTP decoding
python scripts/run_qwen3_6_35b_a3b_mopd.py --layout b200-sync --mtp-spec-decode --teacher-host $NODE_B \
    --dump-dir /root/shared_data/mopd-dump
```

Replace `b200-sync` with `gb300-sync` on 2 × 4 GB300.

**Expected results** (`gb300-sync`, 20 updates; "Validation on 8× GB300" below says how these runs differ
from this launcher):

| Configuration | Final caesar_cipher / simple_geometry | Median step, excl. eval | Training GPU-hours |
|---|---|---|---|
| MTP decoding + token-mean loss | 80.1% / 99.8% | 198 s | 9.1 |
| No MTP, Miles' per-sample loss | 75.4% / 99.8% | 291 s | 13.1 |

The matched baseline itself (no MTP, token-mean loss) was not run on its own. MTP decoding is
lossless, so expect its accuracy to match the first row. Expect its step time near the second
row's, less the old-policy forward that `--skip-actor-forward-only` now removes (10–30 s per step
in that run). MTP cut rollout time 37% and step time 28% on this layout.

## Setup 2: one-step overlap (recommended)

The trainer and the rollout engines run on separate GPUs (`train_async.py` without
`--fully-async`): rollout k+1 generates while step k trains. Every sample is exactly one update
stale and nothing is dropped. The old-policy log-probs are the rollout engines' behavior log-probs
(`--use-rollout-logprobs`), so the trainer runs no forward-only pass. Weights cross nodes over
NCCL after each step.

| `--layout` | Node A | Node B |
|---|---|---|
| `b200-async` | 8 B200 for the trainer (EP8) | 6 TP1 rollout engines on GPUs 0–5, caesar_cipher teacher on 6, simple_geometry on 7 |
| `gb300-async` | 4 GB300 for the trainer (EP4) | 2 TP1 rollout engines on GPUs 0–1, caesar_cipher teacher on 2, simple_geometry on 3 |

This setup needs one Ray cluster across both nodes, started by hand:

- The head runs on node A and exposes all of node A's GPUs.
- The worker on node B exposes only the rollout GPUs; the teacher GPUs stay outside Ray.
- Miles fills the trainer's GPUs first, in node-IP order. Node A must therefore have the
  numerically lower IP. Before submitting, the launcher checks that the first GPU node by IP has
  exactly the trainer's GPU count.
- On RoCE clusters whose nodes sit on different subnets, NCCL picks a link-local GID and the first
  cross-node weight broadcast fails with `IBV_WC_RETRY_EXC_ERR`. Export `NCCL_IB_GID_INDEX=3` (the
  routable RoCE v2 GID) on both nodes before `ray start`, and pass
  `--extra-env-vars "NCCL_IB_GID_INDEX=3"` to the launcher. Other clusters can leave both out.

Node B, teachers on the GPUs Ray will not see:

```bash
# b200-async
python examples/on_policy_distillation/mopd_reasoning_gym/serve_teachers.py --models-dir /root/models \
    --teacher caesar_cipher=6 --teacher simple_geometry=7 --log-dir /root/shared_data/teachers
# gb300-async
python examples/on_policy_distillation/mopd_reasoning_gym/serve_teachers.py --models-dir /root/models \
    --teacher caesar_cipher=2 --teacher simple_geometry=3 --log-dir /root/shared_data/teachers
```

Ray cluster (`b200-async`; for `gb300-async` use `--num-gpus 4` on the head and
`CUDA_VISIBLE_DEVICES=0,1 ... --num-gpus 2` on the worker):

```bash
# both nodes, RoCE clusters across subnets only
export NCCL_IB_GID_INDEX=3
# node A (lower IP)
ray start --head --node-ip-address $NODE_A --num-gpus 8 --disable-usage-stats
# node B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ray start --address $NODE_A:6379 --node-ip-address $NODE_B \
    --num-gpus 6 --disable-usage-stats
```

Node A, once node B prints `READY`:

```bash
MASTER_ADDR=$NODE_A MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_qwen3_6_35b_a3b_mopd.py \
    --layout b200-async --mtp-spec-decode --teacher-host $NODE_B \
    --dump-dir /root/shared_data/mopd-dump --extra-env-vars "NCCL_IB_GID_INDEX=3"
```

Replace `b200-async` with `gb300-async` on 2 × 4 GB300.

**Expected results** (`gb300-async`, 20 updates; see "Validation on 8× GB300" below):

| Configuration | Final caesar_cipher / simple_geometry | Median step, excl. eval | Training GPU-hours |
|---|---|---|---|
| MTP decoding + token-mean loss | 78.5% / 99.6% | 173 s | 7.9 |

On 2 × 4 GB300 this setup trains on 4 GPUs while only 2 generate, so the step is rollout-bound;
`b200-async` gives rollout 6 GPUs. A cross-tray weight sync took about 10 s, and trainer and
rollout log-probs differed by 0.016 on average.

## Choices shared by both setups

These keep the math unchanged:

- **EP only, no TP/CP and no CPU offload.** A 3B-active MoE's optimizer state fits on the trainer GPUs.
- **TP1 rollout engines.** The bf16 student is 70 GB; TP > 1 only adds all-reduce latency to every decode step.
- **`--log-probs-backend fused`.** A 30,720-token response against the 248k vocabulary is a 28 GiB fp32 logits copy under the torch backend.
- **Token-mean loss** (`--token-mean-loss`, on by default). Miles' default averages within each sample first, which gives each token of a ~10k-token caesar_cipher response about a third of the weight of a simple_geometry token.
- **MTP speculative decoding** (`--mtp-spec-decode`, off by default). Qwen3.6's MTP head drafts for SGLang (accept length ~2.7), and sampling stays lossless. It speeds up the single long sequence that sets rollout time. Report it separately from a matched baseline.
- **Teachers outside Ray**, one SGLang server per teacher with DP replicas (TP1, radix cache off since n = 1).

`--dump-dir` turns on `--dump-details` and the Miles dashboard. That records per-rank phase
timings, GPU utilization, SGLang engine metrics, and each sample's generation and teacher-scoring
timestamps. `analyze.py` turns a run's dump into `steps.csv`, `summary.json` and plots;
`make_report.py` builds an HTML page from them.

## Validation on 8× GB300

Both setups ran end to end on 2 × 4 GB300 with `radixark/miles:latest` (Transformer Engine
2.17.0+miles, flash-attn-4 4.0.0b35, SGLang 0.5.23), using the `gb300-sync` and `gb300-async`
layouts. Those runs predate the Miles fixes this launcher relies on and used equivalent
workarounds in their own hooks instead: a FlashAttention-4 causal-window patch, chunked TE
FusedAdam calls, and extracting the teacher log-probs inside the reward hook. They also ran the
separate old-policy forward in the synchronous setup, which `--skip-actor-forward-only` now removes.
This launcher as committed, and the B200 layouts, have not been run on GPU.

Held-out accuracy (avg@4) of the base student is 26.2% caesar_cipher / 55.5% simple_geometry; the
teachers' cards report 73.3% / 99.8%.

- **Rollout is 80–85% of every step.** Half of each batch is scored by 38 s and 90% by 166 s, but
  the step waits until 224 s (medians, synchronous without MTP) for the last sample. Nearly every
  batch has a caesar_cipher response that runs to the 30,720-token cap and finishes decoding almost
  alone. More rollout GPUs barely help this tail; faster single-sequence decoding does.
- **MTP decoding** cut rollout time 37% and step time 28% in the synchronous setup (updates 1–4:
  228 s → 144 s rollout, 303 s → 219 s step, longest-sample decode 212 → 315 tok/s), with the same
  training accuracy and response lengths. The untrained base MTP head accepted 2.6–2.9 tokens.
- **Token-mean loss** raised caesar_cipher from 75.4% to 80.1%, with training accuracy at update 4
  of 0.87 against 0.68. One seed per setting; the eval's standard error is about 2–3 points.
- **Teachers** score a rollout in 0.05–0.4 s and are idle most of the step. Each in-loop eval
  (1,024 rollouts of up to 30,720 tokens) takes 5–7 min; `analyze.py` reports it separately.

## Reporting a comparison

When comparing this recipe with another framework's run of the same workload, keep the GPU count
and type, objective and batch the same; report eval separately from training wall-clock and cost;
count only trained tokens in cost per token and report dropped or filtered rollouts (and their
teacher scoring) on their own line; use one MFU definition with the right per-GPU peak; and for
async runs state the staleness bound and how many rollouts were dropped.
