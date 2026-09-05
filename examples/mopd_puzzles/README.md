# Multi-teacher OPD on short puzzles

Train two Qwen3.6 puzzle specialists, then distill their complementary capabilities into one student using routed teacher scoring.

This example uses **Qwen/Qwen3.6-35B-A3B** for the initial student and both
teacher initializations. Countdown answers are arithmetic expressions; graph
coloring answers are JSON. A bounded arithmetic parser and strict graph checker
compute rewards locally. No generated code, sandbox, or external environment is
needed. Thinking is disabled and responses are capped at 256 tokens.

> **Read the docs:** [On-policy distillation](../../docs/advanced/on-policy-distillation.md)
> describes the candidate objective, supported configurations, and domain weights.

The proof of concept trains two independent specialists from the same base model,
then distills both into a fresh base-model student. Each student response is scored
by one routed teacher; teacher weights are not averaged.

## Prerequisites

- A Miles training environment supporting Qwen3.6, including Megatron, FLA and
  Transformer Engine. The measured runtime uses PyTorch 2.13.0+cu130, Transformers 5.12.1,
  Ray 2.56.0, Transformer Engine 2.17.0, and FLA/fla-core 0.5.2. The production
  implementation in this PR was tested against Miles base `7741150a2925` and
  SGLang base `a3591427c204` plus the accompanying sparse-scoring patch.
- The HF model at `/root/models/Qwen3.6-35B-A3B` and its converted checkpoint at
  `/root/models/Qwen3.6-35B-A3B_torch_dist`.
- Reasoning Gym pinned to `49b07130b3fcd12f2d064bba7c43869543a0e7e7`.
- Eight H200s per training node for the default colocated recipe. External
  frozen teachers need additional inference GPUs.
- Candidate scoring uses an accompanying SGLang change adding
  `token_ids_logprob_positions`. Until that change is available in your runtime,
  pass `--no-sparse-scoring` to use the existing dense requested-ID interface.
  Dense scoring returns the sequence-wide union of candidates at every position
  and can be substantially more expensive for long responses.

## Prepare and screen the tasks

```bash
python -m examples.mopd_puzzles.prepare --output /root/datasets/mopd_puzzles \
  --configs countdown4 graph12 --splits train --size 10000
python -m examples.mopd_puzzles.prepare --output /root/datasets/mopd_puzzles \
  --configs countdown4 graph12 --splits dev --size 512
python -m examples.mopd_puzzles.prepare --output /root/datasets/mopd_puzzles \
  --configs countdown4 graph12 --splits test --size 1024
```

The generator checks oracle answers and deduplicates canonical puzzle identities
against all JSONL files already in the output directory. Keep the splits together
and generate them in the order above. Existing split files are never overwritten.
Combine the training files without changing their metadata:

```bash
cat /root/datasets/mopd_puzzles/countdown4-train.jsonl \
  /root/datasets/mopd_puzzles/graph12-train.jsonl \
  > /root/datasets/mopd_puzzles/mixed-train.jsonl
```

The student data source shuffles each epoch and alternates the two domains;
128-prompt batches contain 64 examples from each domain before filtering.

Serve the initial model and use `python -m examples.mopd_puzzles.screen --help`
to evaluate the development files. Use the same final prompt and verifier for
every model. Use `--max-tokens 256 --stop-at-answer` for every screening run. Under the final
protocol, S0 scored 13.086% on Countdown and 26.172% on graph coloring on 512
examples each; Countdown truncation was 0.391% and graph truncation was zero.

## Train and select teachers

Run one command on each training node, changing `--domain` to `graph_color` for
the second teacher:

```bash
python scripts/run_mopd_puzzles.py --mode teacher --domain countdown \
  --num-rollout 40 --rollout-batch-size 32 --n-samples-per-prompt 8 \
  --global-batch-size 256 --learning-rate 1e-6 \
  --eval-interval 10 --save-interval 40 \
  --checkpoint-dir /scratch/mopd/checkpoints
```

Both teachers start from the original checkpoint and receive only their own
domain's verifier reward. Evaluate each teacher on **both** development sets.
Proceed only after each improves its own domain and the teachers show useful
complementarity. Test examples are reserved for the final comparison.

The default GPU optimizer is faster than the tested four-GPU learner with CPU
optimizer offload. Colocation still pays for model swapping. The example disables its router
circuit breaker because intentional model offloading otherwise caused repeated
evaluation delays in the tested runtime. Use
`--no-colocate --actor-gpus 4 --rollout-gpus 4 --optimizer-cpu-offload` only when
that memory/performance tradeoff suits your hardware.

Use `--extra-args '--save-hf /scratch/mopd/teacher-hf'` to export an HF checkpoint
at save points. Confirm its `.complete` marker before serving it, and retain
enough local disk for both checkpoint formats. Save flags omit optimizer/RNG
state; these snapshots do not provide exact training resumption.

## Distill the teachers

Serve the two selected HF checkpoints through separate SGLang `/generate`
endpoints, with the same tokenizer and vocabulary as the student. On a separate
teacher node, run one process per GPU (change the checkpoint, GPU, and port for
the second teacher):

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path /scratch/mopd/teacher-hf --host 0.0.0.0 --port 30000 \
  --tp-size 1 --context-length 2048 --mem-fraction-static 0.8 \
  --max-running-requests 128 --disable-cuda-graph \
  --chunked-prefill-size -1 --disable-radix-cache --prefill-max-requests 1
```

The last three options are required for the validated Qwen3.6 recipe. The pinned
runtime showed a shared dense/sparse scoring discrepancy with chunked or batched
long-sequence prefill. Limiting the prefill batch fixes the measured discrepancy;
HTTP requests and the two teacher servers still run concurrently. The validated
sparse scorer sustains approximately 13 requests/s per GPU. Dense fallback also
needs these options. Dense and sparse results matched exactly with these settings across 101/512-token
workloads, scoring offsets 0/190, and concurrency 1/8. Mixed generation/scoring
traffic also passed after the empty-decode tensor fix in the SGLang change.

Set
`MILES_USE_LEGACY_ROLLOUT_V1=1`; the launcher supplies it to Ray workers because
rollout v1 collects the student's top-k candidates.

```bash
python scripts/run_mopd_puzzles.py --mode student \
  --teacher-urls 'countdown=http://teacher-a:30000/generate graph_color=http://teacher-b:30000/generate' \
  --candidate-top-k 16 --loss-mode topk-candidate --reward-refresh --resident-models \
  --domain-balance static --num-rollout 40 --eval-interval 10 --save-interval 40 \
  --checkpoint-dir /scratch/mopd/checkpoints
```

The tested eight-H200 student command keeps learner and rollout models resident.
`--resident-models` disables swapping and bounds SGLang to 8,192 cached tokens
and 128 running/Mamba-cache requests. It requires colocation. Omit it to use
the default offloaded layout when GPU memory is tighter.

Each response is scored only by its routed teacher. Candidate IDs and teacher
scores remain fixed across learner updates. Before optimization, the old learner
scores the fixed support once and caches the PPO denominator. Reward refresh
uses current learner scores while preserving that denominator. The original
SGLang candidate scores remain in serialized rollout samples for diagnosis.
The launcher enables dual clipping with bound 3.0. `--use-rollout-logprobs`
skips the old learner forward; the initial Qwen3.6 experiment showed unstable
tail ratios with this option, so learner caching is the default. Static domain balancing targets equal domain mass
under the configured loss reduction. `--domain-balance gap` additionally scales
the domain budget using the measured distillation gap.

For sampled-token OPD, pass `--loss-mode legacy --candidate-top-k 0`.
For legacy scalar top-k OPD, pass `--loss-mode legacy --no-reward-refresh`.
Compare the routed student against single-teacher routes and the unchanged
initial student, using identical data, token budgets and evaluation settings.

For an existing Ray cluster, set `MILES_SCRIPT_EXTERNAL_RAY=1` and pass
`--no-cleanup-processes` to preserve independent services. The default launcher
cleanup assumes a dedicated training node.

Authenticate W&B using a restricted NETRC file and the `NETRC` environment
variable. `--wandb-project` and `--wandb-team` configure the destination; the
launcher never puts an API key into a printed command.

## Correctness checks

`python -m examples.mopd_puzzles.check_scoring --help` exposes the live
dense-versus-sparse score oracle. Run it against each teacher before distillation.
`python -m examples.mopd_puzzles.benchmark_scoring --help` adds warm concurrency
and length stress tests. Use `--require-replica-parity` only when endpoints load
the identical checkpoint. The tests cover candidate loss and gradients, fixed reward/old-policy tensors,
TP reduction, CP/DP transport, malformed scoring responses, routing, domain
weights, sample retries, and the puzzle verifiers. The full-model runs used learner TP1/CP1/EP8. TP2 candidate gradients were
checked on B200 GPUs; CP/DP transport was checked with focused fixtures.

## Measured proof of concept

Each teacher trained independently for 40 updates on its domain's verifier reward
(32 prompts × 8 completions per update). The routed student and both single-teacher
mixed-data controls each trained from the original model for 40 updates of 128
prompts with top-16 candidates, LR 1e-6, static equal-domain mass, and reward refresh.
All training runs used one seed. Training sets contain 10,000 puzzles per domain;
dev sets contain 512. Checkpoints were selected by equal-domain dev accuracy over
updates 20 and 40 before evaluating the selected model on 1,024 heldout puzzles
per domain. Evaluation uses TP1, temperature 0, and the same 256-token answer cap.

| Model | Countdown | Graph coloring | Macro |
|---|---:|---:|---:|
| Initial student | 9.86% | 29.00% | 19.43% |
| Countdown-only OPD, update 20 | 37.21% | 31.05% | 34.13% |
| Graph-only OPD, update 40 | 8.98% | 87.40% | 48.19% |
| Routed MOPD, update 40 | **39.36%** | **90.43%** | **64.89%** |

MOPD's gains over the initial model are +29.49 points on Countdown (paired 95%
interval +26.07 to +32.91) and +61.43 on graph coloring (+58.11 to +64.65).
Its macro gains over the single-teacher controls are +30.76 points (+28.91 to
+32.67) and +16.70 (+14.75 to +18.65). Intervals use 5,000 paired puzzle bootstrap
resamples; they do not quantify variation across training seeds. Median response
lengths are 19 and 101 tokens, with truncation rates 0.49% and 0%.

On eight colocated H200s with two external one-GPU teachers, keeping models
resident reduced median warm-step duration from 48.92 to 20.31 seconds across
34 steps per configuration. The old-learner score pass costs 0.433 seconds.
This is a configuration comparison: loss settings and residency changed together,
and teachers were shared by concurrent runs. Startup, export, and evaluation
boundaries are excluded. Peak observed total GPU memory was 122,985 MiB.

Validation includes 196 focused Miles tests, 25 SGLang scoring tests, a TP2
GPU gradient oracle (maximum error 1.79e-7), three sampled-OPD GPU updates,
and eight candidate gap/refresh GPU updates with four updates per rollout.
The launcher suite passed 502 checks with four unrelated base-revision AMD
launcher errors. Broader multi-seed, RouteOPD, mixed-RL, and gap/refresh quality
ablations have not been run. General batched/chunked Qwen3.6 scoring remains
outside the validated serving configuration above.
