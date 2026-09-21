---
title: "Multi-teacher OPD on short puzzles"
description: "Multi-teacher OPD with Qwen3.6 specialists on short, locally verifiable puzzles."
# Generated from examples/mopd_puzzles/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
> **Read the docs:** [On-policy distillation](/advanced/on-policy-distillation)
> describes the candidate objective and supported configurations.

Train two Qwen3.6 specialists on Reasoning Gym's Countdown and graph-coloring
puzzles, then distill them into a fresh student through the existing Miles OPD
teacher routes. Reasoning Gym generates the puzzles; Miles supplies rollout and
evaluation. The small local verifiers use only the Python standard library.

## Setup and data

Use a Qwen3.6-capable Miles runtime, eight H200s for training, and additional
GPUs for the frozen teachers. Place `Qwen/Qwen3.6-35B-A3B` and its converted
`Qwen3.6-35B-A3B_torch_dist` checkpoint under `/root/models`, or set `--model-dir`.
Install Reasoning Gym only in the data-preparation environment. Training and
evaluation workers consume the generated JSONL files without this dependency:

```bash
pip install "reasoning-gym @ git+https://github.com/open-thought/reasoning-gym.git@49b07130b3fcd12f2d064bba7c43869543a0e7e7"
python -m examples.mopd_puzzles.prepare --output /root/datasets/mopd_puzzles \
  --configs countdown4 graph12 --splits train --size 10000
python -m examples.mopd_puzzles.prepare --output /root/datasets/mopd_puzzles \
  --configs countdown4 graph12 --splits dev --size 512
python -m examples.mopd_puzzles.prepare --output /root/datasets/mopd_puzzles \
  --configs countdown4 graph12 --splits test --size 1024
cat /root/datasets/mopd_puzzles/countdown4-train.jsonl \
  /root/datasets/mopd_puzzles/graph12-train.jsonl \
  > /root/datasets/mopd_puzzles/mixed-train.jsonl
```

Keep all splits in that directory and generate them in order: the adapter checks
oracle answers and deduplicates puzzle identities against existing JSONL files.
It adds the Miles prompt, label, and teacher route. Existing split files are
never overwritten.
The data-source adapter uses Miles' buffering/resume support and alternates domains
for equal batches; Reasoning Gym's weighted sampling does not guarantee that count.

Answers use one `<answer>...</answer>` block, with thinking disabled and a
256-token cap. The reward adapter retains two differences from the pinned
library: exact, bounded four-operator arithmetic for Countdown, and strict JSON
keys/integer colors for graph coloring. Reasoning Gym's Countdown
scorer uses general SymPy parsing and partial rewards. These checks preserve the
binary scoring rules used for the results below.

## Train teachers and student

Train each teacher independently from the initial checkpoint, changing `--domain`
to `graph_color` for the second run:

```bash
python scripts/run_mopd_puzzles.py --mode teacher --domain countdown \
  --num-rollout 40 --rollout-batch-size 32 --n-samples-per-prompt 8 \
  --global-batch-size 256 --eval-interval 10 --save-interval 40 \
  --checkpoint-dir /scratch/mopd/checkpoints \
  --extra-args '--save-hf /scratch/mopd/teacher-hf'
```

The standard Miles evaluator scores both development sets at each evaluation
point. Select complementary teachers on development accuracy, reserving test data
for the final comparison. Confirm the HF export's `.complete` marker before serving.
Saved checkpoints omit optimizer/RNG state and cannot resume training exactly.

Serve each selected teacher on a separate GPU, with the student's tokenizer:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path /scratch/mopd/teacher-hf --host 0.0.0.0 --port 30000 \
  --tp-size 1 --context-length 2048 --mem-fraction-static 0.8 \
  --max-running-requests 128 --disable-cuda-graph \
  --chunked-prefill-size -1 --disable-radix-cache --prefill-max-requests 1
```

The last three flags are required by the validated Qwen3.6 scoring configuration.
Sparse scoring requires [SGLang PR #38098](https://github.com/sgl-project/sglang/pull/38098);
use `--no-sparse-scoring` for the existing dense requested-ID API. Serving API
correctness and throughput checks belong with that SGLang change.

```bash
python scripts/run_mopd_puzzles.py --mode student \
  --teacher-urls 'countdown=http://teacher-a:30000/generate graph_color=http://teacher-b:30000/generate' \
  --candidate-top-k 16 --loss-mode topk-candidate --reward-refresh \
  --domain-balance static --resident-models --num-rollout 40 \
  --eval-interval 10 --save-interval 40 --checkpoint-dir /scratch/mopd/checkpoints
```

The launcher supplies `MILES_USE_LEGACY_ROLLOUT_V1=1`, caches old-learner candidate
scores, and enables dual clipping at 3.0. `--resident-models` keeps colocated models
on GPU with bounded caches; omit it for model offloading. The router circuit breaker
is disabled because intentional offloading otherwise delayed evaluation in this runtime.
For an existing Ray cluster, set `MILES_SCRIPT_EXTERNAL_RAY=1` and pass
`--no-cleanup-processes`. W&B authentication uses `NETRC`; `--wandb-project` and
`--wandb-team` set its destination.

## Evaluate with Miles

Use the same launcher with zero rollouts to evaluate an existing TP1 server.
`--debug-rollout-only` skips learner initialization and weight updates:

```bash
python scripts/run_mopd_puzzles.py --mode teacher --num-rollout 0 --save-interval 0 \
  --no-colocate --actor-gpus 1 --rollout-gpus 1 --no-cleanup-processes \
  --extra-args '--debug-rollout-only --no-offload-rollout --rollout-external \
    --rollout-external-engine-addrs teacher-a:30000 \
    --hf-checkpoint /scratch/mopd/teacher-hf \
    --save-debug-rollout-data /scratch/mopd/eval/{rollout_id}.pt'
```

This uses the development files by default. For final heldout evaluation, add
`--eval-prompt-data countdown /root/datasets/mopd_puzzles/countdown4-test.jsonl graph_color /root/datasets/mopd_puzzles/graph12-test.jsonl`
to `--extra-args`. Miles logs accuracy and truncation per dataset and saves the
samples in `eval_0.pt`. Use the same prompts, TP1 serving, temperature 0, and
256-token cap for the initial model, teachers, and distilled students.

## Reference result

Each teacher received 40 verifier-RL updates. The routed student and both
single-teacher controls received 40 candidate-OPD updates, with top-16 candidates,
LR 1e-6, static equal-domain weights, and reward refresh. Checkpoints were selected
at updates 20/40 by macro development accuracy, then scored on 1,024 test puzzles
per domain. All training runs used one seed.

| Model | Countdown | Graph coloring | Macro |
|---|---:|---:|---:|
| Initial student | 9.86% | 29.00% | 19.43% |
| Countdown-only OPD | 37.21% | 31.05% | 34.13% |
| Graph-only OPD | 8.98% | 87.40% | 48.19% |
| Routed MOPD | **39.36%** | **90.43%** | **64.89%** |

[PR #3116](https://github.com/radixark/miles/pull/3116) records the original runtime,
confidence intervals, serving measurements, and validation. These are prior-run
results, not a new training run of this simplified example. The experiment does
not compare candidate OPD against routed legacy OPD or isolate gap/refresh gains.
