---
title: DeepSeek-V3.2
description: Launch recipe for DeepSeek-V3.2 (671 B total / 37 B active) — BF16 training, NSA rollout, 8 training nodes and up.
---
## 1. Model Introduction

[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) is a 37 B-active / 671 B-total Mixture-of-Experts (MoE) model. It keeps the MoE and Multi-head Latent Attention (MLA) shapes of DeepSeek-V3 and adds DeepSeek Sparse Attention (DSA): a separate indexer scores the preceding tokens, and each query attends only to the highest-scoring ones.

**Key highlights:**

- **Fine-grained MoE**: 61 layers (3 dense, then 58 MoE), 256 routed experts with top-8 plus 1 shared expert, a sigmoid router with expert bias, group-limited routing over 8 groups with top-4 groups, and a routed scaling factor of 2.5.
- **MLA with a sparse indexer**: q-LoRA rank 1536, KV-LoRA rank 512, and 128 attention heads split into a 128-dim NoPE part and a 64-dim RoPE part. The indexer runs 64 heads at head dim 128 and keeps `index_topk=2048` keys per query.
- **YaRN RoPE**: rotary base 10000 with scaling factor 40, giving 163,840 max positions.
- **Block-wise FP8 checkpoint**: the published weights are 128×128 FP8 blocks with `ue8m0` scales.

In miles, V3.2 shares its DSA attention implementation with the GLM-5 family — both select it through `--spec miles_plugins.models.glm5.glm5 get_glm5_spec`. Weight import and export run through `DeepseekV32Bridge` (`miles_plugins/mbridge/deepseek_v32.py`), which adds the indexer tensors on top of the V3 bridge. Training is BF16, so the FP8 checkpoint is cast up before conversion.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| DeepSeek-V3.2 | 37 B / 671 B, 61 layers | [deepseek-ai/DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) |

A 5-layer pruned Megatron config also ships as `deepseek-v32-5layer`. You can select it with `--megatron-model-type deepseek-v32-5layer` when you have a matching pruned checkpoint and want a single-node smoke test.

## 3. Environment Setup

Run everything inside the `radixark/miles:latest` container at `/root/miles`. The whole recipe is driven by one Typer launcher, `scripts/run_deepseek_v32.py`.

### 3.1 Launcher defaults

| Flag | Default | Use |
|---|---|---|
| `--model-org` / `--model-name` | `deepseek-ai` / `DeepSeek-V3.2` | The HF repository to download, and the stem of every derived directory name. |
| `--model-dir` | `/root/models` | Holds the HF checkpoint, the `-bf16` cast, and the Megatron `_torch_dist` directory as siblings. |
| `--model-local-dir` | `/root/models` | Node-local destination that `prepare-cp` rsyncs into; worth changing only when `--model-dir` is on shared storage. |
| `--data-dir` | `/root/datasets` | Where dapo-math-17k and aime-2024 are downloaded. |
| `--megatron-path` | `/root/Megatron-LM` | Added to `PYTHONPATH` for both conversion and training. |
| `--output-dir` | `/root/shared_data` | Training checkpoints land under `{output-dir}/{run-id}/checkpoints`. |
| `--hardware` | `B200` | One of `B200`, `B300`, `GB200`, `GB300`, `H100`, `H200`. |

Every option also binds to an env var named `MILES_SCRIPT_<FIELD_NAME_UPPER>` (for example `MILES_SCRIPT_MODEL_DIR`), with precedence CLI flag > env var > built-in default. Run `python scripts/run_deepseek_v32.py train --help` to see each option's env var name.

The launcher does not pass `--colocate`, so training and rollout occupy disjoint GPUs. The flags `--actor-num-nodes` and `--rollout-num-gpus` have no defaults, and every multi-node invocation has to supply them.

### 3.2 Download and convert

The `prepare` subcommand downloads the model and the datasets, casts the FP8 checkpoint to BF16, and converts it to a Megatron `torch_dist` checkpoint:

```bash
python scripts/run_deepseek_v32.py prepare --actor-num-nodes 8
```

The stages it runs are the following:

```bash
hf download deepseek-ai/DeepSeek-V3.2 --local-dir /root/models/DeepSeek-V3.2
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/datasets/aime-2024

python tools/fp8_cast_bf16.py \
   --input-fp8-hf-path /root/models/DeepSeek-V3.2 \
   --output-bf16-hf-path /root/models/DeepSeek-V3.2-bf16/
```

The `torch_dist` conversion then runs `tools/convert_hf_to_torch_dist.py` under `torchrun` across the Ray cluster. Both the cast and the conversion detect their own output and skip it, so re-running `prepare` after a failure is cheap. If your checkpoint is already BF16, `--from-bf16-ckpt` downloads it straight into the `-bf16` directory and skips the cast.

Because the conversion fans out over Ray with `ray.init(address="auto")`, the cluster has to be up before you call `prepare` on more than one node.

### 3.3 Multi-node Ray

Start Ray yourself and point the launcher at it:

```bash
# on the head node
ray start --head --num-gpus 8 --disable-usage-stats
# on every worker
ray start --address=${HEAD_IP}:6379 --num-gpus 8 --disable-usage-stats

export MILES_SCRIPT_EXTERNAL_RAY=1
export RAY_ADDRESS=http://${HEAD_IP}:8265
```

Without `MILES_SCRIPT_EXTERNAL_RAY=1`, the training stage runs `ray stop --force` and starts a fresh local head, tearing down the cluster that the conversion just used. When `RAY_ADDRESS` is unset the launcher submits to `http://127.0.0.1:8265`.

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
python scripts/run_deepseek_v32.py full-train \
   --actor-num-nodes 8 --rollout-num-gpus 8
```

The `full-train` subcommand chains download → FP8 → BF16 cast → optional rollout quantization → `torch_dist` conversion → training. It does not run `prepare-cp`; call that separately if you stage checkpoints onto node-local disk.

### 4.2 Individual stages

```bash
# download, cast, and convert only
python scripts/run_deepseek_v32.py prepare --actor-num-nodes 8

# re-run just the Megatron conversion
python scripts/run_deepseek_v32.py prepare-megatron-ckpt --actor-num-nodes 8

# rsync the HF checkpoint and torch_dist into --model-local-dir on every node
python scripts/run_deepseek_v32.py prepare-cp --actor-num-nodes 8

# train, assuming the stages above already ran
python scripts/run_deepseek_v32.py train --actor-num-nodes 8 --rollout-num-gpus 8
```

### 4.3 Single-node smoke test

The `--use-single-node` flag pins the run to one node with 4 training GPUs and 4 rollout GPUs, switches the parallelism to TP4 / PP1 / EP4, converts the checkpoint on that single node, and runs SGLang with 2-GPU engines. Pair it with a pruned checkpoint — the full 671 B model does not fit on 4 GPUs.

## 5. Recipe Configuration

### 5.1 Parallelism

| Stage | TP | PP | CP | EP | expert-TP | Last PP stage |
|---|---|---|---|---|---|---|
| Training, multi-node | 2 | 4 | 1 | 16 | 1 | 13 layers |
| Training, `--use-single-node` | 4 | 1 | 1 | 4 | 1 | — |
| `torch_dist` conversion, multi-node | 4 | 6 | — | 16 | 1 | 13 layers |

61 layers do not divide evenly into PP=4, so `--decoder-last-pipeline-num-layers 13` splits the training stages 16 / 16 / 16 / 13. Megatron also requires the world size to be divisible by `expert-TP × EP × PP` = 64, so `--actor-num-nodes` has to be a multiple of 8 at 8 GPUs per node.

The rest of the performance arguments are fixed by the launcher:

```bash
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
--use-dynamic-batch-size
--max-tokens-per-gpu 32768
--data-pad-size-multiplier 4096
--log-probs-chunk-size 1024
--allgather-cp  # DSA with context parallel uses the sequential allgather-CP layout
```

### 5.2 Algorithm

Using GRPO as an example, you can configure the algorithm with the following flags:

```bash
--advantage-estimator grpo
--use-kl-loss
--kl-loss-coef 0.00
--kl-loss-type low_var_kl
--entropy-coef 0.00
--eps-clip 0.2
--eps-clip-high 0.28
```

Rollout reads dapo-math-17k and scores it with `--rm-type deepscaler`: 32 prompts per rollout step, 8 samples per prompt, global batch size 256, and responses capped at 8192 tokens. Evaluation is off until you pass `--enable-eval`, which adds an aime-2024 pass every 20 rollouts at 16 samples per prompt.

The `--enable-mis` flag turns on truncated importance sampling to correct the train/inference mismatch. It writes a custom config with `use_tis: true`, `tis_mode: truncate`, and bounds `[0.5, 2.0]`, then routes the policy loss through `examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp`. Rejection sampling rides along by default and is disabled with `--no-tis-use-rs`.

### 5.3 Rollout & SGLang

```bash
--sglang-mem-fraction-static 0.8
--sglang-attention-backend nsa
--sglang-nsa-decode-backend flashmla_sparse
--sglang-nsa-prefill-backend flashmla_sparse
--sglang-kv-cache-dtype bf16
--sglang-page-size 64  # the NSA KV cache requires 64 on CUDA
--rollout-num-gpus-per-engine 8  # 2 with --use-single-node
--sglang-tp-size 8
--sglang-dp-size 8
--sglang-enable-dp-attention
--sglang-enable-dp-lm-head
--sglang-cuda-graph-max-bs 256
--sglang-moe-runner-backend flashinfer_trtllm_routed  # triton on H100 / H200
```

The launcher exports three env vars into the Ray runtime environment: `SGLANG_NSA_FORCE_MLA=1`, `SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0`, and `NVSHMEM_DISABLE_NCCL=1`.

### 5.4 Optimizer

```bash
--optimizer adam
--lr 1e-6
--lr-decay-style constant
--weight-decay 0.1
--adam-beta1 0.9
--adam-beta2 0.98
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

The Adam state is offloaded to host memory in every configuration the launcher can produce.

### 5.5 Low-precision options

| Flag | Effect |
|---|---|
| `--rollout-fp8` | Runs `tools/convert_hf_to_fp8.py --strategy block --block-size 128 128` and serves the resulting `-FP8` directory to SGLang, while Megatron stays BF16. |
| `--rollout-mxfp8` | Runs `tools/convert_hf_to_mxfp8.py`, serves the `-MXFP8` directory, and switches SGLang to `--sglang-fp8-gemm-backend flashinfer_trtllm`. |
| `--train-mxfp8` | Trains through Transformer Engine with `--fp8-format e4m3 --fp8-recipe mxfp8`. |

Both MXFP8 options require Blackwell and are rejected on H100 / H200, and the two rollout options are mutually exclusive. The `--rollout-mxfp8` path additionally keeps the MLA up-projections in BF16 — `.kv_b_proj.` on the HF side, `linear_kv_up_proj` / `linear_k_up_proj` / `linear_v_up_proj` on the Megatron side — via `--extra-high-precision-layers-hf`, `--extra-high-precision-layers-megatron`, and a generated Transformer Engine precision config.

### 5.6 Notable quirks

- The launcher always passes `--use-fault-tolerance`, and checkpoints every 20 rollouts unless you pass `--no-save`.
- Combining `--train-mxfp8` with `--fp8-param-gather` raises `NotImplementedError`; MXFP8 parameter all-gather is not wired up yet.
- Passing `--train-mxfp8` without `--rollout-mxfp8` points `--hf-checkpoint` at an `-MXFP8` directory that the prepare stage never builds, because only `--rollout-mxfp8` runs the MXFP8 conversion. Pass both flags together.
- The multi-node conversion branch passes PP=6 with `--decoder-last-pipeline-num-layers 13`, which leaves 48 layers for 5 middle stages and fails Megatron's even-split check. Until that default is fixed, convert by calling `tools/convert_hf_to_torch_dist.py` directly with a layout that divides — PP=4 with the same last-stage size matches the training layout and works.

## 6. Pairs Well With

- [Low Precision RL](/advanced/low-precision) — background for the rollout and training quantization flags above.
- [Fault Tolerance](/advanced/fault-tolerance) — enabled by default in this recipe.
- [GLM-5.2](/models/glm/glm5-2) — the other recipe built on the same DSA attention implementation.
- [Agentic Rollout](/user-guide/agentic-rollout) — V3.2 renders through the `deepseekv32` template.
