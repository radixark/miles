---
title: DeepSeek-V4 Flash
description: Launch recipe for DeepSeek-V4-Flash (284 B) — torch_dist or direct-HF initialization, FP8 / MXFP8 / MXFP4-expert precision, H200 or GB300.
---
DeepSeek V4 training tracking issue: [`radixark/miles#1046`](https://github.com/radixark/miles/issues/1046).

## 1. Model Introduction

[DeepSeek-V4-Flash](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8) is a 13 B-active / 284 B-total MoE model with a substantially different attention stack from V3/R1. It ships in the `radixark/miles:latest` image. The larger [DeepSeek-V4-Pro](/models/deepseek/deepseek-v4-pro) shares the same V4 architecture family at Pro scale.

**Key highlights:**

- **Hybrid Attention (CSA + HCA)**: combines **Compressed Sparse Attention** (light compression) and **Heavily Compressed Attention** (heavy compression) layers — DeepSeek's official V4 name (see [HF model card §Introduction](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash#introduction)). Implementation uses low-rank Q (`q_lora_rank=1024`), single-head latent KV (`head_dim=512`), grouped output projection (8 groups, LoRA rank 1024). A learned topk **indexer** (`index_topk=512`, 64 heads × 128 dim) picks 512 KV per query at runtime, inheriting V3.2's DSA-style design.
- **KV compressors**: 44-element compression schedule `compress_ratios = [0, 0, 4, 128, 4, 128, …, 4, 0]` — first / last few layers are uncompressed (ratio 0), middle layers alternate 4× (CSA) and 128× (HCA). The compressor RoPE has its own base (`compress_rope_theta=160000`), separate from the main attention RoPE.
- **Hyper-connection (HC) routing**: each layer expands hidden state into `hc_mult=4` parallel streams and recombines via sinkhorn-normalized mixing. Pipeline-parallel buffers are 4-D `[s, b, hc_mult, d]` instead of 3-D.
- **YaRN RoPE on main attention**: `rope_theta=10000`, YaRN `factor=16`, `original_max_position_embeddings=65536` → effective context length **1,048,576 tokens (1 M)**. Per-head learnable attention sinks (one scalar per head, added to softmax denominator).
- **FP8 weights with simulated FP8 QAT** on indexer and compressor activations.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| DeepSeek-V4-Flash | 13 B / 284 B | [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) |
| DeepSeek-V4-Flash-FP8 | 13 B / 284 B | [sgl-project/DeepSeek-V4-Flash-FP8](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8) |
| DeepSeek-V4-Flash-0731 | 13 B / 284 B | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |

`DeepSeek-V4-Flash` and `DeepSeek-V4-Flash-0731` are official releases with MXFP4 routed experts;
`DeepSeek-V4-Flash-FP8` repackages `DeepSeek-V4-Flash` with blockwise-FP8 experts. On the `torch_dist`
path, the launcher's `prepare-fp8` stage (`tools/convert_mxfp4_to_fp8.py`) casts MXFP4 experts
losslessly to blockwise FP8, after which the pipeline is identical to `DeepSeek-V4-Flash-FP8`.
The direct-HF path ([§3.4](#34-initialization-and-rollout-weights)) trains from the official
checkpoint without this cast.

## 3. Quick start

### 3.1 One-line launch

One command runs the full pipeline — dataset download, FP8 → BF16 cast, distributed `torch_dist` conversion, and the training loop:

```bash
# Pull the image:
docker pull radixark/miles:latest

# 8-node Flash run (colocated), inside the container
cd /root/miles
python scripts/run_deepseek_v4.py full-train \
   --model-name DeepSeek-V4-Flash-FP8 \
   --num-nodes 8 --num-gpus-per-node 8

# 16-node disaggregated run: 8 actor (training) nodes + 8 dedicated rollout nodes
python scripts/run_deepseek_v4.py full-train \
   --model-name DeepSeek-V4-Flash-FP8 \
   --num-nodes 16 --num-gpus-per-node 8 --rollout-num-nodes 8
```

```bash
# Official MXFP4 release (0731) — adds an MXFP4->FP8 cast before the same pipeline.
# 8-node GB300 run (32 GPUs):
python scripts/run_deepseek_v4.py full-train \
   --model-name DeepSeek-V4-Flash-0731 \
   --num-nodes 8 --num-gpus-per-node 4
```

The `full-train` subcommand chains `prepare-download → [prepare-fp8 →] prepare-single → prepare-spmd → train` (`prepare-fp8` only for the MXFP4 releases `DeepSeek-V4-Flash` and `DeepSeek-V4-Flash-0731`). Each stage has a sentinel-based skip so you can re-run safely after the first invocation. With `--init-model-source hf` ([§3.4](#34-initialization-and-rollout-weights)), it skips `prepare-spmd` and runs the casts only when a converted rollout checkpoint needs them.

### 3.2 Launcher path defaults

The Python launcher (`scripts/run_deepseek_v4.py`) takes its path arguments from CLI flags. The defaults are:

| Flag | Default | Use |
|---|---|---|
| `--data-dir` | `/root/datasets` | HF datasets (e.g. dapo-math-17k, …) |
| `--model-dir` | `/root/models` | parent directory holding the HF checkpoint and Megatron `_torch_dist` artifacts as separate sibling sub-directories |
| `--model-local-dir` | unset → same as `--model-dir` | local NVMe path on each node; when it differs from `--model-dir`, every trainer rsyncs the HF checkpoint and `_torch_dist` here before training starts, so the trainer reads from local disk instead of shared storage (only worth setting when `--model-dir` is on shared/remote storage) |
| `--save-dir` | `/root/models` | training checkpoints under `{save-dir}/{run-id}/checkpoints/` |

You can override these via the CLI flags above or equivalently via env vars — every launcher option binds to `MILES_SCRIPT_<FIELD_NAME_UPPER>` (e.g. `MILES_SCRIPT_MODEL_DIR`), with precedence CLI flag > env var > built-in default; run `train --help` to see each option's `[env var: …]` name.

### 3.3 Colocated vs. disaggregated rollout

By default the launcher runs **colocated**: training and SGLang rollout share all `--num-nodes × --num-gpus-per-node` GPUs. Pass `--rollout-num-nodes N` (`0 < N < --num-nodes`) to run **disaggregated**: `N` nodes serve rollout, the rest train. The verified parallelism recipes are keyed on the **training** nodes, so the 8-node Flash recipe in disaggregated form is `--num-nodes 16 --rollout-num-nodes 8` (8 train + 8 rollout — the validated layout).

### 3.4 Initialization and rollout weights

`--init-model-source` selects where the trainer's initial weights come from:

- `torch_dist` (default): `full-train` casts the HF checkpoint to BF16 and converts
  it to a Megatron `torch_dist` checkpoint (§4.2), which the trainer loads.
- `hf`: Megatron-Bridge builds the model and loads the HF checkpoint directly
  (`--megatron-to-hf-mode bridge`), so no `torch_dist` conversion runs. This
  requires `--dsv4-impl megatron`, the default.

`--rollout-weight-source` selects where SGLang's weights come from:

- `checkpoint` (default): SGLang loads a checkpoint in the rollout precision
  ([§3.5](#35-training-and-rollout-precision)), which `full-train` converts first if needed.
- `trainer`: SGLang reads only the model layout and starts with dummy weights;
  the trainer sends the actual weights in the weight update that precedes the
  first rollout. The layout comes from the source HF checkpoint, with the
  routed-expert override of `--rollout-precision` applied. BF16 rollout requires `checkpoint`.

Together, `--init-model-source hf --rollout-weight-source trainer` starts RL from
the official `DeepSeek-V4-Flash` checkpoint with no offline weight conversion:

```bash
python scripts/run_deepseek_v4.py full-train \
   --model-name DeepSeek-V4-Flash --hardware GB300 \
   --num-nodes 8 --num-gpus-per-node 4 \
   --init-model-source hf --rollout-weight-source trainer \
   --train-precision mxfp8-mxfp4-routed-expert \
   --rollout-precision mxfp8-mxfp4-routed-expert
```

In either mode, checkpoints saved during training are Megatron distributed
checkpoints under `--save-dir`, and resuming loads them.

### 3.5 Training and rollout precision

`--train-precision` and `--rollout-precision` accept the same values. Each value
names the routed-expert format, so equal values align the routed experts between
training and rollout. Rollout keeps the other quantized weights in block-wise FP8,
the layout of the official checkpoint: `mxfp8` and `mxfp8-mxfp4-routed-expert`
override only the routed experts.

| Value | Training | Rollout |
|---|---|---|
| `bf16` | BF16 GEMMs. | BF16 weights. |
| `blockwise-fp8` (default) | Transformer Engine block-wise FP8 recipe (`--fp8-recipe blockwise`). | Block-wise FP8 weights, including the routed experts. |
| `mxfp8` | Transformer Engine MXFP8 recipe (`--fp8-recipe mxfp8`). | Block-wise FP8 weights, with the routed experts overridden to MXFP8. |
| `mxfp8-mxfp4-routed-expert` | `mxfp8`, with routed-expert weights trained on the MXFP4 grid (see below). | Block-wise FP8 weights, with the routed experts overridden to MXFP4 weights and MXFP8 activations, as in the official checkpoint. |

`mxfp8` and `mxfp8-mxfp4-routed-expert` require Blackwell, and
`mxfp8-mxfp4-routed-expert` requires an MXFP4 release (`DeepSeek-V4-Flash` or
`DeepSeek-V4-Flash-0731`). BF16 rollout loads a BF16 checkpoint and requires
`--rollout-weight-source checkpoint`; no checkpoint has the `mxfp8` rollout
layout, so it requires `--rollout-weight-source trainer`. For example, `--train-precision mxfp8
--rollout-precision mxfp8-mxfp4-routed-expert` serves the official MXFP4 experts
while training the routed experts in MXFP8, so their numerics differ between
training and rollout.

The training option selects the GEMM recipe, not the optimizer master-weight dtype.
Model parameters remain BF16, except parameters explicitly kept in FP32, and the
FP8 recipes leave the DSA indexer's weights projection in BF16.

`mxfp8-mxfp4-routed-expert` follows the routed-expert setting of the official
release and the [DeepSeek-V4 technical report](https://arxiv.org/abs/2606.19348):
the release stores routed experts in MXFP4, and the report applies MXFP4
quantization-aware training to the MoE expert weights during post-training,
with a straight-through estimator, and samples RL rollouts from the native FP4
weights.

Training has no MXFP4 GEMM. With `--train-precision mxfp8-mxfp4-routed-expert`,
routed-expert weights are rounded to the MXFP4 grid, with the block layout and
rounding of the rollout encoder, before their MXFP8 GEMMs, and the weight gradient
passes straight through. The forward pass of the routed experts thus matches their
MXFP4 × MXFP8 rollout, without saving memory or time. Master weights and optimizer
state keep their precision; attention, shared experts and activations are not
rounded to MXFP4. This value requires `--init-model-source hf` and the same
`--rollout-precision`. The report computes on the rounded weights in block-wise
FP8; this value uses MXFP8, whose 1×32 activation scaling matches the MXFP4 ×
MXFP8 rollout kernel. [Low Precision RL](/advanced/low-precision#5-mxfp4-routed-experts-blackwell)
shows how the formats line up between rollout and training.

The launcher picks SGLang's GEMM and MoE backends from `--rollout-precision`.
`--sglang-fp8-gemm-backend` and `--sglang-moe-runner-backend` override that choice;
`auto` lets SGLang choose.

## 4. Script breakdown

In this section, we explain what `full-train` does under the hood, and how to drive each stage manually if you need to debug or run outside the one-line launcher.

### 4.1 Download model + datasets

```bash
# inside the radixark/miles:latest container
hf download sgl-project/DeepSeek-V4-Flash-FP8 --local-dir /root/models/DeepSeek-V4-Flash-FP8
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/datasets/aime-2024
```

The Python launcher's `prepare-download` subcommand does the dataset fetch automatically; pass `--hf-checkpoint <path>` to skip the model download when the FP8 weights are already on a shared filesystem.

### 4.2 HF → Megatron `torch_dist` conversion

The direct-HF path ([§3.4](#34-initialization-and-rollout-weights)) skips this step. Otherwise, the conversion happens in two stages — a single-rank FP8 → BF16 cast, followed by a distributed `torch_dist` shard:

```bash
cd /root/miles
python tools/fp8_cast_bf16.py \
   --input-fp8-hf-path /root/models/DeepSeek-V4-Flash-FP8 \
   --output-bf16-hf-path /root/models/DeepSeek-V4-Flash-FP8-bf16/

MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py deepseek-v4-flash)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
PYTHONPATH=/root/Megatron-LM torchrun \
   --nproc-per-node 4 --nnodes 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --tensor-model-parallel-size 1 \
   --pipeline-model-parallel-size 8 \
   --expert-model-parallel-size 4 \
   --decoder-first-pipeline-num-layers 7 \
   --decoder-last-pipeline-num-layers 6 \
   --hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8-bf16/ \
   --save /root/models/DeepSeek-V4-Flash-FP8_torch_dist/
```

The Python launcher's `prepare-spmd` subcommand drives the same conversion.

### 4.3 Multi-node fan-out

The Python launcher manages Ray internally — start each pod with the `radixark/miles:latest` image and a working shared filesystem mounted at the same path on every node, then on the head node:

```bash
ray start --head --num-gpus 8 --disable-usage-stats
# … then on each worker:
ray start --address=${HEAD_IP}:6379 --num-gpus 8 --disable-usage-stats
```

Alternatively, you can set `MILES_SCRIPT_EXTERNAL_RAY=1` and `RAY_ADDRESS=…` to point the launcher at an existing Ray cluster (for example, one that an orchestration layer has already brought up). When `RAY_ADDRESS` is unset, the launcher boots a local Ray head.

### 4.4 Notable quirks

- **`model_type` rewrite for pruned checkpoints.** The older layer-pruning flow wrote `"model_type": "deepseek_ref"` into the local `config.json`. The launcher rewrites it back to `deepseek_v4` in place (`scripts/run_deepseek_v4.py`), so HF's `AutoConfig.from_pretrained` resolves the checkpoint without a patched `transformers`.

## 5. Example Recipe Configuration

### 5.1 Megatron Parallelism

These are the validated layouts shipped with the launcher; All parallelisms are supported, you can supply any other TP / EP / PP / CP combination that fits your compute.

| Hardware | Nodes × GPUs | TP | PP | CP | EP | expert-TP | Pipeline layout |
|---|---|---|---|---|---|---|---|
| H200 | 8 × 8 = 64 | 8 | 8 | 1 | 8 | 1 | first 4 / last 3 layers |
| GB300 | 8 × 4 = 32 | 8 | 4 | 1 | 8 | 1 | first 11 / last 10 layers |
| GB300 | 8 × 4 = 32 | 2 | 8 | 2 | 4 | 1 | first 4 / last 3 layers |

The Nodes × GPUs column counts **actor (training) nodes** — in disaggregated mode (see [§3.3](#33-colocated-vs-disaggregated-rollout)) rollout nodes come on top of these.

### 5.2 Algorithm

Using GRPO as an example, you can configure the algorithm with the following flags:

```bash
--advantage-estimator grpo
--eps-clip 0.2
--eps-clip-high 0.28
--kl-loss-coef 0.00
--kl-loss-type low_var_kl
--entropy-coef 0.00
```

The flags `--moe-router-freeze-gate` and `--freeze-e-score-correction-bias` are required and asserted on the `mcore` side — bias-update during RL is forbidden.

The launcher's `--enable-mis` adds importance sampling to correct the train/rollout mismatch.

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-tp-size 8
   --sglang-dp-size 8
   --sglang-ep-size 8
   --sglang-enable-dp-attention
   --sglang-attention-backend compressed  # V4 sparse-MLA backend
   --sglang-page-size 256
   --sglang-max-running-requests 64
   --sglang-chunked-prefill-size 8192
   --sglang-mem-fraction-static 0.5  # leave headroom for Megatron during wake_up
   --use-rollout-routing-replay  # MoE routing replay (R3)
)
```

The launcher sets the required env vars for you: `SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1`, `SGLANG_DSV4_FP4_EXPERTS` (`1` for packed MXFP4 rollout experts, otherwise `0`), `SGLANG_HEALTH_CHECK_TIMEOUT=120`, `SGLANG_DG_CACHE_DIR_PER_PROCESS=1`, and `SGLANG_OPT_FP8_WO_A_GEMM=0`, plus `MILES_SGLANG_DUMMY_LOAD=1` for trainer-owned rollout. Because `--train-deterministic` defaults to on, a stock run also gets `--deterministic-mode` plus `NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; pass `--no-train-deterministic` to drop those four.

On the Megatron side, V4 needs `--qkv-format bshd` with CP-aware data slicing. The DSA indexer additionally supports replay via `--use-rollout-indexer-replay` (off by default).

### 5.4 Optimizer

```bash
--optimizer adam
--lr 1e-6 --lr-decay-style constant
--weight-decay 0.1
--adam-beta1 0.9 --adam-beta2 0.98
--accumulate-allreduce-grads-in-fp32
--attention-softmax-in-fp32
--clip-grad 1.0  # Megatron default; not overridden by the launcher
```

The `--low-memory-resume` flag (off by default) puts optimizer states on CPU during ckpt resume to avoid OOM on the very first iteration.

## 6. Pairs Well With

- [Low Precision RL](/advanced/low-precision)
- [Architecture Support](/advanced/architecture-support) — the V4 plugin lives under `miles_plugins/models/deepseek_v4/`.
