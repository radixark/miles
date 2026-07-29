---
title: Inkling
description: Launch recipe for Inkling (975 B), Thinking Machines' multimodal MoE with short convolution, relative attention, and a shared-expert sink.
---

The complete Inkling RL implementation is open at the Miles pull request: [`radixark/miles#1683`](https://github.com/radixark/miles/pull/1683).

## 1. Model Introduction

[Inkling](https://huggingface.co/thinkingmachines/Inkling) is a mixture-of-experts transformer released by Thinking Machines Lab, with 975 B total parameters and 41 B active, a context window of up to 1 M tokens, and pretraining on 45 trillion tokens of text, images, audio and video. Its architecture introduces short convolution, attention with relative positional embedding, and a novel MoE design with a shared-expert sink. Miles implements Inkling as a native Megatron model: local and global relative attention, the residual ShortConv, the shared-sink router and experts, and the image and audio encoders, and the same backend drives both full-parameter and LoRA RL.

**Key highlights**:

- **ShortConv**: a short causal convolution with a residual connection, applied on the K and V streams and on the attention and MLP/MoE outputs. Miles implements it with fused, packing-aware Triton kernels.
- **Relative attention**: a learned relative-position bias replaces positional embeddings; the stack mixes sliding-window and full-attention layers, with context length up to 1 M tokens.
- **Shared-expert sink MoE**: sigmoid top-k routing where the router also scores the shared experts and renormalizes their weights together with the selected routed experts'.
- **Train–inference consistency by construction**: customized relative-attention score-mod, ShortConv, and FP32 SwiGLU/combine kernels, while Rollout Routing Replay (R3) replays the rollout's routed expert IDs, including over media-expanded multimodal sequences.

## 2. Supported Variants

| Model | Active / Total | Layers | HF ID |
|---|---|---|---|
| Inkling | 41 B / 975 B | 66 | [thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling) |

## 3. Quick start

### 3.1 One-line launch

```bash
# Pull the image
docker pull radixark/miles:inkling

# Full-parameter GRPO on 16 nodes x 4 GB300, inside the container
cd /root/miles
python scripts/run_inkling_975b.py train \
   --model-name Inkling --train-mode full --task dapo_math \
   --num-nodes 16 --num-gpus-per-node 4

# LoRA GRPO (rank 32, all-linear), same cluster
python scripts/run_inkling_975b.py train \
   --model-name Inkling --train-mode lora --task dapo_math \
   --num-nodes 16 --num-gpus-per-node 4
```


### 3.2 Launcher path defaults

| Flag | Default | Use |
|---|---|---|
| `--model-dir` | `/root/models` | parent directory holding the HF checkpoint and Megatron `_torch_dist` artifacts |
| `--hf-checkpoint` | `{model-dir}/{model-name}` | released HF weights |
| `--torch-dist` | `{model-dir}/{model-name}_torch_dist` | converted Megatron torch-dist checkpoint |
| `--data-dir` | `/root/datasets` | HF datasets (dapo-math-17k, geo3k, …) |
| `--save-dir` | unset | training checkpoints |

Every option can also be preconfigured via `MILES_SCRIPT_<FIELD_NAME_UPPER>` env vars (precedence: CLI flag > env var > built-in default). Pass any extra Miles / Megatron / SGLang flags through `--extra-args`.

## 4. Script breakdown

This section explains what the launcher does under the hood, and how to drive each stage manually.

### 4.1 Download model + datasets

```bash
# inside the container
hf download thinkingmachines/Inkling --local-dir /root/models/Inkling
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset hiyouga/geometry3k --local-dir /root/datasets/geo3k
```

Pass `--hf-checkpoint <path>` to the launcher when the weights are already on a shared filesystem.

### 4.2 HF → Megatron `torch_dist` conversion

Inkling ships in BF16, so conversion is a single distributed `torch_dist` shard (no precision cast). The model definition comes from `scripts/models/inkling-975b.sh`:

```bash
cd /root/miles
source scripts/models/inkling-975b.sh
PYTHONPATH=/root/Megatron-LM torchrun \
   --nproc-per-node 4 --nnodes 4 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --tensor-model-parallel-size 4 \
   --pipeline-model-parallel-size 1 \
   --expert-model-parallel-size 4 \
   --hf-checkpoint /root/models/Inkling \
   --save /root/models/Inkling_torch_dist/
```

The saved `torch_dist` checkpoint is parallelism-agnostic: training can load it under any validated TP / PP / EP layout. Point the launcher at the result with `--torch-dist`.

### 4.3 Multi-node fan-out

Start each pod with the image and a shared filesystem mounted at the same path on every node, then bring up Ray:

```bash
# head node
ray start --head --num-gpus 4 --disable-usage-stats
# each worker
ray start --address=${HEAD_IP}:6379 --num-gpus 4 --disable-usage-stats
```

Set `MILES_SCRIPT_EXTERNAL_RAY=1` to point the launcher at this existing Ray cluster. When it is unset, the launcher boots a local Ray head itself.

## 5. Example Recipe Configuration

### 5.1 Megatron Parallelism

These are the validated layouts shipped with the launcher. Other TP / EP / PP / CP combinations that fit your compute can be supplied via `--extra-args` (the launcher raises on untested GPU counts to keep you honest).

| Hardware | GPUs | TP | SP | PP | EP | expert-TP | Notes |
|---|---|---|---|---|---|---|---|
| GB300 | 64 | 4 | on | 4 | 16 | 1 | `--decoder-last-pipeline-num-layers 15` |


### 5.2 Algorithm

GRPO with truncated importance sampling. The launcher defaults: global batch size 32, group size 8, maximum response length 4096 with truncation, Adam with `lr 1e-6`, constant decay. Rollout Routing Replay is on by default (`--use-rollout-routing-replay`). It replays only the routed expert IDs; the continuous routed *and* shared weights are recomputed from the current router under one common normalization, so gradients still flow through both.

### 5.3 Training attention backends

`--inkling-attn-backend` selects the training-side attention implementation:

| Backend | Role | 8K packed, rel-extent 1024, GB300 |
|---|---|---|
| `flex` (default) | block-sparse FlexAttention, differentiable fwd+bwd | ~34 ms fwd+bwd, 10.9 GB peak |
| `te` | Transformer Engine DPA reference | ~189 ms, 46 GB peak |
| `fa4` | serving-bit-identical FA4 forward + TE-recompute backward | ~195 ms, 46 GB peak |


### 5.4 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 16
   --sglang-attention-backend fa4
   --sglang-moe-runner-backend triton
   --sglang-mamba-scheduler-strategy extra_buffer
   --sglang-enable-multimodal
   --sglang-context-length 8192

   # full-parameter
   --sglang-mem-fraction-static 0.6
   --sglang-max-running-requests 64
   --sglang-max-total-tokens 327680

   # LoRA (replaces the three flags above)
   --sglang-ep-size 16
   --no-offload-rollout --no-offload-train
   --sglang-mem-fraction-static 0.65
   --sglang-max-running-requests 32
   --sglang-lora-backend triton
   --sglang-lora-use-virtual-experts
   --sglang-max-loras-per-batch 1
)
```

Weight updates stream Megatron shards into SGLang's tensor layout in bounded buckets over CUDA IPC (colocated). In LoRA mode only the adapter is synchronized.

### 5.5 Optimizer

```bash
--optimizer adam
--lr 1e-6 --lr-decay-style constant
--accumulate-allreduce-grads-in-fp32
--attention-softmax-in-fp32

# full-parameter only, set by the launcher
--optimizer-state-nvme-dir /tmp/opt_offload
--optimizer-state-nvme-chunk-mb 256
--offload-train-target disk
--offload-train-disk-dir /tmp/train_offload
--micro-batch-size 1
```

Within a single GB300 rack the 975 B policy, gradients, and FP32 optimizer state exceed GPU memory, so Miles streams Megatron `DistributedOptimizer` state between a bounded GPU working set and node-local NVMe. The offload changes storage placement, not the update math. The paused training actor's weights are additionally disk-backed through `torch_memory_saver`. LoRA training skips both offloads and uses dynamic batching (`--use-dynamic-batch-size --max-tokens-per-gpu 4096`) instead of the fixed micro-batch.

<Warning title="Inkling-specific caveats">

1. **Fixed micro-batches for full-parameter runs.** Dynamic token packing exposes a PP-p2p × EP-all-to-all NCCL launch-order race on varlen shapes; the launcher pins `--micro-batch-size 1` for full-parameter training. Keep it pinned until the pad-to-fixed fix lands.
2. **One LoRA adapter per engine.** RL serving uses `--sglang-max-loras-per-batch 1` with the triton LoRA backend and virtual experts: the engine holds exactly the current policy's adapter (see [LoRA RL](#6-lora-rl)).

</Warning>

## 6. LoRA RL

`--train-mode lora` switches the same backend and parallel stack to adapter-only training. The base model stays frozen in both runtimes, and GRPO updates only the low-rank factors:

$$
y = Wx + \frac{\alpha}{r}\,B(Ax).
$$

The adapter follows Inkling's released LoRA schema, covering the attention, dense MLP, MoE, and LM-head projections (defaults: rank 32, `alpha = 32`, all-linear). The routed experts use a shared-outer factorization: one factor is shared across experts while the expert-specific factors follow EP sharding.

After each optimizer step, Miles exports a serving-ready adapter directly from the distributed training state and hands it to the colocated SGLang worker over CUDA IPC; the frozen base is never re-transferred. This cuts the weight update from 49.4 s to 2.5 s (20×) and brings training-step time to ~85 % of full-parameter training, with no optimizer offload needed. Released Inkling adapters in safetensors format support warm starts via `--lora-adapter-path`, and per-rank adapter checkpoints allow exact training resume.

## 7. Multimodal RL

`--task geo3k` switches to the vision-language recipe: structured multimodal rollouts, Megatron-side vision and audio towers, and routing replay preserved across the media-expanded sequence. Each image or audio sentinel in the rendered prompt expands into its patch/frame positions before packing, and the R3 trace is indexed over the expanded sequence, so replay stays aligned even though SGLang made its routing decisions after expansion. Production recipes keep the vision and audio towers frozen and train the language model (or its adapter) around their embeddings; training the towers is available as an experimental option. The same path supports audio inputs and both full-parameter and LoRA training.
