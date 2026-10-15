---
title: Muse Spark Open
description: Launch recipe for Muse Spark Open (816B omni MoE) — full-parameter GRPO on 64 H200 or 32 B300 GPUs, on math or on image-grounded math.
---
## 1. Model Introduction

[Muse Spark Open](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B) is an 816B-parameter mixture-of-experts model (42B active) with vision and audio towers, released in BF16. Miles implements its language model natively in Megatron and trains it with RL; the towers stay frozen.

**Key highlights:**

- **Hybrid attention**: 46 sliding-window layers (window 2048, RoPE) and 16 full-attention layers without positional encoding; 128 query heads, 16 KV heads, head dimension 64.
- **MoE in every layer**: 62 layers, each with 256 routed experts and top-8 routing, no shared experts.
- **Gated residuals**: a learned per-channel gate mixes the residual stream and each sublayer output, which is post-normalized.
- **Scaled embeddings and logits**: the embeddings are RMS-normalized and scaled, and the logits are scaled and soft-capped at 20.

## 2. Supported Variants

| `--model-name` | What it is | HF source |
|---|---|---|
| `Muse-Spark-Open-816B-A42B` | omni checkpoint; Miles trains the 62-layer language model | [meta-models/Muse-Spark-Open-816B-A42B](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B) |

## 3. Environment Setup

### 3.1 Container and storage

Run the `radixark/miles:muse-spark-open` image on 8 nodes with 8 H200 GPUs each, or on 4 nodes with 8 B300 GPUs each, connected by InfiniBand. Every node needs:

- shared storage at `/root/models` (`--model-dir`) and `/root/datasets` (`--data-dir`);
- shared storage at `/root/shared_data` (`--output-dir`) to save weights;
- node-local NVMe at `/scratch` (`--offload-train-disk-dir`) with about 1.2 TB free (2.5 TB on B300) for the streamed Adam state.

```bash
docker run -d --name miles --gpus all --network host --ipc host --shm-size 512g \
  --ulimit memlock=-1 --ulimit stack=67108864 --device /dev/infiniband \
  -v /path/to/shared/models:/root/models \
  -v /path/to/shared/datasets:/root/datasets \
  -v /path/to/shared/outputs:/root/shared_data \
  -v /path/to/local/nvme:/scratch \
  radixark/miles:muse-spark-open sleep infinity
```

### 3.2 Download + `torch_dist` conversion

`prepare` (run by the launcher before training) downloads the checkpoint to `<model-dir>/Muse-Spark-Open-816B-A42B` and converts it on all nodes into `<model-dir>/Muse-Spark-Open-816B-A42B_torch_dist`, skipping a finished conversion. The `torch_dist` release is about 1.6 TB and loads into any training layout. It also downloads `zhuzilin/dapo-math-17k`, or with `--vlm` prepares `hiyouga/geometry3k` with `scripts/muse_spark/prepare_geo3k.py`.

### 3.3 SGLang

The engine runs the omni checkpoint natively from the [`sglang-miles-muse-spark-open`](https://github.com/sgl-project/sglang/tree/sglang-miles-muse-spark-open) branch, which the image contains. Math reasoning is validated on H200 and B300, image-grounded math on H200 only; B200 is untested.

## 4. Launch

Start one Ray cluster across the nodes, then launch from node 0. On 8 nodes with 8 H200 GPUs each:

```bash
# node 0
ray start --head --port 6379 --num-gpus 8
# nodes 1-7
ray start --address <node-0-address>:6379 --num-gpus 8

# math reasoning on DAPO-Math-17k
cd /root/miles
MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_muse_spark_64xh200.py

# image-grounded math on Geometry3K
MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_muse_spark_64xh200.py \
  --vlm --use-dynamic-sampling-filter --reasoning-effort medium --eval-interval 5
```

On 4 nodes with 8 B300 GPUs each, run `scripts/run_muse_spark_32xb300.py`, which takes the same options:

```bash
MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_muse_spark_32xb300.py
```

Time per step:

| Mode | GPUs | Rollout | Training | Weight sync |
|---|---|---|---|---|
| Math reasoning | 64 H200 | 2–5.5 min | 3–5.5 min | about 50 s |
| Image-grounded math | 64 H200 | 8–14 min | about 5 min | about 50 s |
| Math reasoning | 32 B300 | 1.7–3.5 min | 8.5–12.5 min (18.5 min at step 0) | about 90 s |

`train/train_rollout_logprob_abs_diff` stays around 0.007 for math reasoning (0.004–0.007 on B300) and 0.013 for image-grounded math. Every engine holds bit-identical weights after every sync. The reward starts at about 0.65–0.75; for math reasoning it reached 0.8–0.95 within 10 steps in our runs.

## 5. Recipe Configuration

### 5.1 Parallelism

| Launcher | TP | PP | CP | EP | expert-TP | first/last stage layers |
|---|---|---|---|---|---|---|
| `run_muse_spark_64xh200.py` | 8 | 8 | 1 | 8 | 1 | 7 |
| `run_muse_spark_32xb300.py` | 8 | 4 | 1 | 8 | 1 | 15 |

`--sequence-parallel`, THD packing with `--use-dynamic-batch-size --max-tokens-per-gpu 4096` (8192 on B300), and full recomputation are on. Each launcher sets these as the defaults of `--tensor-model-parallel-size`, `--pipeline-model-parallel-size`, `--context-parallel-size`, `--expert-model-parallel-size` and `--decoder-first-last-pipeline-num-layers`, together with its node count, engine size and engine memory settings; the shared recipe is `scripts/muse_spark/common.py`. On H200, TP4/PP8/CP2 (7 edge-stage layers) and TP4/PP4/CP4/EP16 (15) also fit; the numbers above use the defaults.

Rollout runs on 4 colocated SGLang engines with a BF16 KV cache: on H200, 16 GPUs each (`--sglang-ep-size 16`) with up to 64 concurrent requests; on B300, one node each (`--sglang-ep-size 8`) with up to 128. The B300 launcher also runs the rollout executor, router and session servers on the Ray head node (`--pin-rollout-manager-to-head`), as the validated B300 run did.

### 5.2 Algorithm

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28`, no KL or entropy term, per-token loss, and Rollout Routing Replay (`--use-rollout-routing-replay`). Each step samples 32 prompts × 16 responses of up to 4096 tokens (8192-token context) at temperature 1 and performs one update. `--rm-type math` gives reward 1 when the final `\boxed{}` answer matches the label.

The chat template's reasoning effort defaults to `xhigh`; the image-grounded math runs used `medium` to reduce response length.

### 5.3 Optimizer

Adam at `--lr 1e-6`, constant, with weight decay 0.1 and no gradient clipping. The FP32 main weights and moments do not fit the GPUs, so they stream through node-local NVMe:

```bash
--stream-optimizer-state-to-disk
--stream-optimizer-state-moment-dtype fp32
--offload-train-target disk
--offload-train-disk-chunk-mb 256
```

`--optimizer muon` switches to Muon (`dist_muon`), which keeps its state and FP32 main weights in CPU memory in chunks (`--chunked-optimizer-state-offload --chunked-muon-master-weights`).

### 5.4 Image-grounded math (`--vlm`)

`--vlm` trains on Geometry3K with the omni model provider: the frozen NaViT vision tower runs on the first pipeline stage, outside the optimizer and checkpoints, and the weight sync re-sends the towers from the HF checkpoint after every update. `--eval-interval N` scores the Geometry3K test split (601 problems, 4 samples each) before training, every N steps and after the last step; it requires `--vlm`.

The image-grounded mode renders prompts with `miles/utils/chat_template_utils/templates/muse_spark_fixed.jinja`, which places the `<image>` placeholder the train processor expands; its text rendering matches the checkpoint's template, which the math mode uses.

### 5.5 Save and export

`--save-interval N` saves the weights, without optimizer state, every N steps and after the last step to `<output-dir>/<run-id>/checkpoints`. After training, the launcher converts the last save on CPU into a Hugging Face checkpoint at `<output-dir>/<run-id>/hf`, with the files and layout of the released checkpoint; each save and each conversion takes about 1.6 TB. Convert any other save with:

```bash
python tools/convert_muse_spark_torch_dist_to_hf.py \
  --input-dir <output-dir>/<run-id>/checkpoints/iter_0000009 \
  --hf-checkpoint /root/models/Muse-Spark-Open-816B-A42B \
  --output-dir <output-dir>/<run-id>/hf-step-10
```

Save directories count steps from 0, so the save after step 10 is `iter_0000009`.

### 5.6 Notable quirks

- **BF16 KV cache**: the launcher passes `--sglang-kv-cache-dtype bfloat16`; the FP8 KV cache SGLang would otherwise pick widens the gap between rollout and training log-probabilities.
- **Muon cannot save its optimizer state**: Megatron requires `--no-load-optim --no-save-optim` with `--chunked-muon-master-weights`.

## 6. Pairs Well With

- [Disk Offload](/advanced/disk-offload)
