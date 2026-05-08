---
title: DeepSeek R1 / V3
description: Launch recipe for DeepSeek-R1 / DeepSeek-V3 (671 B total / 37 B active) on 16 nodes × 8 H100.
---

# DeepSeek R1 / V3

## 1. Model Introduction

[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) is a large-scale Mixture-of-Experts language model from DeepSeek, and [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) is the reasoning-tuned variant built on the same architecture. Both expose the same Megatron-side definition in miles and share the launch recipe on this page.

**Key highlights:**

- **Fine-grained MoE architecture**: 671 B total / 37 B active per token, 256 routed experts with top-8 plus 1 shared expert, sigmoid router with bias.
- **MLA attention**: Multi-head Latent Attention with q-LoRA rank 1536, keeping the KV cache compact under long contexts.
- **Long-context capability**: trained at 32 K response length in this recipe; supports extended reasoning and agent-style workflows.
- **Strong reasoning and coding**: R1 in particular targets mathematical reasoning and step-by-step inference; V3 is the general-purpose base.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| DeepSeek-V3 | 37 B / 671 B | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) |
| DeepSeek-R1 | 37 B / 671 B | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |

## 3. Environment Setup

### 3.1 Required env vars

```bash
export BASE_DIR=<shared FS path, reachable from every node>
export MASTER_ADDR=<head node IP>
```

### 3.2 Download model + datasets

```bash
hf download deepseek-ai/DeepSeek-R1 --local-dir $BASE_DIR/DeepSeek-R1
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir $BASE_DIR/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir $BASE_DIR/rl_data/aime-2024
```

### 3.3 HF → Megatron `torch_dist` conversion

The HF checkpoint ships in block-quantized FP8 — first cast it to BF16:

```bash
cd miles/
python tools/fp8_cast_bf16.py \
   --input-fp8-hf-path  $BASE_DIR/DeepSeek-R1 \
   --output-bf16-hf-path $BASE_DIR/DeepSeek-R1-bf16/
```

Then convert BF16 HF → Megatron `torch_dist`. Run on **4 separate nodes** (`NODE_RANK=0..3`); `MASTER_ADDR` is the IP of node 0:

```bash
cd miles/
source scripts/models/deepseek-v3.sh
PYTHONPATH=/root/Megatron-LM/ torchrun \
   --nproc-per-node 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --nnodes=4 --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --tensor-model-parallel-size 1 \
   --pipeline-model-parallel-size 8 \
   --expert-tensor-parallel-size 1 \
   --expert-model-parallel-size 4 \
   --decoder-first-pipeline-num-layers 7 \
   --decoder-last-pipeline-num-layers 6 \
   --hf-checkpoint $BASE_DIR/DeepSeek-R1-bf16/ \
   --save $BASE_DIR/DeepSeek-R1_torch_dist/
```

## 4. Launch

### 4.1 Quick start

On node 0:

```bash
cd miles/
bash scripts/run-deepseek-r1.sh
```

### 4.2 Multi-node fan-out

On every other node, join the Ray cluster:

```bash
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 \
          --node-ip-address ${WORKER_IP} --disable-usage-stats
```

Alternatively, with an MPI-style hostfile (each line `ip slot=8`), append a loop after `ray start --head` in `scripts/run-deepseek-r1.sh` to ssh out from node 0:

```bash
for WORKER_IP in $(awk '{print $1}' $BASE_DIR/mpi_hostfile); do
  if [[ "$WORKER_IP" == "$MASTER_ADDR" ]]; then
    continue
  fi
  ssh root@"${WORKER_IP}" \
    "pkill -9 sglang ; ray stop --force ; pkill -9 python ; \
     ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 \
               --node-ip-address ${WORKER_IP} --disable-usage-stats" &
done
wait
```

`scripts/run_deepseek.py` is an alternative Python entry point (in preview) that wraps download + FP8→BF16 cast + `torch_dist` conversion + `train.py` submission behind a Typer CLI.

## 5. Recipe Configuration

All values below come straight from `scripts/run-deepseek-r1.sh`.

### 5.1 Parallelism

| TP | PP | CP | EP | expert-TP | `decoder-last-pipeline-num-layers` | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|---|
| 8 | 4 | 4 | 32 | 1 | 13 | 16384 | 128 (16 × 8) |

```bash
PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 4
   --expert-model-parallel-size 32
   --expert-tensor-parallel-size 1
   --decoder-last-pipeline-num-layers 13

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)
```

DeepSeek-R1 has 61 layers, which doesn't divide evenly into PP=4 — `--decoder-last-pipeline-num-layers 13` puts the extra layers on the last stage. With `--use-dynamic-batch-size`, miles packs samples up to `--max-tokens-per-gpu`; under CP=4, a CP group shares a `CP × max-tokens-per-gpu` budget. miles always trains with data packing and per-token loss, so dynamic batch size doesn't change the loss.

### 5.2 Algorithm

GRPO with DAPO-style dynamic sampling:

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 128
   --n-samples-per-prompt 8
   --rollout-max-response-len 32768
   --rollout-temperature 1

   --over-sampling-batch-size 256
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --num-steps-per-rollout 4
   --balance-data
)
```

`--use-kl-loss` is enabled but the coefficient is 0 — to drop the reference model entirely, remove `--use-kl-loss`. `--over-sampling-batch-size 256` paired with `check_reward_nonzero_std` is the DAPO-style setup: oversample, then drop prompts whose reward distribution has zero variance.

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 64
   --sglang-mem-fraction-static 0.7
   --sglang-enable-ep-moe

   # dp attention
   --sglang-enable-dp-attention
   --sglang-dp-size 8
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head

   # enable deepep for sglang
   --sglang-enable-deepep-moe
   --sglang-deepep-mode auto

   # make every dp rank has 128 concurrency
   --sglang-server-concurrency 1024
)
```

`--rollout-num-gpus-per-engine 64` corresponds to SGLang's `tp_size`. To exploit large-EP inference, the recipe sets EP64, DP-attention with DP8, and DeepEP `auto`. `--sglang-server-concurrency` is a miles-specific knob to keep the SGLang HTTP server from being swamped — default 512, raised to 1024 here so each of the 8 DP ranks gets 128 concurrent requests.

Megatron side enables DeepEP as well:

```bash
MISC_ARGS=(
   --moe-enable-deepep
   --moe-token-dispatcher-type flex
   ...
)
```

### 5.4 Optimizer

```bash
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)
```

`--optimizer-cpu-offload` puts the Adam state on host RAM (~1.4–1.5 TB / 8-GPU node). If a node runs out of host memory, add more nodes to widen parallelism rather than swapping.

### 5.5 Notable quirks

- **Online FP8 quantization against the HF config**: `--hf-checkpoint` points at the FP8 HF directory (also where the tokenizer is read from). miles applies the quantization config from the HF checkpoint, so weights are block-wise quantized before being passed to SGLang. The BF16 HF directory is available as a commented alternative in `CKPT_ARGS`.
- **`--decoder-last-pipeline-num-layers 13`** is mandatory under PP=4 (61 layers don't divide evenly).
- **CKPT_ARGS** point at `$BASE_DIR/DeepSeek-R1_miles/` for both `--load` and `--save`; `--load` defaults to `--ref-load` when empty, so first run reads from `torch_dist`.
- **`--colocate`** runs actor and rollout on the same GPUs (16 nodes × 8 GPU = 128 GPUs total via `--actor-num-nodes 16 --actor-num-gpus-per-node 8`).

## 6. Pairs Well With

- [PD Disaggregation](../../advanced/pd-disaggregation.md)
- [P2P Weight Transfer](../../advanced/p2p-weight-transfer.md)
- [Fault Tolerance](../../advanced/fault-tolerance.md)
