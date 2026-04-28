---
title: DeepSeek R1 / V3
description: Launch recipe for DeepSeek-R1 / DeepSeek-V3 (671 B total / 37 B active) on 16 nodes × 8 H100.
---

# DeepSeek R1 / V3

DeepSeek-V3 holds 671 B total / 37 B active. The canonical recipe is a 16-node × 8 H100 run with BF16 training, FP8 (128×128 block-wise) inference, DeepEP, and DAPO-style dynamic sampling. Max response length 32 K. CPU Adam is enabled to save GPU memory — each node holds ~1.4–1.5 TB of optimiser state in host RAM.

For SGLang: EP64 + DP-attention (DP8) + DeepEP. For Megatron: TP8, PP4, EP32, CP4.

## Variants

| Model | Active / Total | HF ID | Model config |
|---|---|---|---|
| DeepSeek-V3 | 37 B / 671 B | `deepseek-ai/DeepSeek-V3` | `scripts/models/deepseek-v3.sh` |
| DeepSeek-R1 | 37 B / 671 B | `deepseek-ai/DeepSeek-R1` | `scripts/models/deepseek-v3.sh` |

## Environment setup

Download the HF checkpoint to a directory accessible by all machines (referred to as `$BASE_DIR` from here on):

```bash
hf download deepseek-ai/DeepSeek-R1 --local-dir $BASE_DIR/DeepSeek-R1
```

The HF checkpoint ships in block-quantised FP8 — first cast it to a BF16 HF checkpoint:

```bash
cd miles/
python tools/fp8_cast_bf16.py \
   --input-fp8-hf-path  $BASE_DIR/DeepSeek-R1 \
   --output-bf16-hf-path $BASE_DIR/DeepSeek-R1-bf16/
```

Then convert the BF16 HF checkpoint to Megatron `torch_dist` format. Run the following on **4 separate nodes** (`NODE_RANK=0..3`); `MASTER_ADDR` is the IP of node 0:

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

After this step `$BASE_DIR/DeepSeek-R1_torch_dist/` is what `run-deepseek-r1.sh` will pass as `--ref-load`.

## Executing the training

On node 0:

```bash
cd miles/
bash scripts/run-deepseek-r1.sh
```

On every other node, join the Ray cluster:

```bash
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 \
          --node-ip-address ${WORKER_IP} --disable-usage-stats
```

Alternatively, if you have an MPI-style hostfile (each line is `ip slot=8`), you can append a loop after the `ray start --head` line in `scripts/run-deepseek-r1.sh` to ssh out and attach all workers from node 0:

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

## Parameter walkthrough

All values below are taken directly from `scripts/run-deepseek-r1.sh`.

### Model config

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/deepseek-v3.sh"
```

The launcher sources `scripts/models/deepseek-v3.sh`, which exposes `MODEL_ARGS` — the Megatron-side architectural definition of DeepSeek-V3 / R1 (61 layers, hidden 7168, 256 experts, top-k 8, MLA with q-LoRA rank 1536, sigmoid router with bias, MoE shared-expert intermediate 2048, etc.). Megatron can't read these from the HF checkpoint, so the model config is shipped in `scripts/models/`.

### CKPT_ARGS

```bash
CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/DeepSeek-R1/
   #--hf-checkpoint $BASE_DIR/DeepSeek-R1-bf16/
   --ref-load $BASE_DIR/DeepSeek-R1_torch_dist/
   --load $BASE_DIR/DeepSeek-R1_miles/
   --save $BASE_DIR/DeepSeek-R1_miles/
   --save-interval 20
)
```

`--hf-checkpoint` is the FP8 HF dir (also where the tokenizer is read from). `--ref-load` is the torch_dist directory you produced earlier. `--load` defaults to `--ref-load` when empty. miles performs online quantisation against the quantisation config in the HF checkpoint — with the FP8 checkpoint shown here, weights are block-wise quantised before being passed to SGLang.

### ROLLOUT_ARGS

```bash
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

`--over-sampling-batch-size 256` paired with the `check_reward_nonzero_std` filter is the DAPO-style dynamic-sampling setup: oversample, then drop prompts whose reward distribution has zero variance.

### EVAL_ARGS

```bash
EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime $BASE_DIR/rl_data/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 32768
   --eval-top-p 1
)
```

### PERF_ARGS

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

DeepSeek-R1 has 61 layers, which doesn't divide evenly into PP=4 — `--decoder-last-pipeline-num-layers 13` puts the extra layers on the last stage. With `--use-dynamic-batch-size`, miles packs samples up to `--max-tokens-per-gpu`; with CP=4, a CP group shares a `CP × max-tokens-per-gpu` budget. miles always trains with data packing and per-token loss, so dynamic batch size doesn't change the loss.

### GRPO_ARGS

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
```

`--use-kl-loss` is enabled but the coefficient is 0 — to drop the reference model entirely, remove `--use-kl-loss`.

### OPTIMIZER_ARGS

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

### SGLANG_ARGS

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

`--rollout-num-gpus-per-engine 64` corresponds to SGLang's `tp_size`. Other SGLang options are passed through with a `--sglang-` prefix. To exploit large-EP inference, the recipe sets EP64, DP-attention with DP8, and DeepEP `auto`. `--sglang-server-concurrency` is a miles-specific knob to keep the SGLang HTTP server from being swamped — default 512, raised to 1024 here so each of the 8 DP ranks gets 128 concurrent requests.

### MISC_ARGS

```bash
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   # use deepep for megatron
   --moe-enable-deepep
   --moe-token-dispatcher-type flex
)
```

DeepEP is enabled on the Megatron side here as well (`--moe-enable-deepep`, `--moe-token-dispatcher-type flex`).

### Job submit

```bash
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
        "MASTER_ADDR": "${MASTER_ADDR}",
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/sgl-workspace/nvshmem/install/lib/"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 16 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
```

`--actor-num-nodes 16 --actor-num-gpus-per-node 8` is what defines the 16-node × 8-GPU footprint. `--colocate` runs actor and rollout on the same GPUs.

## Python launcher

`scripts/run_deepseek.py` is an alternative entry point (in preview). It wraps download + FP8→BF16 cast + torch_dist conversion + `train.py` submission behind a Typer CLI, defaults to `deepseek-ai/DeepSeek-V3`, and supports the `*Nlayer` debug variants by routing them through `fzyzcjy/DeepSeek-V3-0324-{N}layer`.

## Pairs well with

- [PD Disaggregation](../../advanced/pd-disaggregation.md)
- [P2P Weight Transfer](../../advanced/p2p-weight-transfer.md)
- [Fault Tolerance](../../advanced/fault-tolerance.md)
