---
title: DeepSeek V3
description: Launch recipe for DeepSeek-V3 (671 B total / 37 B active) via scripts/run_deepseek.py.
---
## 1. Model Introduction

[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) is a large-scale Mixture-of-Experts language model from DeepSeek.

**Key highlights:**

- **Fine-grained MoE architecture**: 671 B total / 37 B active per token, 256 routed experts with top-8 plus 1 shared expert, sigmoid router with bias.
- **MLA attention**: Multi-head Latent Attention with q-LoRA rank 1536, keeping the KV cache compact under long contexts.
- **Long-context capability**: trained at 32 K response length in this recipe; supports extended reasoning and agent-style workflows.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| DeepSeek-V3 | 37 B / 671 B | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) |

Passing a `--model-name` that contains `<N>layer` (for example `DeepSeek-V3-0324-5layer`) selects a
layer-pruned checkpoint from the `fzyzcjy` org and the matching `deepseek-v3-<N>layer` Megatron
definition — the single-node smoke-test path.

## 3. Launch

`scripts/run_deepseek.py` drives the whole pipeline: HF download, FP8 → BF16 cast, HF → Megatron
`torch_dist` conversion, an rsync of both directories to node-local storage, and the `train.py`
submission.

```bash
python scripts/run_deepseek.py train --num-nodes 16 --num-gpus-per-node 8
```

Single-node smoke test on a pruned checkpoint:

```bash
python scripts/run_deepseek.py train --model-name DeepSeek-V3-0324-5layer --num-nodes 1
```

Directories default to `--model-dir /root/models` (shared FS), `--data-dir /root/datasets`, and
`--model-local-dir /root/local_data` (node-local copy the training job reads from). `--task`
selects `dapo_aime` (default) or `gsm8k`; `--mode debug_minimal` shortens responses and drops
dynamic sampling and eval.

### Multi-node fan-out

The `torch_dist` conversion and the node-local rsync fan out across every node of the Ray
cluster, so a multi-node run needs the whole cluster joined **before** the launcher starts, and
the launcher must be told not to replace it. Bring up the head, join the workers, then run with
`MILES_SCRIPT_EXTERNAL_RAY=1`:

```bash
# on node 0
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

# on every other node
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 \
          --node-ip-address ${WORKER_IP} --disable-usage-stats

# back on node 0
MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_deepseek.py train \
   --num-nodes 16 --num-gpus-per-node 8
```

With an MPI-style hostfile (each line `ip slot=8`), fan the workers out from node 0:

```bash
for WORKER_IP in $(awk '{print $1}' $BASE_DIR/mpi_hostfile); do
  if [[ "$WORKER_IP" == "$MASTER_ADDR" ]]; then
    continue
  fi
  ssh root@"${WORKER_IP}" \
    "pkill -9 sglang ; ray stop --force ; pkill -9 miles ; \
     ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 \
               --node-ip-address ${WORKER_IP} --disable-usage-stats" &
done
wait
```

Without `MILES_SCRIPT_EXTERNAL_RAY=1` the launcher runs `ray stop --force` and starts its own
single-node head, which is what the single-node invocation above relies on — the conversion then
runs on that one node.

## 4. Checkpoint conversion

`train` performs the two conversion steps for you; the equivalent manual commands are below.

The HF checkpoint ships in block-quantized FP8 — first cast it to BF16:

```bash
python tools/fp8_cast_bf16.py \
   --input-fp8-hf-path  /root/models/DeepSeek-V3 \
   --output-bf16-hf-path /root/models/DeepSeek-V3-bf16/
```

Then convert BF16 HF → Megatron `torch_dist`. Run on **4 separate nodes** (`NODE_RANK=0..3`);
`MASTER_ADDR` is the IP of node 0:

```bash
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py deepseek-v3)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
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
   --hf-checkpoint /root/models/DeepSeek-V3-bf16/ \
   --save /root/models/DeepSeek-V3_torch_dist/
```

## 5. Recipe Configuration

All values below come straight from `scripts/run_deepseek.py`.

### 5.1 Parallelism

The layout is chosen from `--num-nodes`:

| Nodes | TP | PP | CP | EP | expert-TP | `decoder-last-pipeline-num-layers` |
|---|---|---|---|---|---|---|
| ≤ 2 | 1 | 1 | 4 | 4 | 1 | — |
| ≤ 4 | 4 | 1 | 4 | 4 | 1 | — |
| > 4 | 4 | 4 | 4 | 16 | 1 | 13 (full model only) |

Recomputation is `full` / `uniform` / 1 layer, with `--use-dynamic-batch-size` and
`--max-tokens-per-gpu 2048`.

<Warning>

`--max-tokens-per-gpu 2048` is a deliberately tiny placeholder in the current script (the
production value is 16384). Raise it before you read any throughput number off this recipe.

</Warning>

DeepSeek-V3 has 61 layers, which doesn't divide evenly into PP=4 —
`--decoder-last-pipeline-num-layers 13` puts the extra layers on the last stage. With
`--use-dynamic-batch-size`, miles packs samples up to `--max-tokens-per-gpu`; under CP=4, a CP
group shares a `CP × max-tokens-per-gpu` budget. miles always trains with data packing and
per-token loss, so dynamic batch size doesn't change the loss.

### 5.2 Algorithm

GRPO with DAPO-style dynamic sampling:

```
--advantage-estimator grpo
--kl-loss-coef 0.00
--kl-loss-type low_var_kl
--entropy-coef 0.00
--eps-clip 0.2
--eps-clip-high 0.28

--rm-type math
--num-rollout 3000
--rollout-batch-size 128
--n-samples-per-prompt 8
--rollout-max-response-len 32768
--rollout-temperature 1
--num-steps-per-rollout 4
--balance-data

--over-sampling-batch-size 256
--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

`--use-kl-loss` is absent, so the reference model is never loaded and the zero KL coefficient
costs nothing. `--over-sampling-batch-size 256` paired with `check_reward_nonzero_std` is the
DAPO-style setup: oversample, then drop prompts whose reward distribution has zero variance. The
two dynamic-sampling flags and the whole eval block are skipped under `--mode debug_minimal`.

### 5.3 Rollout & SGLang

The engine world size follows the node count: 4 GPUs per engine up to 4 nodes, 64 beyond that,
with attention DP 1 and 8 respectively.

```
--rollout-num-gpus-per-engine 64
--sglang-mem-fraction-static 0.7
--sglang-tp-size 64
--sglang-ep-size 64

# dp attention
--sglang-enable-dp-attention
--sglang-dp-size 8
--sglang-moe-dense-tp-size 1
--sglang-enable-dp-lm-head

# enable deepep for sglang
--sglang-moe-a2a-backend deepep
--sglang-deepep-mode low_latency

# make every dp rank has 128 concurrency
--sglang-server-concurrency 1024
--sglang-max-running-requests 2048
--sglang-chunked-prefill-size 16384
--sglang-cuda-graph-max-bs 256
```

`--rollout-num-gpus-per-engine` corresponds to SGLang's `tp_size`. To exploit large-EP inference,
the recipe sets EP64, DP-attention with DP8, and DeepEP `low_latency`.
`--sglang-server-concurrency` is a miles-specific knob to keep the SGLang HTTP server from being
swamped — default 512, raised to 1024 here so each of the 8 DP ranks gets 128 concurrent requests.
`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` is exported to match `--sglang-cuda-graph-max-bs`.

### 5.4 Optimizer

```
--optimizer adam
--lr 1e-6
--lr-decay-style constant
--weight-decay 0.1
--adam-beta1 0.9
--adam-beta2 0.98
```

CPU offload (`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer`) is available but commented out in the script. Enabling it puts
the Adam state on host RAM (~1.4–1.5 TB / 8-GPU node); if a node runs out of host memory, add
more nodes to widen parallelism rather than swapping.

### 5.5 Notable quirks

- **Online FP8 quantization against the HF config**: `--hf-checkpoint` points at the FP8 HF
  directory (also where the tokenizer is read from). miles applies the quantization config from
  the HF checkpoint, so weights are block-wise quantized before being passed to SGLang. The BF16
  directory produced by the cast is used only as the conversion input.
- **`--decoder-last-pipeline-num-layers 13`** is mandatory under PP=4 (61 layers don't divide
  evenly), and is skipped for layer-pruned variants.
- **Node-local reads**: training reads `--hf-checkpoint` and `--ref-load` out of
  `--model-local-dir`, which `train` populates by rsync from the shared `--model-dir`.
- **`--colocate`** runs actor and rollout on the same GPUs
  (`--actor-num-nodes` × `--actor-num-gpus-per-node`).
- **`--attention-backend flash` is deliberately unset** — MLA models need the default backend.

## 6. Pairs Well With

- [PD Disaggregation](/advanced/pd-disaggregation)
- [P2P Weight Transfer](/advanced/p2p-weight-transfer)
- [Fault Tolerance](/advanced/fault-tolerance)
