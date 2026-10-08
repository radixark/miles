---
title: P2P Weight Transfer
description: Direct rank-to-rank weight sync from actor to rollout via RDMA writes.
corresponding author: Jiadong Guo (JD-ETH)
---
miles supports P2P (point-to-point) weight transfer between training and rollout engines. By using `--update-weight-transfer-mode p2p`, miles enables more efficient weight transfer from training ranks to rollout engine ranks. More details on the design and implementation can be found in [this issue](https://github.com/radixark/miles/issues/755).

## Usage

To enable P2P weight transfer, add the following flag to your training command:

```
--update-weight-transfer-mode p2p
```

## How It Works

The default weight transfer mode in miles is `broadcast`: after training, updated weights are broadcast via NCCL to all rollout engine ranks. This works but does not fully utilize the available bandwidth, as redundant copies of the same weights are transferred to multiple ranks.

P2P mode addresses this by having each training rank transfer only the specific weight shards required by its target rollout engine rank(s), writing them directly to remote memory without redundant copies. The key steps are:

1. **Initialization**: Training ranks establish point-to-point connections (via RDMA) to their target rollout engine ranks. Including:
   - Create a transfer plan that maps each training rank to its target rollout rank(s) based on GPU counts and parallelism configuration.
   - Query remote rollout engines for their weight memory registration info (addresses and sizes for RDMA writes).
   - Query each target rank's parallelism config and server args, and build a model replica of its sharding layout without parameter storage, which converts HF weights into the exact bytes that rank's loader would write.

2. **Weight gather**: Megatron TP/EP shards are all-gathered and converted to HF format, same as the broadcast path.

3. **P2P transfer**: Instead of a collective broadcast, each source rank writes bucketed weight tensors directly to the destination rollout rank's memory, in a write-only fashion.

4. **Synchronization**: Once all RDMA writes are confirmed complete, rollout engines increment their weight version and resume generation for the next training step.

## Architecture

Both broadcast and P2P modes share the same bucketed weight-update pipeline in `miles/backends/megatron_utils/update_weight/`. The diagram below shows which components are shared and which are P2P-specific.

### Shared components (broadcast & P2P)

| Component | Description |
|---|---|
| **TP/EP all-gather** | Megatron TP shards are all-gathered within each PP stage; EP shards are gathered per-bucket when the accumulated expert data exceeds `buffer_size * ep_size`. Both modes perform this identically via `common.py`. |
| **Bucketed update** | Weights are not transferred one parameter at a time. Instead, converted tensors are accumulated into a fixed-size buffer (`--update-weight-buffer-size`, default 512 MB). When the buffer is full, the entire bucket is flushed — via NCCL broadcast or RDMA write depending on the mode. This amortizes per-transfer overhead. Non-expert and expert weights use separate buckets. |
| **PP independence** | Each pipeline-parallel stage updates its own weights independently. In broadcast mode, each PP rank has its own NCCL group (`miles-pp_{pp_rank}`). In P2P mode, each PP stage writes its own params to every engine rank (with Megatron Bridge, which gathers PP, each engine rank is written once). No cross-PP synchronization is needed during weight transfer, which is key to scaling. |
| **HF format conversion** | After all-gather, Megatron-format tensors (with custom naming and sharding) are converted to HuggingFace-format names expected by the sglang rollout engine. |

### P2P-specific components

| Component | File | Description |
|---|---|---|
| **Rollout engine rank assignment** | `protocols/utils/rollout_engine_rank_assignment.py` | Decides which rollout engine ranks each training rank sends weights to, for the rollout engines the rollout manager passes in (`engine_gpu_counts`). Senders are the training ranks that hold the same weights: those of one PP stage, or all ranks when PP is gathered (Megatron Bridge). Every (rollout engine, rollout engine rank) gets exactly one sender. First every sender gets one, so all of them send in parallel; the rest go to a sender that already sends the same rollout engine rank, so it prepares those weights once and sends them to several rollout engines. |
| **Model replica** | `protocols/utils/model_replica.py` | An sglang model of the target rank's layout, built on `meta` with no parameter storage and brought, one module at a time on the GPU, to the state the engine loads updates into: its startup postprocess, then the restore every update session begins with. Its `weight_loader` functions re-shard all-gathered HF weights into the exact bytes the target rank expects, loaded into a transfer buffer. `ModelReplicas` keeps one replica per shard layout (the rank's parallelism and the sglang arguments that change its bytes and may differ between server groups, such as quantization and MoE and GEMM backends) for the whole trainer process, so a reconnect rebuilds only the connections to the new engines. Rollout engines whose expert placement a replica cannot reproduce (redundant experts, a non-trivial `init_expert_location`, EPLB, elastic EP, DWDP, KTransformers) are rejected at connect. |
| **Transfer buffers** | `protocols/utils/transfer_buffers.py` | Two pinned host buffers per sender, sized to the larger of `--update-weight-buffer-size` and the largest complete parameter group, including alignment. Parameters sharing HF inputs stay in one group. Buffers are registered once per trainer process and reused only after every write reading them has returned. |
| **Pipelined transfer** | `p2p.py`, `protocols/transports/mooncake.py` | While the writes reading one transfer buffer are in flight, the sender loads the next group of parameters into the other. Each rollout engine has its own write thread, so a stuck engine delays only its own writes. The end of the update waits for every write and fails it if any write failed or is still running after `--p2p-transfer-timeout`. |

## Supported Model Architectures

P2P derives HF-to-parameter mappings by probing each replica's `load_weights` at the first update (`protocols/utils/loader_probe.py`). The probe uses real GPU inputs and meta parameters, then caches the mapping for later updates. Models need no separate mapping implementation.

P2P writes **parameters only**. The engine recomputes derived buffers, such as Gemma norm's `weight + 1`, after the update; `--check-weight-update-equal` checks those buffers too.

The following sglang model classes are tested:

| sglang Model Class | Model Family | Example Models |
|---|---|---|
| `Qwen2ForCausalLM` | Qwen2 (dense) | Qwen2.5-0.5B, Qwen2.5-7B |
| `Qwen3ForCausalLM` | Qwen3 (dense) | Qwen3-4B, Qwen3-8B |
| `Qwen3MoeForCausalLM` | Qwen3-MoE | Qwen3-30B-A3B, Qwen3-235B-A22B |
| `Glm4MoeForCausalLM` | GLM4-MoE | GLM-4.5-Air |
| `Glm4MoeLiteForCausalLM` | GLM4-MoE | GLM-4.7-9B-Flash |
| `DeepseekV3ForCausalLM` | DeepSeek V3p2 | GLM-5 (744B-A40B) |
| `DeepseekV3ForCausalLM` | DeepSeek V3 | Kimi-K2 (1T) \*, Kimi-K2.6 (1T) |

> **Note:** All the above models are tested on H100-80GB clusters.
>
> For Kimi-K2, we set `training nodes = inference nodes = 32` to ensure sufficient memory.
> The original Kimi-K2 checkpoint uses a block-quant size of `[128, 128]`, which will trigger errors if `sglang-tp-size = 32`.
> To work around this, we re-quantize to `[64, 64]` and update all affected scale tensors accordingly.

## Validated Models

All models below have been validated with `--check-weight-update-equal` in P2P mode.

| Model | sglang Model Class | Nodes |
|---|---|---|
| Qwen3-4B | `Qwen3ForCausalLM` | 1 |
| GLM-4.7-9B-Flash | `Glm4MoeLiteForCausalLM` | 2 |
| GLM-5_4layer | `DeepseekV3ForCausalLM` | 2 |
| Qwen3-30B-A3B | `Qwen3MoeForCausalLM` | 4 |
| GLM-4.5-Air | `Glm4MoeForCausalLM` | 8 |

> Enabling `--check-weight-update-equal` for Kimi-K2 is non-trivial due to several issues:
>
> - The user must first dequantize the Kimi-K2 model to BF16 for training, then re-quantize the parameters for weight updating. Meanwhile, the rollout side snapshots the original Kimi-K2 tensors as the reference when enabling `--check-weight-update-equal`. This means the rollout SGLang engine should load the checkpoints processed by the `dequant-requant` pipeline.
> - We use modified checkpoints with block-quant size `[64, 64]`, so any `quant/dequant` code that hard-codes the block-quant size as `[128, 128]` will break.
> - Certain tensors that are only initialized on the rollout side (e.g., `k_scale` / `v_scale`) must be skipped during the weight-check process.
>
> We verified P2P correctness for Kimi-K2 by enabling `--check-weight-update-equal` with hard-coded workarounds for the issues above. The transferred weights were confirmed correct. These hard-coded workarounds are verification-only and will not be merged into the main branch to keep the codebase maintainable.

## Profiling Results

For M source ranks and N target ranks, with source pp size `src_pp` and target ep size `sgl_ep`, the benefit of P2P transfer is approximately:
1. utilizes `M // src_pp` times more source transmission bandwidth.
2. each target rank receives `sgl_ep` times less data.

Thus we expect our solution to scale better, especially on the MoE models.

All profiling is run on H100-80GB clusters with 1GB transfer bucket.
Timing measures after `pause_generation` call returns and before `update_weight` call exits to exclude request queue abortion time.
Table includes steady-state steps 3–12 average. For Kimi-K2, the RDMA (ms) column includes GPU-side post-processing time (`post_load_weights` ~884ms) since this model requires GPU-side weight requantization after RDMA transfer.

Models marked with ★ are MoE architectures, where P2P benefits are most pronounced due to expert-parallel sharding reducing per-target transfer volume.

| Model Family | Model Name | Total Param | sglang Model Class | Train Config | Inference Config | NCCL (ms) | RDMA (ms) | Delta |
|---|---|---|---|---|---|---|---|---|
| GLM4-MoE ★ | GLM-4.7-9B-Flash | 30B(3B) | `Glm4MoeLiteForCausalLM` | TP=4, PP=1, CP=1, EP=8, ETP=1, 1 node | TP=4, EP=4, 1 node | 2,508.6 | 4,229.0 | +68.6% |
| DeepSeek-V3 ★ | GLM-5_4layer | 4-layer | `DeepseekV3ForCausalLM` | TP=4, PP=1, CP=1, EP=8, ETP=1, 1 node | TP=8, EP=8, 1 node | 732.2 | 1,260.8 | +72.2% |
| Qwen3-MoE ★ | Qwen3-30B-A3B | 30B(3B) | `Qwen3MoeForCausalLM` | TP=4, PP=1, CP=1, EP=8, ETP=1, 2 nodes | TP=8, EP=8, 2 nodes | 2,670.0 | 2,160.2 | **−19.1%** |
| GLM4-MoE ★ | GLM-4.5-Air | 106B(12B) | `Glm4MoeForCausalLM` | TP=1, PP=4, CP=1, EP=8, ETP=1, 4 nodes | TP=8, EP=8, 4 nodes | 5,001.1 | 2,637.2 | **−47.3%** |
| Qwen3-MoE ★ | Qwen3-235B-A22B | 235B(22B) | `Qwen3MoeForCausalLM` | TP=4, PP=4, CP=2, EP=16, ETP=1, 8 nodes | TP=32, EP=32, 8 nodes | 10,753.6 | 3,162.0 | **−70.6%** |
| DeepSeek-V3p2 ★ | GLM-5 | 744B(40B) | `DeepseekV3ForCausalLM` | TP=4, PP=8, CP=2, EP=16, ETP=1, 16 nodes | TP=64, EP=64, 16 nodes | 58,301.5 | 8,479.7 | **−85.5%** |
| DeepSeek-V3 ★ | Kimi-K2 | 1T(64B) | `DeepseekV3ForCausalLM` | TP=8, PP=8, CP=4, EP=32, ETP=1, 32 nodes | TP=32, EP=32, 32 nodes | 53,279.1 | 7,227.3 | **−86.4%** |


![P2P vs NCCL Broadcast Scaling](/assets/images/p2p_vs_nccl_scaling.png)

\* Kimi-K2 RDMA time includes ~884 ms GPU-side `post_load_weights` requantization on rollout engines.

## Examples

### CI Test (single-node, Qwen3-4B)

The P2P weight transfer E2E test validates correctness on a single node using `Qwen3-4B`:

```python
#
# Train: 4 GPUs (TP=2, CP=2)
# Rollout: 4 GPUs (sglang, 2 engines × 2 GPUs each)
# Flags: --update-weight-transfer-mode p2p --check-weight-update-equal
```
