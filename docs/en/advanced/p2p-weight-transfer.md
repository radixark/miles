# P2P Weight Transfer

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
   - Query remote parallelism config and construct a local CPU model replica that mirrors the target's sharding layout, enabling correct weight format conversion before transfer.

2. **Weight gather**: Megatron TP/EP shards are all-gathered and converted to HF format, same as the broadcast path.

3. **P2P transfer**: Instead of a collective broadcast, each source rank writes bucketed weight tensors directly to the destination rollout rank's memory, in a write-only fashion.

4. **Synchronization**: Once all RDMA writes are confirmed complete, rollout engines increment their weight version and resume generation for the next training step.

## Supported Model Architectures

P2P weight transfer relies on a unified weight name mapping interface between Megatron and sglang (see [sglang#17326](https://github.com/sgl-project/sglang/pull/17326)). The following sglang model classes are supported:

| sglang Model Class | Model Family | Example Models |
|---|---|---|
| `Qwen2ForCausalLM` | Qwen2 (dense) | Qwen2.5-0.5B, Qwen2.5-7B |
| `Qwen3ForCausalLM` | Qwen3 (dense) | Qwen3-4B, Qwen3-8B |
| `Qwen3MoeForCausalLM` | Qwen3-MoE | Qwen3-30B-A3B, Qwen3-235B-A22B |
| `Glm4ForCausalLM` | GLM4 (dense) | GLM-Z1-9B-0414 |
| `Glm4MoeForCausalLM` | GLM4-MoE | GLM-4.5-Air |
| `Glm4MoeLiteForCausalLM` | GLM4-MoE | GLM-4.7-9B-Flash |
| `DeepseekV2ForCausalLM` | DeepSeek V2 | Moonlight-16B-A3B |
| `DeepseekV3ForCausalLM` | DeepSeek V3 | GLM-5 (744B-A40B) |

## Profiling Results

All profiling is run on H100-80GB clusters with `--check-weight-update-equal` validation enabled.
Timing measures `update_weights_implementation` (the actual weight transfer), averaged over
steady-state steps 3–14 (steps 1–2 are warmup).

### Small- to Medium-Scale (1–8 nodes)

| Model Family | Model Name | Total Param | sglang Model Class | Train Config | Inference Config | NCCL (ms) | RDMA (ms) | Delta |
|---|---|---|---|---|---|---|---|---|
| GLM4 | GLM-Z1-9B-0414 | 9B | `Glm4ForCausalLM` | TP=2, PP=1, CP=2, EP=1, ETP=1, 1 node | WS=8, EP=1, 1 node | 714.5 | 894.4 | +25.2% |
| Dpsk-V2 | Moonlight-16B-A3B | 16B(3B) | `DeepseekV2ForCausalLM` | TP=2, PP=1, CP=1, EP=8, ETP=1, 1 node | WS=8, EP=8, 1 node | 3,688 | 1,695.1 | **−54.0%** |
| GLM4-MoE | GLM-4.7-9B-Flash | 9B | `Glm4MoeLiteForCausalLM` | TP=4, PP=1, CP=1, EP=8, ETP=1, 1 node | WS=4, EP=4, 1 node | 2,525 | 4,218 | +67.0% |
| Dpsk-V3 | GLM-5_4layer | 4-layer | `DeepseekV3ForCausalLM` | TP=4, PP=1, CP=1, EP=8, ETP=1, 1 node | WS=8, EP=8, 1 node | 678 | 1,309 | +93.0% |
| Qwen3-MoE | Qwen3-30B-A3B | 30B(3B) | `Qwen3MoeForCausalLM` | TP=4, PP=1, CP=1, EP=8, ETP=1, 2 nodes | WS=8, EP=8, 2 nodes | 2,672 | 2,161 | **−19.1%** |
| GLM4-MoE | GLM-4.5-Air | 106B(12B) | `Glm4MoeForCausalLM` | TP=1, PP=4, CP=1, EP=8, ETP=1, 4 nodes | WS=8, EP=8, 4 nodes | 6,433.3 | 2,637.2 | **−59.0%** |
| Qwen3-MoE | Qwen3-235B-A22B | 235B(22B) | `Qwen3MoeForCausalLM` | TP=4, PP=4, CP=2, EP=16, ETP=1, 8 nodes | WS=32, EP=32, 8 nodes | 10,759 | 3,196 | **−70.3%** |

### Large-Scale (16–32 nodes)

| Model Family | Model Name | Total Param | sglang Model Class | Train Config | Inference Config | Nodes | NCCL (s) | RDMA (s) | Post-process (s) | Speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| Dpsk-V3 | GLM-5 | 744B(40B) | `DeepseekV3ForCausalLM` | TP=4, PP=4, CP=2, EP=16, ETP=1 | WS=64, EP=64 | 16+16 | 55.44 | 8.32 | 0.04 | **6.7×** |
| Dpsk-V3 | Kimi K2 | 1T(64B) | `DeepseekV3ForCausalLM` | TP=8, PP=8, CP=4, EP=32, ETP=1 | WS=256, EP=256 | 32+32 | 52.98 | 6.29 | 0.88 | **8.4×** |

> **RDMA** = `update_weights_implementation` steady-state average.
> **Post-process** = `finalize_and_resume_engines` steady-state average. For P2P mode this
> includes `post_load_weights` on the rollout GPUs (e.g. FP8 requantization for Kimi K2).
> **Speedup** = NCCL / RDMA on `update_weights_implementation`.

### Key Takeaways

1. **P2P scales with cluster size.** The benefit grows as more nodes participate:
   broadcast scales poorly with node count because every rank must receive from rank 0,
   while P2P uses direct GPU-to-GPU transfers.

2. **6.7× speedup at 16+16 nodes (GLM-5 744B), 8.4× at 32+32 nodes (Kimi K2 1T).**
   P2P saves ~47s per step on GLM-5 and ~47s per step on Kimi K2, which compounds
   over thousands of RL iterations.

3. **Crossover around 2+2 nodes.** Qwen3-30B (2+2 nodes) shows a modest 1.24× P2P advantage.
   Below that, broadcast's simpler control path wins.

4. **Post-process overhead for FP8 models.** Kimi K2 P2P adds 0.88s for
   `finalize_and_resume_engines`, which runs `post_load_weights` on the rollout GPUs
   to requantize FP8 weights after RDMA transfer. This is still negligible compared to
   the ~47s saved on the transfer itself.

## Examples

### CI Test (single-node, Qwen3-4B)

The P2P weight transfer E2E test validates correctness on a single node using `Qwen3-4B`:

```python
# tests/e2e/megatron/test_qwen3_4B_p2p.py
#
# Train: 4 GPUs (TP=2, CP=2)
# Rollout: 4 GPUs (sglang, 2 engines × 2 GPUs each)
# Flags: --update-weight-transfer-mode p2p --check-weight-update-equal
```

### GLM-4.7-Flash (2 nodes, disaggregated)

Each profiling example follows the same pattern: a `prepare` script (download model, convert
checkpoint) and a `run` script (launch Ray cluster, submit training job).

```bash
# 1. Prepare (head node does full prepare; workers use --download-only)
bash examples/p2p_weight_transfer/prepare-glm4.7-flash.sh               # head
bash examples/p2p_weight_transfer/prepare-glm4.7-flash.sh --download-only  # worker

# 2. Launch on each node
bash examples/p2p_weight_transfer/run-glm4.7-flash-2node-profile.sh p2p 0 $HEAD_NODE_IP  # head
bash examples/p2p_weight_transfer/run-glm4.7-flash-2node-profile.sh p2p 1 $HEAD_NODE_IP  # worker
```

All other models (Qwen3-30B, Qwen3-235B, GLM-4.5-Air, GLM-5 variants, Kimi K2) follow the same pattern
with their respective `prepare-*.sh` and `run-*-profile.sh` scripts under
`examples/p2p_weight_transfer/`.
