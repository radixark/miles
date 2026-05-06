# P2P Weight Transfer Examples

Example scripts for running P2P (RDMA) and broadcast (NCCL) weight transfer between
Megatron training and sglang rollout engines.

See [docs/en/advanced/p2p-weight-transfer.md](../../docs/en/advanced/p2p-weight-transfer.md)
for full documentation, architecture details, and profiling results.

## Quick Start

All examples use a single `run.py` script with two subcommands (`prepare` and `run`) and
per-model thin bash wrappers:

```
examples/p2p_weight_transfer/
├── run.py                # Shared logic: prepare + run (model registry)
├── Qwen3-4B.sh           # 1 node  – bash Qwen3-4B.sh [MODE]
├── GLM-Z1-9B.sh          # 1 node  – bash GLM-Z1-9B.sh [MODE]
├── Moonlight-16B.sh      # 2 nodes – bash Moonlight-16B.sh [MODE] [NODE_RANK] [HEAD_IP]
├── GLM-4.7-Flash.sh      # 2 nodes
├── GLM-5.sh              # 2/12/32 nodes – bash GLM-5.sh <VARIANT> [MODE] [NODE_RANK] [HEAD_IP]
├── Qwen3-30B-A3B.sh      # 4 nodes
├── GLM-4.5-Air.sh        # 8 nodes
├── Qwen3-235B-A22B.sh    # 16 nodes
└── Kimi-K2.sh            # 64 nodes
```

## Usage

### Single-node models (Qwen3-4B, GLM-Z1-9B)

Single-node models auto-detect networking and don't require node rank or head IP:

```bash
bash examples/p2p_weight_transfer/Qwen3-4B.sh p2p
bash examples/p2p_weight_transfer/GLM-Z1-9B.sh broadcast
```

### Multi-node models

```bash
# GLM-4.7-Flash (2 nodes, disaggregated)
bash examples/p2p_weight_transfer/GLM-4.7-Flash.sh p2p 0 $HEAD_NODE_IP  # head
bash examples/p2p_weight_transfer/GLM-4.7-Flash.sh p2p 1 $HEAD_NODE_IP  # worker

# Qwen3-30B-A3B (4 nodes)
bash examples/p2p_weight_transfer/Qwen3-30B-A3B.sh p2p 0 $HEAD_NODE_IP  # head
bash examples/p2p_weight_transfer/Qwen3-30B-A3B.sh p2p 1 $HEAD_NODE_IP  # worker 1
# ... workers 2, 3
```

### GLM-5 variants (2/12/32 nodes)

The `GLM-5.sh` wrapper accepts a `VARIANT` argument:

```bash
# GLM-5_4layer (2 nodes)
bash examples/p2p_weight_transfer/GLM-5.sh GLM-5_4layer p2p 0 $HEAD_NODE_IP

# GLM-5 full (32 nodes)
bash examples/p2p_weight_transfer/GLM-5.sh GLM-5 p2p 0 $HEAD_NODE_IP
```

### Using run.py directly

```bash
# Prepare: download model, datasets, convert checkpoint
python examples/p2p_weight_transfer/run.py prepare GLM-4.7-Flash

# Run: launch training with P2P weight transfer
python examples/p2p_weight_transfer/run.py run GLM-4.7-Flash \
    --mode p2p --node-rank 0 --head-ip $HEAD_NODE_IP

# List available models
python examples/p2p_weight_transfer/run.py list
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `CKPT_SAVE_DIR` | `/root/multinode` | Override checkpoint save directory |
| `SKIP_VALIDATION` | `0` | Set to `1` to skip `--check-weight-update-equal` |
| `BUCKET_SIZE_GB` | `1` | Transfer bucket size in GB |
| `MILES_LOG_DIR` | — | Directory for training logs |

## Notes

\* **Kimi-K2**: The profiling RDMA time for Kimi-K2 includes ~884 ms of GPU-side `post_load_weights` requantization on the rollout engines, since this model requires weight requantization after RDMA transfer.
