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

## Examples

All the following settings are tested under h100-80g clusters.

### Single node: Qwen3-4B
`tests/e2e/megatron/test_qwen3_4B_p2p.py`
### Multi node

Each multi-node example ships a **prepare** script (download model/datasets, convert checkpoint) and a **run** script (launch Ray jobs). All scripts accept three positional arguments:

| Argument | Description |
|---|---|
| `MODE` | `broadcast` or `p2p` |
| `NODE_RANK` | `0` (head node) or `1..N` (worker nodes) |
| `HEAD_NODE_IP` | IP address of the head node |

#### Qwen3-30B-A3B (4 nodes)

| Script | Description |
|---|---|
| `examples/p2p_weight_transfer/prepare-qwen3-30B-A3B.sh` | Prepare: download model/datasets, convert checkpoint |
| `examples/p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh` | Run: 4-node launch (`broadcast` or `p2p`) |

```bash
# 1. Prepare (single node)
bash examples/p2p_weight_transfer/prepare-qwen3-30B-A3B.sh

# 2. Launch on each node
bash examples/p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh p2p 0 $HEAD_NODE_IP  # head
bash examples/p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh p2p 1 $HEAD_NODE_IP  # worker
bash examples/p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh p2p 2 $HEAD_NODE_IP
bash examples/p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh p2p 3 $HEAD_NODE_IP
```

#### Qwen3-235B-A22B (16 nodes)

| Script | Description |
|---|---|
| `examples/p2p_weight_transfer/prepare-qwen3-235b-A22B.sh` | Prepare: download model/datasets, convert checkpoint |
| `examples/p2p_weight_transfer/run-qwen3-235B-A22B-16node-profile.sh` | Run: 16-node launch (`broadcast` or `p2p`) |

```bash
# 1. Prepare (single node)
bash examples/p2p_weight_transfer/prepare-qwen3-235b-A22B.sh

# 2. Launch on each node
bash examples/p2p_weight_transfer/run-qwen3-235B-A22B-16node-profile.sh p2p 0 $HEAD_NODE_IP   # head
bash examples/p2p_weight_transfer/run-qwen3-235B-A22B-16node-profile.sh p2p 1 $HEAD_NODE_IP   # worker
...
bash examples/p2p_weight_transfer/run-qwen3-235B-A22B-16node-profile.sh p2p 15 $HEAD_NODE_IP
```

