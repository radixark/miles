# Single Node - Qwen3-4B P2P Weight Transfer

## Files

* `test_qwen3_4B_p2p.py`: single-node test with Qwen3-4B using P2P weight transfer.

## Quick Start

```bash
python examples/p2p_weight_transfer/test_qwen3_4B_p2p.py
```

This script will:
1. Download the Qwen3-4B model and datasets.
2. Convert the checkpoint to Megatron format.
3. Run a short training loop with `--update-weight-transfer-mode p2p`.
