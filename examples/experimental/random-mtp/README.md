# Random MTP (Multi-Token Prediction) Training

Train a randomly initialized MTP head alongside RL, even when the base model has no pretrained MTP weights.

## What this does

1. Takes a standard model (e.g. Qwen3-4B) that has **no** MTP layers
2. Adds MTP layers with **random initialization** at step 0
3. Trains the MTP head as an auxiliary loss during RL (via `--enable-mtp-training`)
4. Syncs MTP weights between Megatron (training) and SGLang (rollout) every update
5. Logs `spec_accept_rate` and `spec_accept_length` so you can track MTP quality over training

After enough RL steps, the MTP head learns to predict future tokens, and speculative decoding acceptance rate should climb from ~0% to meaningful levels.

## Usage

```bash
# Default: colocated, 8 GPUs, Qwen3-4B
python run_random_mtp.py

# Disaggregated mode (separate training and inference GPUs)
python run_random_mtp.py --disagg

# Quick smoke test
python run_random_mtp.py --mode debug_minimal

# Custom MTP config
python run_random_mtp.py --mtp-num-layers 1 --mtp-loss-scaling-factor 0.3
```

## Key flags

| Flag | Description |
|------|-------------|
| `--init-random-mtp` | Enables random MTP initialization (set automatically by this script) |
| `--mtp-num-layers` | Number of MTP layers to add (default: 1) |
| `--enable-mtp-training` | Allow MTP parameters to receive gradients |
| `--mtp-loss-scaling-factor` | Weight of MTP auxiliary loss (default: 0.2) |

## How it works

- **Megatron side**: `--mtp-num-layers` builds the MTP block regardless of HF config. `--init-random-mtp` tells the checkpoint loader to tolerate missing MTP keys.
- **SGLang side**: `num_nextn_predict_layers` is injected into the HF config override so SGLang sets up the MTP draft worker. `speculative_algorithm=NEXTN` is auto-configured.
- **Weight sync**: Both colocated (tensor) and disaggregated (distributed) paths now update the draft/MTP model alongside the target model.
- **Checkpoint**: MTP weights are saved to Megatron checkpoints. Subsequent loads work normally since MTP keys exist.

## What to watch

In W&B / logs, look for:
- `mtp_loss` — should decrease over training
- `spec_accept_rate` — should increase from ~0
- `spec_accept_length` — should increase from ~1
