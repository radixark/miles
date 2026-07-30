# Tinker SDK-compatible service

Miles can expose its Megatron multi-LoRA trainer and SGLang rollout engines
through the public [Tinker Python SDK](https://github.com/thinking-machines-lab/tinker).
Existing Tinker data-plane code can create LoRA training clients, run losses and
optimizer steps, checkpoint or restore state, and sample from immutable adapter
snapshots without importing Miles.

## Prerequisites

- Miles and its normal Megatron dependencies.
- Megatron-Bridge from the `bridge` branch.
- SGLang from the `sglang-miles` branch.
- The public Tinker SDK in the client environment (`pip install tinker`).
- Disaggregated trainer and rollout GPUs. Pipeline and context parallel sizes
  must both be 1, and the trainer must use the default packed THD layout.

The example below uses one trainer GPU and one rollout GPU:

```bash
ray start --head --node-ip-address 127.0.0.1 --num-gpus 2 --disable-usage-stats

TINKER_MODEL_PATH=/root/Qwen3-0.6B \
MEGATRON_LM_PATH=/root/Megatron-LM \
TINKER_API_KEY=tml-local \
bash examples/tinker/run-qwen3-0.6b.sh
```

The service listens on `http://127.0.0.1:8068` by default. It remains alive
until its Ray job is stopped. Set `TINKER_TRAIN_GPUS` and `TINKER_TP_SIZE` to
exercise a larger tensor- or data-parallel trainer.

## Verify with the official SDK

From the node running the service:

```bash
python examples/tinker/client_smoke.py \
  --base-url http://127.0.0.1:8068 \
  --api-key tml-local \
  --base-model Qwen/Qwen3-0.6B \
  --tokenizer /root/Qwen3-0.6B
```

The smoke test exercises:

1. server capabilities, sessions, and LoRA model creation;
2. every built-in loss, top-k cross-entropy, custom loss, forward-backward, and Adam;
3. checkpoints with retained gradients and optimizer state;
4. exact-state restore into a second training client;
5. weights-only restore with fresh optimizer state;
6. named and ephemeral sampler snapshots;
7. sampling, generation logprobs, prompt logprobs, and prompt top-k logprobs;
8. base-model sampling.

See [the Tinker API guide](../../docs/user-guide/tinker-api.md) for the
supported loss contracts, service flags, and current compatibility boundary.
