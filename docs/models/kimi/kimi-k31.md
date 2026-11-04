---
title: Kimi-K3.1
description: LoRA RL recipe for Kimi-K3.1 on 32 H200s, with the routed experts kept as the checkpoint's MXFP4 bytes in the trainer (W4A16) and one MXFP4 rollout engine with attention DP.
---

Kimi-K3.1 is [Kimi-K3](/models/kimi/kimi-k3) plus a DSA lightning indexer on each of its 24 MLA layers (top-2048 keys, 32 index heads of width 128, LayerNorm keys, no RoPE). The 93 layers, the 896 routed experts at top-16 and the KDA/MLA layout are K3's. This recipe trains a rank-32 LoRA with RL on dapo-math on **4 nodes × 8 H200**, half the GPUs of the K3 recipe. The indexer stays frozen.

## 1. What makes it fit

- **W4A16 trainer.** Held in BF16, the frozen base would need about 180 GB per GPU on 32 H200s. The trainer instead keeps every routed expert as the checkpoint's MXFP4 bytes (`weight_packed` plus `weight_scale`, 0.53 B per parameter, about 47 GB per GPU) and dequantizes them exactly into BF16 inside the grouped GEMM, so a layer's BF16 experts live only through its forward and recompute.
- **No converted checkpoint.** `--hf-checkpoint` and `--ref-load` both point at the native checkpoint, which the raw bridge mode loads directly; there is no BF16 copy and no `torch_dist` conversion.
- **Only the adapter moves.** The adapters are a BF16 bypass on both sides and are never merged. SGLang keeps a host copy of the base (`--lora-base-cpu-backup`, the launcher default), so a sync sends only the adapter: about 46 GiB at rank 32, 99% of it the per-expert factors.
- **One engine on all 32 GPUs.** The MoE runs EP32 on SGLang's Marlin MXFP4 runner, so each rank holds 28 whole experts with no tile padding, and keeps only its own experts' rows of each streamed adapter stack. Attention runs as 4 DP groups of TP 8, one per node.

## 2. Environment

- Megatron: `radixark/Megatron-LM` `miles-main` (validated at `eb547f33e`).
- SGLang: a build with Kimi-K3.1 and the rollout changes this recipe relies on: a BF16 KV cache for K3.1's DSA, MoE LoRA on the Marlin MXFP4 runner under expert parallelism with each rank keeping its own experts' rows of a streamed adapter, LoRA under attention DP with attention TP above 1, the DSA top-k broadcast (`SGLANG_DSA_TOPK_BROADCAST`) and the DSA prefill padding-row fix. These are not in public SGLang yet, so the recipe does not run on the published Miles image until they land.
- Checkpoint: `moonshotai/Kimi-K3.1` as published (compressed-tensors MXFP4, 1.5 TB), on a filesystem every node reads.
- Data: dapo-math-17k, which the recipe's `prepare-download` fetches with the checkpoint.

The adapter saves need a filesystem shared by all nodes as well: every training rank writes its own shard into the directory rank 0 creates.

## 3. Launch

The recipe lives in `scripts/run_kimi_k31_w4a16_lora_32xh200.py`, with the workload, loss and checks it shares with other hardware in `scripts/kimi_k31_w4a16_lora_common.py`. One container per node, a Ray cluster across them, `export MILES_SCRIPT_EXTERNAL_RAY=1`, then:

```bash
python scripts/run_kimi_k31_w4a16_lora_32xh200.py prepare-download --model-dir <models> --data-dir <datasets>
python scripts/run_kimi_k31_w4a16_lora_32xh200.py train \
  --hf-checkpoint <models>/Kimi-K3.1 --data-dir <datasets> --num-rollout 100 --rollout-max-response-len 3072 \
  --save-dir <shared-dir>
```

Every 5 rollouts the run saves the adapter shards and `adapter_config.json` under `<shared-dir>/<run-id>` (48 GB each), without the Adam state (about 294 GB at rank 32); `--skip-saving` turns that off. To continue from a save, add `--extra-args "--lora-adapter-path <dir> --no-load-optim"`, where the directory holds that save's `adapter_config.json` and `adapter_megatron_rank*.pt`; the shards are per training rank, so the layout must stay the same.

`scripts/run_kimi_k31_w4a16_lora_16xb300.py` carries the same recipe to 2 nodes of 8 B300 (training PP2, rollout TP16 EP16 with attention DP 2, and the base re-sent at every sync instead of kept on the host). Its workload is 16 prompts x 16 samples per step behind the dynamic-sampling filter: a group whose 16 rewards are all equal is dropped, and new prompts are drawn 4 groups at a time until 16 groups are kept. An earlier version (32 prompts per step, refilled 16 groups at a time) ran 15 steps on 16 B300 with the train/rollout log-prob gap (0.014 to 0.015) and KL (3.6e-4 to 3.8e-4) in the H200 range; this version has not run.

## 4. What the recipe sets

- Training: PP4 x TP4 x CP2 with EP8 (DP2), full recompute, 8192 tokens per GPU.
- Rollout: one engine, TP32 with attention DP 4 x TP 8, MoE EP32 on Marlin MXFP4, triton LoRA.
- Workload: 32 prompts x 16 samples per step, responses up to `--rollout-max-response-len` tokens (3072), 256 concurrent requests, one optimizer step per rollout; Adam at LR 1e-5 (the launcher's LoRA default).
- KV cache: BF16, sized per DP group from the response cap so the group's 64 requests are admitted at once: 64 x (3072 + 1024 prompt tokens) = 262144 at the default cap.
- Loss: score centering on the sampler's top-128 candidates, no std normalization, no KL or entropy term. Sampling is unfiltered (temperature 1, top-p 1), which Miles maps to the "selected" candidate mode, and the rollout log-probs replace the trainer's forward-only pass.
- Replay: routing replay on, indexer replay off.

The chat template uses the tokenizer's default (max) thinking effort.

## 5. What a healthy run looks like

From 44 steps of the validated run:

- 1422 to 2010 s per step (mean 1715): about 20 min of rollout, 9 min of training and 3 min of weight sync.
- Same-version train/rollout log-prob gap 0.014 to 0.018 and KL 4.4e-4 to 8.2e-4.
- `train/grad_norm` around 0.011, with isolated spikes (0.026, and 0.49 once).
- `rollout/raw_reward` from 0.61 (mean of steps 0 to 4) to 0.74 (steps 39 to 43), while the mean response shortens from 1890 to 1370 tokens and truncation falls from 36% to 18%. deepscaler scores a truncated response 0 in practice, so part of the reward gain is fewer truncations.
- GPU peak 130 GiB per card; host peak 1483 GB per node.

## 6. Notable quirks

- The recipe leaves `--check-lora-weight-equal` off: it hashes every adapter tensor on the CPU, on the trainer and again on every engine rank, about 167 s of each 187 s sync. Pass it through `--extra-args` for a checked run.
- The recipe sets `NCCL_RAS_ENABLE=0`: with NCCL 2.30.7, the RAS thread segfaulted on every node right after the engine first released its memory.
- With attention DP the launcher sets `SGLANG_DSA_TOPK_BROADCAST=1`, so every attention-TP rank of a DP group attends rank 0's DSA top-k; without it the ranks can pick different keys beyond the first 2048.
- Requests are spread over the DP groups by the engine (`--sglang-load-balance-method total_requests`), not a dp-aware router: `sglang-router` 0.3.2 reads the engine's `dp_size`, which attention DP reports as 1, and would send every request to DP group 0.
- Under attention DP, `--sglang-max-total-tokens` is per DP group, while running requests, mamba slots and chunked prefill are engine-wide and split over the groups.
- Indexer replay is off: shipping 196 KB of top-k per token cost about 30 minutes a step after the rollout.
- An InfiniBand port going down mid-run kills it with `IBV_WC_RETRY_EXC_ERR`; leave the port out with `--extra-env-vars "NCCL_IB_HCA=^<hca>"` before resuming.

## 7. Pairs Well With

- [Kimi-K3](/models/kimi/kimi-k3)
- [LoRA](/advanced/lora)
