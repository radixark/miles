---
title: Reproducibility Recipe
description: Bit-stable training across reruns. Determinism flags, seeds, and what to watch.
---

# Reproducibility Recipe

**What you'll learn:** how to configure Miles + SGLang + Megatron for **bit-wise
reproducible** RL training. Same inputs → identical outputs across reruns, machines,
and time.

Reproducibility matters more than people admit: it's the only way to A/B test changes
in a system this complex. If you can't trust that "I changed flag X and reward went up",
you're flying blind.

## How it works

Bit-wise reproducibility requires **three** independent stacks to be deterministic:

1. **Inference (SGLang)** — every kernel must be deterministic.
2. **Training (Megatron-LM)** — same.
3. **Communication (NCCL)** — algorithm choice and CUBLAS workspace can be
   non-deterministic by default.

Miles ships verified configurations that switch all three into deterministic mode.

## Quick start

We use the smallest Miles model (Qwen2.5-0.5B) on GSM8K so the loop fits in 5 minutes
and you can reproduce the bit-stability check yourself.

### 1. Disable FA3

Flash-Attention 3 currently has non-deterministic backward kernels. Drop it:

```bash
pip uninstall flash_attn_3 -y
```

### 2. Set the deterministic flags

```bash
SGLANG_ARGS+=(
   --sglang-enable-deterministic-inference
   --sglang-attention-backend flashinfer
)

PERF_ARGS+=(
   --deterministic-mode
)
```

### 3. Set the env vars (Ray `env_vars`)

```python
"env_vars": {
   "NCCL_ALGO": "Ring",
   "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
   "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
}
```

| Variable | Why |
|---|---|
| `NCCL_ALGO=Ring` | Forces a deterministic collective algorithm |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | Disables non-deterministic Transformer-Engine kernels |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | cuBLAS's deterministic workspace allocation |

### 4. Download + convert + run

```bash
hf download --repo-type dataset openai/gsm8k         --local-dir /root/gsm8k
hf download Qwen/Qwen2.5-0.5B-Instruct               --local-dir /root/Qwen2.5-0.5B-Instruct

cd /root/miles
source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct \
   --save           /root/Qwen2.5-0.5B-Instruct_torch_dist/

bash examples/reproducibility/run-qwen2.5-0.5B-gsm8k.sh
```

### 5. Verify

Run twice, then:

```bash
md5sum /root/Qwen2.5-0.5B-Instruct_torch_dist_run1/iter_0000020/*.pt
md5sum /root/Qwen2.5-0.5B-Instruct_torch_dist_run2/iter_0000020/*.pt
```

The hashes should match exactly. If they don't, something escaped the deterministic
configuration — see troubleshooting below.

## What's deterministic and what isn't

| Component | Default | Deterministic mode |
|---|---|---|
| Megatron forward | non-det | ✅ via `--deterministic-mode` |
| Megatron backward | non-det | ✅ |
| SGLang kernels | non-det | ✅ via `--sglang-enable-deterministic-inference` |
| Flash-Attn 3 | non-det | ❌ — uninstall |
| NCCL collectives | non-det | ✅ via `NCCL_ALGO=Ring` |
| cuBLAS GEMM | non-det | ✅ via `CUBLAS_WORKSPACE_CONFIG` |
| TE fused kernels | non-det | ✅ via `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` |
| Python dataloader shuffle | seeded | ✅ already |

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Hashes diverge after iter 1 | Flash-Attn 3 still installed |
| Hashes match for trainer but not SGLang | `--sglang-attention-backend flashinfer` not set |
| Hashes diverge across nodes | `NCCL_ALGO=Ring` not propagated to all workers |
| Hashes match locally but not on a different machine | cuDNN version mismatch |

## Cost of determinism

Roughly:

| Component | Throughput cost |
|---|---|
| Megatron deterministic mode | -3% to -8% |
| SGLang deterministic | -10% to -15% |
| NCCL Ring | -2% (vs. Tree) |
| Drop FA3 | -10% to -25% |

Total: ~25% slower for full bit-wise determinism. Worth it for debugging, science, and
A/B tests; usually disabled for production runs.

## When to disable determinism

* Production training runs where the cost is too high.
* When you've already nailed the result and want maximum throughput.
* On hardware that doesn't support deterministic kernels.

For most other cases — **keep it on while developing**. The hours you save when "I
swear that worked yesterday" stops being a phrase you say will pay back the throughput
many times over.

## References

* [SGLang deterministic inference blog](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)
* Megatron-LM deterministic-mode docs
