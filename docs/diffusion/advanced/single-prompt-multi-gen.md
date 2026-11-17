---
title: Single-Prompt Multi-Generation
description: One rollout request, N outputs — engine-side conditioning expansion, microgroup mechanics, and seed layout.
---
GRPO needs `n_samples_per_prompt` outputs per prompt to normalize advantages within a group. The naive way is N separate
requests, which encodes the same prompt N times and denoises N batches of size 1. Miles-diffusion instead sends **one
request per microgroup** of up to M of those N samples: sglang-diffusion encodes the conditioning once, expands it
engine-side, and denoises the microgroup's outputs as a batch. Batched expansion is sglang-diffusion's default; Krea-2
and FLUX 3 Image Klein validate a microgroup of 1 instead.

## 1. Microgroups

`--rollout-microgroup-size M` splits each prompt group of `n_samples_per_prompt` samples into requests of at most M
(`generate_and_rm_group` in `miles/rollout/sglang_diffusion_rollout.py`). Each request carries
`num_outputs_per_prompt` = its microgroup's size: M, or less for the last microgroup when M does not divide
`n_samples_per_prompt`:

```
group (1 prompt × n_samples_per_prompt samples)
  ├─ microgroup 0  → POST /rollout/generate  num_outputs_per_prompt=M
  ├─ microgroup 1  → POST /rollout/generate  num_outputs_per_prompt=M
  └─ ...           (the last one may be smaller)
```

Microgroups of one group run as concurrent asyncio tasks, each routed to the least-loaded engine; in-flight requests
across all engines are capped at `--sglang-server-concurrency` × the number of engines.

Canonical values on `main`:

| Recipe | `n_samples_per_prompt` | microgroup size |
|---|---|---|
| SD3.5 GRPO + OCR | 16 | 8 |
| Qwen-Image GRPO + PickScore | 16 | 8 |
| Wan2.2 GRPO + PickScore (5-GPU) | 16 | 8 |
| Wan2.2 GRPO + PickScore (17-GPU) | 16 | 4 |
| LTX-2.3 GRPO + PickScore | 8 | 1 — multi-gen works but single-gen is faster |
| Cosmos3 GRPO | 16 | 1 — packed forward is single-sample |

LTX-2.3 supports engine-side expansion, but the recipe generates one output per request: a multi-output video response
is large enough that deserializing and scoring it stops overlapping with other in-flight requests, so single-output
requests pipeline better end to end.

## 2. Seed layout

Rollout stays deterministic and collision-free per sample:

- sgl-d expands one request's `seed` into `seed, seed+1, …, seed+M−1` — one RNG stream per output.
- The rollout (`generate_and_rm_group`) spaces request seeds so streams never overlap:

```python
seed_base = (rollout_seed + group_index * n_samples_per_prompt) % 2**31
microgroup_seed = seed_base + idx   # idx = index of the microgroup's first sample: 0, M, 2M, …
```

`group_index` is monotonic across the run, so every `(rollout, prompt-group, sample)` triple gets a distinct seed
(until `seed_base` wraps at 2^31). This
is what makes rollout replay and [deterministic mode](/diffusion/advanced/deterministic) possible at microgroup granularity.


## 3. Pairs well with

- [Streaming Reward and Deserialization](/diffusion/advanced/streaming-reward) — what happens to a microgroup response after
  generation.
- [Deterministic Training](/diffusion/advanced/deterministic) — seed layout is half of run reproducibility.
- [Core Concepts](/diffusion/user-guide/concepts) — where microgroup size sits among the batch knobs.
