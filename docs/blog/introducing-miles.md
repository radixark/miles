---
title: Introducing Miles
description: Why RadixArk built Miles, and what it means for production-scale RL post-training.
date: 2025-11-19
---

# Introducing Miles

*November 19, 2025 — RadixArk team*

Today RadixArk is open-sourcing **Miles**, a reinforcement learning framework purpose-
built for post-training large language models. Miles pairs high-throughput rollout
(SGLang) with scalable training (Megatron-LM and FSDP), and adds the precision,
stability, and observability features that production RL actually needs.

## Why another RL framework?

Reinforcement learning at the trillion-parameter scale runs into problems that a
research codebase never has to solve: the training and inference engines disagree on
expert routing; the numerical precision drifts silently between rollout and backprop;
long runs get killed by flaky NICs; adding a new model architecture means patching
Megatron.

Miles attacks each of these problems at the system layer so the RL researcher can focus
on the RL part.

## What ships on day one

- **Unified low-precision pipeline.** End-to-end FP8 sampling and training share a
  single quantization path. Rollout and trainer see bit-identical policies.
- **Rollout Routing Replay (R3).** For MoE models, expert routing captured at
  inference time is replayed during training, eliminating the mismatch that destabilizes
  large-scale MoE RL.
- **Speculative rollout with online MTP-SFT.** The draft model's acceptance rate stays
  high through training because its MTP layers are fine-tuned on-policy.
- **Algorithmic safety nets.** Truncated Importance Sampling (TIS) and Masked
  Importance Sampling (MIS) for off-policy correction when train ≠ inference.
- **Fault tolerance.** Rank-level recovery, step-level replay, RDMA P2P weight sync —
  weeks-long runs survive routine hardware faults.
- **First-class agentic rollout.** Tool use, search, code execution, and asynchronous
  multi-agent co-evolution all supported through clean Python extension points.
- **Customize without forking.** Twenty-plus plug-points let you replace the rollout,
  reward, loss, filter, or Megatron hook through CLI flags.

## Design principles

**Small core, many edges.** The trainer is a short Python program; almost every
behaviour is swappable through a `--*-path` flag rather than a code patch.

**Match the hardware.** Miles is designed around NVLink, InfiniBand, and RDMA — at
trillion-parameter scale, the interconnect is the rate limiter, so we optimize for it
first.

**Systems-first, algorithms-second.** We chase the instability that kills production
runs (routing mismatch, precision drift, NCCL hangs) before chasing the next algorithm.

## Try it

Head to the [Quick Start](../getting-started/quick-start.md) for a quick GRPO
run on Qwen3-4B with a single 8-GPU node.

## What's next

- FSDP-only trainer path for teams without Megatron expertise.
- Built-in Chrome-trace profiling and Grafana dashboards.
- Adaptive per-source data weighting driven by rolling reward signals.
- Deeper VLM multi-turn support.

The repo is on GitHub: [github.com/radixark/miles](https://github.com/radixark/miles).
Come say hi in the `#miles` channel of the [SGLang Slack](https://slack.sglang.ai).
