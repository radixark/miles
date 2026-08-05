---
title: OpenEnv
description: Train on Hugging Face OpenEnv environments through the agent-function extension point.
---

[OpenEnv](https://github.com/huggingface/openenv) is Hugging Face's open
protocol for RL environments: an environment is an HTTP service exposing
`reset` / `step` (and optionally `evaluate`), so any environment speaking the
protocol can serve any trainer.

Miles integrates OpenEnv as an
[agent-function integration](/user-guide/environments): a Miles-side agent
function drives the agentic loop — `reset(task_id)`, repeated `step`s, then
scoring the episode with the task's own tests — against an unmodified OpenEnv
server, and the score becomes the sample's reward through a custom reward
hook.

## Try it

The maintained end-to-end recipe is **Terminal-Bench-2 GRPO** in
[`examples/experimental/openenv`](https://github.com/radixark/miles/tree/main/examples/experimental/openenv).
It runs against a shared Docker env server (full per-task image fidelity) or
per-episode [Daytona](https://www.daytona.io/) cloud sandboxes built from each
task's official image (no resident infrastructure). Follow the
[recipe README](https://github.com/radixark/miles/blob/main/examples/experimental/openenv/README.md)
for prompt-data preparation, env-server modes, launcher flags, and operational
notes.
