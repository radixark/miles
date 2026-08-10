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
It gives every episode its own cloud sandbox, built from that task's official
image so no resident infrastructure is left behind, on a choice of providers —
[AgentENV](https://github.com/kvcache-ai/AgentENV) (self-hosted, E2B-compatible),
[Daytona](https://www.daytona.io/), [E2B](https://e2b.dev/), or
[Modal](https://modal.com/). One shared Docker env server is supported as well,
for running without any sandbox platform.
Follow the
[recipe README](https://github.com/radixark/miles/blob/main/examples/experimental/openenv/README.md)
for prompt-data preparation, environment options, launcher flags, and
operational notes.
