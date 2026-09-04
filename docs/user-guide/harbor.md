---
title: Harbor
description: Train agents on mixed task suites (SWE-bench, Terminal-Bench, custom) through the Harbor framework.
---

[Harbor](https://github.com/harbor-framework/harbor) is an agent-environment
framework from the Laude Institute: agent orchestration and grading are unified
in a single `Trial.run()` call, and a task is fully described by four files
(`instruction.md`, `Dockerfile`, `test.sh`, `task.toml`), so mixed task suites —
SWE-bench, Terminal-Bench, custom tasks — train through one endpoint.

Miles integrates Harbor as an
[agent-function integration](/user-guide/environments): the agent function
hands each session's OpenAI-compatible URL to Harbor, which runs the per-task
sandbox, runs the agent against that URL, and grades the result; the grade
becomes the sample's reward through a custom reward hook.

## Try it

Two execution modes; each example README is the complete guide for its mode:

- Tasks on a **local Docker daemon** → the agent-server mode:
  [`examples/swe-agent-harbor-docker`](https://github.com/radixark/miles/tree/main/examples/swe-agent-harbor-docker).
- Tasks on a **cloud sandbox backend** (E2B, Daytona, Modal, ...) → the
  in-process mode, no server in between:
  [`examples/experimental/harbor`](https://github.com/radixark/miles/tree/main/examples/experimental/harbor).
