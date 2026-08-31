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
hands each session's OpenAI-compatible URL to a Harbor server, which runs the
per-task container, installs and runs the agent against that URL, and grades
the result; the grade becomes the sample's reward through a custom reward
hook.

## Try it

Two execution modes:

- **Agent server** — Harbor runs on a separate host with a Docker daemon and
  the trainer calls it over HTTP. The maintained recipe is
  [`examples/swe-agent-harbor-docker`](https://github.com/radixark/miles/tree/main/examples/swe-agent-harbor-docker),
  with synchronous and fully-async launchers; its
  [README](https://github.com/radixark/miles/blob/main/examples/swe-agent-harbor-docker/README.md)
  covers the architecture, server setup, task format, and launch scripts.
- **In-process** — Harbor runs inside the rollout worker against a cloud
  sandbox backend (E2B / AgentENV, Daytona, Modal, ...), with no server in
  between: [`examples/experimental/harbor`](https://github.com/radixark/miles/tree/main/examples/experimental/harbor).
