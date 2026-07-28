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

The maintained recipe lives in
[`examples/swe-agent`](https://github.com/radixark/miles/tree/main/examples/swe-agent),
with synchronous and fully-async launchers. Follow the
[recipe README](https://github.com/radixark/miles/blob/main/examples/swe-agent/README.md)
for the architecture, Harbor server setup, task format, and launch scripts.
