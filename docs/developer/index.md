---
title: Developer Guide
sidebarTitle: Overview
description: Contribution conventions, internal architecture, dependency versions, and debugging.
---
You're here because you want to change Miles, not just use it. This section is the
short tour for new contributors.

<CardGroup cols={2}>

  <Card title="Contributing" icon="file-pen" href="/developer/contributor-guide">

    Repo layout, what enforces code style, what lives in .claude, and how to drive CI from a PR.

  </Card>

  <Card title="Architecture Overview" icon="diagram-project" href="/developer/architecture">

    The 30-minute tour of how Miles is organized internally.

  </Card>

  <Card title="Versions and Images" icon="layer-group" href="/developer/versions">

    How the miles, SGLang and Megatron-LM trees fit together, and how to bump one.

  </Card>

  <Card title="Debugging" icon="bug" href="/developer/debug">

    Isolating rollout from training, the debug and CI assertion flags, aligning precision.

  </Card>

  <Card title="Training Backends" icon="server" href="/user-guide/training-backend">

    Megatron-LM and FSDP: what each backend owns and where its code lives.

  </Card>

</CardGroup>

## TL;DR for first-time contributors

1. Pick something small from `good first issue` on [GitHub](https://github.com/radixark/miles/issues).
2. Run the [Reproducibility recipe](https://github.com/radixark/miles/tree/main/examples/experimental/reproducibility) so you can be sure
   "I changed X and it broke" actually means that.
3. Use `--debug-train-only` or `--debug-rollout-only` to scope your changes, and
   `--list-only` to confirm your test is actually registered in CI.
4. Open a PR. We'll review within ~48h.
