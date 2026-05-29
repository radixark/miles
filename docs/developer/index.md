---
title: Developer Guide
description: Architecture, contribution conventions, debugging, and migration notes.
---

# Developer Guide

You're here because you want to change Miles, not just use it. This section is the
short tour for new contributors.

<CardGroup cols={2}>

  <Card title="Contributing" icon="file-pen" href="contributing">

    PR conventions, code layout, how reviews work.

  </Card>

  <Card title="Debugging" icon="bug" href="debug">

    Aligning precision, separate train/rollout debugging, common kernel pitfalls.

  </Card>

  <Card title="Migration Guide" icon="code-branch" href="migration">

    Sync → async loop, breaking flag changes between releases.

  </Card>

  <Card title="Architecture Overview" icon="diagram-project" href="architecture">

    The 30-minute tour of how Miles is organized internally.

  </Card>

  <Card title="Experimental Features" icon="flask" href="experimental-features">

    Opt-in backends and features (FSDP, …) that aren't production-ready yet.

  </Card>

</CardGroup>

## TL;DR for first-time contributors

1. Pick something small from `good first issue` on [GitHub](https://github.com/radixark/miles/issues).
2. Run the [Reproducibility recipe](../examples/reproducibility.md) so you can be sure
   "I changed X and it broke" actually means that.
3. Use `--debug-train-only` or `--debug-rollout-only` to scope your changes.
4. Open a PR. We'll review within ~48h.
