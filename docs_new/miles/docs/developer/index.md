---
title: Developer Guide
description: Architecture, contribution conventions, debugging, and migration notes.
---

# Developer Guide

You're here because you want to change Miles, not just use it. This section is the
short tour for new contributors.

<div class="grid cards" markdown>

-   :material-file-document-edit:{ .lg .middle } **[Contributing](contributing.md)**

    ---
    PR conventions, code layout, how reviews work.

-   :material-bug:{ .lg .middle } **[Debugging](debug.md)**

    ---
    Aligning precision, separate train/rollout debugging, common kernel pitfalls.

-   :material-source-branch-sync:{ .lg .middle } **[Migration Guide](migration.md)**

    ---
    Sync → async loop, breaking flag changes between releases.

-   :material-graph-outline:{ .lg .middle } **[Architecture Overview](architecture.md)**

    ---
    The 30-minute tour of how Miles is organized internally.

-   :material-flask-outline:{ .lg .middle } **[Experimental Features](experimental-features.md)**

    ---
    Opt-in backends and features (FSDP, …) that aren't production-ready yet.

</div>

## TL;DR for first-time contributors

1. Pick something small from `good first issue` on [GitHub](https://github.com/radixark/miles/issues).
2. Run the [Reproducibility recipe](../examples/reproducibility.md) so you can be sure
   "I changed X and it broke" actually means that.
3. Use `--debug-train-only` or `--debug-rollout-only` to scope your changes.
4. Open a PR. We'll review within ~48h.
