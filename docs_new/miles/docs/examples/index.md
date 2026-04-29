---
title: Examples
description: Annotated end-to-end walkthroughs for the workflows people actually want to build.
---

# Examples

The model recipes show you how to train a model. The examples below show you how to
*build something useful* with Miles — tools, search, multi-agent, distillation, and
async rollout.

Each example follows the same template:

1. **What you'll learn** — the takeaway in one sentence.
2. **Prerequisites** — what you need installed/downloaded first.
3. **Files** — what's in the example directory.
4. **Quick start** — single command to run.
5. **Walkthrough** — annotated tour of the key code.
6. **What's happening underneath** — the moving parts you can't see.
7. **Tuning knobs** — the levers that matter.
8. **Troubleshooting** — the failure modes we've actually hit.
9. **Variations** — common adaptations.

## The catalogue

<div class="grid cards" markdown>

-   :material-flash-outline:{ .lg .middle } **[Fully Async Rollout](fully-async.md)**

    ---
    Continuous background generation with a queue between rollout and training.
    Up to 2× end-to-end speedup.

-   :material-magnify:{ .lg .middle } **[Search-R1 (Tool Use)](search-r1.md)**

    ---
    Multi-turn rollout where the model can issue `<search>...` actions, get
    observations from a retrieval server, and produce a final answer.

-   :material-tools:{ .lg .middle } **[ReTool (Code Execution)](retool.md)**

    ---
    SFT + RL pipeline for tool-augmented reasoning. Sandboxed Python code execution
    interleaved with thinking.

-   :material-account-group:{ .lg .middle } **[Multi-Agent Co-Evolution](multi-agent.md)**

    ---
    Two specialised agents (e.g. doctor + patient) train together and improve
    each other.

-   :material-replay:{ .lg .middle } **[Reproducibility Recipe](reproducibility.md)**

    ---
    Bit-stable training across reruns. Determinism flags, seeds, and what to
    watch.

-   :material-book-open-page-variant:{ .lg .middle } **[SFT on OpenHermes](openhermes-sft.md)**

    ---
    Plain SFT (no RL) — sometimes you just need a quick fine-tune.

</div>

## Where to start

* **Never used Miles for anything beyond GRPO?** → [Fully Async Rollout](fully-async.md).
* **Want tool use / RAG?** → [Search-R1](search-r1.md), then [ReTool](retool.md).
* **VLM / multi-agent?** → [Multi-Agent Co-Evolution](multi-agent.md).
* **Replay an old result?** → [Reproducibility Recipe](reproducibility.md).
