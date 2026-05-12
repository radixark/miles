---
title: Examples
description: Annotated end-to-end walkthroughs for the workflows people actually want to build.
---
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

<CardGroup cols={2}>

  <Card title="Fully Async Rollout" icon="bolt" href="/miles/docs/examples/fully-async">

    Continuous background generation with a queue between rollout and training.
    Up to 2× end-to-end speedup.

  </Card>

  <Card title="Search-R1 (Tool Use)" icon="magnifying-glass" href="/miles/docs/examples/search-r1">

    Multi-turn rollout where the model can issue `<search>...` actions, get
    observations from a retrieval server, and produce a final answer.

  </Card>

  <Card title="ReTool (Code Execution)" icon="screwdriver-wrench" href="/miles/docs/examples/retool">

    SFT + RL pipeline for tool-augmented reasoning. Sandboxed Python code execution
    interleaved with thinking.

  </Card>

  <Card title="Multi-Agent Co-Evolution" icon="users" href="/miles/docs/examples/multi-agent">

    Two specialized agents (e.g. doctor + patient) train together and improve
    each other.

  </Card>

  <Card title="Reproducibility Recipe" icon="rotate-left" href="/miles/docs/examples/reproducibility">

    Bit-stable training across reruns. Determinism flags, seeds, and what to
    watch.

  </Card>

  <Card title="SFT on OpenHermes" icon="book-open" href="/miles/docs/examples/openhermes-sft">

    Plain SFT (no RL) — sometimes you just need a quick fine-tune.

  </Card>

</CardGroup>

## Where to start

* **Never used Miles for anything beyond GRPO?** → [Fully Async Rollout](/miles/docs/examples/fully-async).
* **Want tool use / RAG?** → [Search-R1](/miles/docs/examples/search-r1), then [ReTool](/miles/docs/examples/retool).
* **VLM / multi-agent?** → [Multi-Agent Co-Evolution](/miles/docs/examples/multi-agent).
* **Replay an old result?** → [Reproducibility Recipe](/miles/docs/examples/reproducibility).
