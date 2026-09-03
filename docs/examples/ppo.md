---
title: "PPO Example"
description: "Actor-critic PPO with GAE advantages, where the critic shares the actor's train GPUs."
# Generated from examples/ppo/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
This example trains Qwen3-4B with **PPO** — the actor-critic algorithm, with a learned value
model and GAE advantages — on a single node with the Megatron backend.

## PPO vs. GRPO in one paragraph

To turn a reward into a learning signal you need a baseline: "was this response better or worse
than expected?" GRPO gets that baseline for free by sampling a *group* of responses per prompt and
comparing each against the group average. PPO instead trains a second network, the **critic**,
whose only job is to predict the expected reward of a partial response; the advantage is then how
much better the actual outcome was than the critic's prediction. The trade-off: PPO carries a
second model (more memory, more code paths), but its baseline is per-token rather than
per-group, and it does not need a large `--n-samples-per-prompt` to be well-behaved.

In miles the critic is **colocated on the actor's train GPUs**, so PPO needs no extra GPUs over
the GRPO equivalent. It pays for that in memory, which is why `--offload-train` is turned on for
you — see [Constraints](#constraints-worth-knowing-before-you-debug).

## Files

* `run_qwen3_4b_ppo.py`: single-node launch script for Qwen3-4B.

## Quick Start

```bash
cd miles
python examples/ppo/run_qwen3_4b_ppo.py
```

The script's `prepare` step downloads Qwen3-4B and the DAPO-Math-17k dataset and converts the
checkpoint to Megatron `torch_dist` format, so there is nothing to set up by hand. Conversion is
skipped on reruns.

## Turning PPO on

The only flag that selects the algorithm is:

```bash
--advantage-estimator ppo
```

Everything else is tuning. Passing it sets `use_critic`, which builds the critic and switches
advantage computation to GAE.

## Critic flags

| Flag | Default | Meaning |
|---|---|---|
| `--critic-lr` | falls back to `--lr` | Critic learning rate. Usually wants to be larger than the actor's — this example uses `1e-5` against an actor `1e-6`. |
| `--critic-load` | falls back to `--load` | Critic init checkpoint. |
| `--critic-save` | `--save` + `_critic` | Sibling directory, so the two models do not clobber each other's iteration tracker. |
| `--critic-lora-adapter-path` | none | LoRA critic resume: the critic's own `iter_*/adapter` directory under `--critic-save`, holding its adapters and value head. Never inherited from `--lora-adapter-path`, since the two models share parameter names. |
| `--critic-lr-warmup-iters` | `0` | Linear warmup for the critic only. |
| `--num-critic-only-steps` | `0` | Value-function warmup: the actor stays frozen for this many initial rollout steps while the critic learns. A critic that starts from noise otherwise injects noisy advantages into the very first actor updates. |
| `--critic-num-nodes`, `--critic-num-gpus-per-node` | inherited from the actor | Set automatically — see the colocation constraint below. |

## Constraints worth knowing before you debug

These are enforced at argument validation, so you get an error rather than a silent wrong result:

* **The critic is colocated with the actor, and inherits its parallelism.** The critic is placed
  on exactly the same GPUs as the actor — `--critic-num-nodes` and `--critic-num-gpus-per-node`
  are overwritten with the actor's values — and it currently reuses the actor's TP/PP/CP as well,
  so there is no way to give the critic its own parallelism. Two consequences: **`--offload-train`
  is forced on**, because both models resident on the same devices at once is usually too much
  (`--no-offload-train` is accepted but warns, and is meant for offload debugging only); and when
  you scale, you only ever change the actor's placement — the actor world size is
  `--actor-num-nodes` × `--actor-num-gpus-per-node`, and `TP × PP × CP` must divide it.
* **Megatron only.** PPO raises with any other train backend. Both `--megatron-to-hf-mode raw`
  and `bridge` are supported; in bridge mode the critic's value head is freshly initialized rather
  than loaded from the HF checkpoint, since it has no HF counterpart.
* **`--kl-coef` must be 0.** Reward-level KL is rejected because the critic trains *before* the
  actor and never sees ref log probs, so its value targets would silently exclude the KL penalty
  applied to the actor's rewards. Use loss-level `--use-kl-loss` / `--kl-loss-coef` instead.
* **Not compatible with `MILES_EXPERIMENTAL_FT_TRAINER=1`.** The v2 fault-tolerant train group
  cannot route critic values yet.

## Which numbers here are verified

The parallelism (`TP=1`, `PP=2`, `CP=2` over 4 GPUs), the GPU count, and the PPO flag set follow
`tests/e2e/megatron/test_qwen3_4B_ppo.py`, which runs in CI.

Three values are deliberately **not** the CI ones, because the CI test is a 3-step smoke test
rather than a training recipe:

* `--eps-clip 0.2` here vs. `4e-4` in CI. `4e-4` pins the actor almost in place, which is useful
  for a fast deterministic test and wrong for actual training. `0.2` is the standard PPO value.
* `--num-rollout 300` here vs. `3` in CI.
* `--rollout-num-gpus-per-engine 1` here vs. `2` in CI. Qwen3-4B fits comfortably on one GPU, so
  one engine per GPU avoids paying tensor-parallel communication for no capacity gain.

Treat the rest — learning rates, `--kl-loss-coef`, `--entropy-coef` — as starting points to tune,
not as tuned values.
