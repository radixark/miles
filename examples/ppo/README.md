# PPO Example

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

In miles the critic **shares the actor's train GPUs**, so PPO needs no extra GPUs over the GRPO
equivalent. It pays for that in memory, which is why `--offload-train` is turned on for you.

## Files

* `run-qwen3-4b-ppo.sh`: single-node launch script for Qwen3-4B.

## Prerequisite

Set up the model, dataset and environment following the Qwen3-4B example. This script expects
`/root/Qwen3-4B`, `/root/Qwen3-4B_torch_dist` and `/root/dapo-math-17k/dapo-math-17k.jsonl`.

## Quick Start

```bash
cd miles
bash examples/ppo/run-qwen3-4b-ppo.sh
```

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
| `--critic-lr-warmup-iters` | `0` | Linear warmup for the critic only. |
| `--num-critic-only-steps` | `0` | Value-function warmup: the actor stays frozen for this many initial rollout steps while the critic learns. A critic that starts from noise otherwise injects noisy advantages into the very first actor updates. |
| `--critic-num-nodes`, `--critic-num-gpus-per-node` | inherited from the actor | Set automatically; the critic shares the actor's placement. |

## Constraints worth knowing before you debug

These are enforced at argument validation, so you get an error rather than a silent wrong result:

* **Megatron only.** PPO raises with any other train backend, and is unsupported with
  `--megatron-to-hf-mode bridge`.
* **`--kl-coef` must be 0.** Reward-level KL is rejected because the critic trains *before* the
  actor and never sees ref log probs, so its value targets would silently exclude the KL penalty
  applied to the actor's rewards. Use loss-level `--use-kl-loss` / `--kl-loss-coef` instead.
* **Not compatible with `MILES_EXPERIMENTAL_FT_TRAINER=1`.** The v2 fault-tolerant train group
  cannot route critic values yet.
* **`--offload-train` is forced on.** Actor and critic share the train GPUs, so both resident at
  once is usually too much. `--no-offload-train` is accepted but warns, and is meant for offload
  debugging only.

## Which numbers here are verified

The parallelism (`TP=1`, `PP=2`, `CP=2` over 4 GPUs), the GPU count, and the PPO flag set follow
`tests/e2e/megatron/test_qwen3_4B_ppo.py`, which runs in CI.

Two values are deliberately **not** the CI ones, because the CI test is a 3-step smoke test rather
than a training recipe:

* `--eps-clip 0.2` here vs. `4e-4` in CI. `4e-4` pins the actor almost in place, which is useful
  for a fast deterministic test and wrong for actual training. `0.2` is the standard PPO value.
* `--num-rollout 300` here vs. `3` in CI.

Treat the rest — learning rates, `--kl-loss-coef`, `--entropy-coef` — as starting points to tune,
not as tuned values.

## Scaling up

The actor world size is `--actor-num-nodes` × `--actor-num-gpus-per-node`, and `TP × PP × CP` must
divide it. The critic inherits the same shape automatically, so when you change the actor's
placement you do not need to touch any critic placement flag.
