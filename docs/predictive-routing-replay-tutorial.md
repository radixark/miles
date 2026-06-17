# Predictive Routing Replay (PR²) — Configuration tutorial

> Companion to [`predictive-routing-replay.md`](predictive-routing-replay.md) (algorithm reference). Read that first if you want the math; read this if you want to know which flag to set.

PR² in Miles is **fully opt-in**. The default state is OFF, every Miles enhancement defaults to OFF, and you turn things on flag-by-flag. The flags fall into four groups:

| Group | Required? | When to turn on |
|---|---|---|
| Prerequisites | Yes | Always (PR² depends on routing replay infra) |
| Core | Yes | Whenever you want PR² |
| Sub-sampling | Optional | Large batches / long sequences (memory / collective pressure) |
| Miles stabilization | Optional | Empirical training stability on top of the paper algorithm |

End-to-end recipes live in:
* [`scripts/run-qwen3-30B-A3B-pr2-paper.sh`](../scripts/run-qwen3-30B-A3B-pr2-paper.sh) — paper-faithful PR² ([arXiv:2606.00395](https://arxiv.org/abs/2606.00395)), no Miles stabilization.
* [`scripts/run-qwen3-30B-A3B-pr2-miles.sh`](../scripts/run-qwen3-30B-A3B-pr2-miles.sh) — same algorithm plus the Miles stabilization layer.

---

## 1. Prerequisites

PR² is layered on top of Miles's routing-replay infrastructure:

```bash
--use-routing-replay              # required
--use-miles-router                # required (the patched TopKRouter)
```

If you see `--enable-predictive-routing-replay requires --use-routing-replay.`, this is the missing piece.

## 2. Core flags (paper-faithful PR²)

```bash
--enable-predictive-routing-replay              # master switch
--bias-predictor-loss-type kl-post              # default; paper Eq. 8 objective
--bias-predictor-lr-mult     1e3                # CLI default; see note below
--predictive-storage-dtype   fp32               # fp32 / bf16 / fp16 cache dtype
```

With these four flags (plus the prerequisites) the code path matches paper Eq. 4–8 exactly. See [`predictive-routing-replay.md`](predictive-routing-replay.md) §4.1 for the formula derivation.

**About `--bias-predictor-lr-mult` ($\alpha$)**: the paper uses different values for different models — Qwen3-30B off-2: $10^4$; Qwen3-30B off-{4,8} and OLMoE: $10^3$; Moonlight: $5\times10^1$–$10^2$. The CLI default $10^3$ matches the most common paper setting (Qwen3 off-4/off-8 + OLMoE). The two example scripts pick different values to reflect their training-step regimes — `run-qwen3-30B-A3B-pr2-paper.sh` uses $10^4$ (Qwen3 off-2 setting), `run-qwen3-30B-A3B-pr2-miles.sh` uses $10^2$ (empirically tuned).

**Loss-type choices**:
* `kl-post` (default): post-softmax KL against the current router — paper Eq. 8.
* `kl`: Miles experimental delta-distribution KL. Not in the paper.

## 3. Sub-sampling caps (memory)

PR² caches one `[tokens × hidden]` fp32 tensor per router per microbatch. On long sequences with many samples per prompt this can exceed CPU offload. The three caps below run in sequence: pick samples → truncate each → hard cap the packed microbatch.

```bash
--predictive-downsample-batch-size      2       # keep N sequences per microbatch
--predictive-downsample-max-len-limit   4096    # truncate each to ≤ this many tokens
--predictive-max-total-tokens           2048    # final hard cap on the packed microbatch
```

Leaving them unset (the default) keeps every token. See [`predictive-routing-replay.md`](predictive-routing-replay.md) §5.2 for how this relates to the paper's per-response cache length `Tc`.

## 4. Miles stabilization (optional)

Every flag in this section defaults to OFF. When all are off, the runtime collapses to the paper algorithm. See [`predictive-routing-replay.md`](predictive-routing-replay.md) §4.2 for formulas, §5.3 for the full flag list. The summary:

* **Depth-aware layer-scale gating** — shrink the predictor delta on deep layers.
  ```bash
  --predictive-layer-scale-schedule sqrt_decay
  --predictive-layer-scale-min      0.5
  ```
* **Magnitude / top-k margin clips** — bound the predictor's perturbation to the base logits, with optional cross-rollout annealing.
* **Flip-fallback** — revert tokens whose predicted top-k flip lands inside a fragile post-correction margin; their predictor loss is zeroed out.
* **Sample reweighting** — hidden-shift mask + boundary-margin weighting.

```bash
# Example: turn on layer-scale gating + boundary-margin weighting.
--predictive-layer-scale-schedule sqrt_decay
--predictive-layer-scale-min      0.5
--predictive-boundary-loss-max-weight 2.0
--predictive-boundary-loss-min-margin 1e-3
```

The Miles example launcher uses this combination; see [`scripts/run-qwen3-30B-A3B-pr2-miles.sh`](../scripts/run-qwen3-30B-A3B-pr2-miles.sh).

---

## 5. Quick reference recipes

### A. Minimal paper-faithful PR²

```bash
--use-routing-replay --use-miles-router \
--enable-predictive-routing-replay \
--bias-predictor-loss-type kl-post \
--bias-predictor-lr-mult   1e3 \
--predictive-storage-dtype fp32
```

### B. Paper-faithful + sub-sampling for large batches

```bash
# A + the following:
--predictive-downsample-batch-size     2 \
--predictive-downsample-max-len-limit  4096 \
--predictive-max-total-tokens          2048
```

### C. PR² + Miles stabilization layer

```bash
# B + the following:
--predictive-layer-scale-schedule sqrt_decay \
--predictive-layer-scale-min      0.5 \
--predictive-boundary-loss-max-weight 2.0 \
--predictive-boundary-loss-min-margin 1e-3
```

This is the recipe shipped in [`scripts/run-qwen3-30B-A3B-pr2-miles.sh`](../scripts/run-qwen3-30B-A3B-pr2-miles.sh).

---

## 6. Monitoring

Watch these wandb metrics during training:

| Metric | What it tells you |
|---|---|
| `train/predictive_loss` | Healthy at 1e-4 magnitude; spikes above 1 mean the predictor is diverging |
| `predictive_topk_accuracy_*` | Overlap between predicted top-k and current router's top-k; > 0.85 healthy |
| `predictive_stabilized_bias_to_logits_ratio_*` | Predicted delta vs base logit magnitude; aim for < 0.02 |
| `predictive_flip_fallback_fraction_*` | Fraction of tokens reverted by flip-fallback; 2–5 % is typical |

If `train/predictive_loss` stays above 1e-3 and does not come down, try enabling the Miles stabilization layer (recipe C) or reducing `--bias-predictor-lr-mult`.

---

## 7. Full flag list

`python3 train.py --help` renders every PR² flag under the `Predictive Routing Replay (PR²)` argparse group. Each flag's mathematical meaning is in [`predictive-routing-replay.md`](predictive-routing-replay.md) §5.
