# Bellman Policy Optimization (BPO)

Custom loss plugin implementing the sampled-token Bellman Policy Optimization objective from [Bellman Policy Optimization](https://arxiv.org/abs/2609.15987) (Eqs. 14–15) via `--loss-type custom_loss`.

## Usage

Add these flags to a GRPO training command:

```bash
--loss-type custom_loss \
--custom-loss-function-path examples.experimental.bpo.bpo_loss.bpo_loss_function \
--advantage-estimator grpo \
--eps-clip 0.2 \
--eps-clip-high 0.28 \
--bpo-smoothing 0.1 \
--bpo-weight-cap 3.0
```

`--bpo-smoothing` and `--bpo-weight-cap` are registered automatically from `BpoConfig` when `--custom-loss-function-path` points to `bpo_loss_function`. Core miles defaults `--eps-clip-high` to `--eps-clip` (`0.2`), so pass `--eps-clip-high 0.28` explicitly to match the paper's ClipHigher setting.

Pass `--use-rollout-logprobs` to use rollout-engine probabilities as $\pi_{\text{old}}$; omit it to use trainer-recomputed `log_probs`.

## Objective

For each active response token with current policy probability $p = \pi_\theta(y_t \mid x, y_{<t})$ and old policy probability $q = \pi_{\text{old}}(y_t \mid x, y_{<t})$, BPO replaces PPO's clipped importance ratio with a stop-gradient complementary-probability weight:

$$
w_t = \frac{1 + \varepsilon - q}{1 + \varepsilon - p}, \qquad
M_t = 1 - \mathbb{I}\!\left[(A_t > 0 \land w_t > 1 + \epsilon_{\text{high}}) \lor (A_t < 0 \land w_t < 1 - \epsilon_{\text{low}})\right]
$$

$$
\ell_t = -\operatorname{sg}\!\left(M_t \cdot \min(w_t, C) \cdot A_t\right) \log \pi_\theta(y_t \mid x, y_{<t})
$$

The rejection mask $M_t$ is evaluated on the uncapped weight $w_t$ before clamping to $C$. All surrounding `policy_loss_function` features — `ResponseLayout`, entropy regularization, reference KL loss, OPSM, TIS/MIS, custom PG reducers, and training diagnostics — remain available unchanged.

## Parameters

| Flag | Default | Meaning |
| --- | --- | --- |
| `--bpo-smoothing` | `0.1` | Additive smoothing $\varepsilon > 0$ for complementary probabilities. |
| `--bpo-weight-cap` | `3.0` | Upper cap $C > 0$ on the detached weight $w_t$. |
| `--eps-clip` | `0.2` | Lower mask threshold $\epsilon_{\text{low}} \in [0, 1)$ when $A_t < 0$. |
| `--eps-clip-high` | `--eps-clip` | Upper mask threshold $\epsilon_{\text{high}} \ge 0$ when $A_t > 0$ (pass `0.28` for the paper default). |

## Constraints

- Requires `--advantage-estimator grpo`.
- Incompatible with Dual-clip PPO (`--eps-clip-c`).
- Because this runs via `--loss-type custom_loss`, options restricted to built-in `policy_loss` (`--log-probs-backend fused_lm_head`, `--skip-actor-forward-only`) are rejected by startup argument validation.

## Relation to Score Centering

Unweighted full-vocabulary BPO (arXiv:2609.15987, Eq. 26) yields the same expected policy gradient identity as [Score Centering](../../infra_features/score_centering/README.md) (arXiv:2609.20807, Eq. 4). Unlike Score Centering, which requires recording rollout top-k log-probabilities, this plugin implements the sampled-token reweighted objective (Eqs. 14–15) and only needs sampled-token log-probabilities.
