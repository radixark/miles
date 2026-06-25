# Predictive Routing Replay (PR²)

This document describes Miles's implementation of Predictive Routing Replay
(PR²): the runtime architecture, the algorithm (and where Miles diverges
from the paper), and the user-facing configuration parameters.

Core source:

- `miles/backends/megatron_utils/predictive_router_replay.py` — runtime: `PredictiveReplayController` state machine, patched `TopKRouter.forward`, state registry, public re-exports of the stateless helpers below.
- `miles/backends/megatron_utils/predictive_router_stabilization.py` — stateless §4.2 helper: depth-aware layer-scale gate.
- `miles/backends/megatron_utils/predictive_router_loss.py` — stateless `compute_predictive_loss` (`kl` / `kl-post`), top-k boundary-margin sample reweighting, synthetic-loss synchronization.
- `miles/backends/megatron_utils/predictive_router_utils.py` — packing/transport of recorded predictive microbatches.
- `miles/backends/megatron_utils/predictive_train_schedule.py` — train-pass plan (skip → compute per mini-step).
- `miles/backends/megatron_utils/router_replay_artifacts.py` — on-disk artifact naming/loading.
- `miles/backends/megatron_utils/router_replay_saver.py` — saver protocol.
- `miles/backends/megatron_utils/model.py`, `model_provider.py`, `actor.py` — integration with the Megatron training loop.
- `miles/utils/arguments.py` — CLI flag definitions (`Predictive Routing Replay (PR²)` argparse group).
- `miles/utils/replay_base.py` — `BaseReplayManager` / `RoutingReplayManager` abstractions PR² depends on.
- `scripts/run-qwen3-30B-A3B-pr2-{paper,miles}.sh` — example launch scripts.

---

## 1. Runtime layout

`predictive_router_replay.py` owns the patch on `TopKRouter.forward` and exposes `PredictiveReplayController` as the single control-plane object. The controller holds:

- registered per-layer router states,
- the global predictive action (RECORD / SKIP / COMPUTE_PREDICTIVE_LOSS / DISABLED),
- the recorded predictive microbatch queue,
- per-train-step usage flag,
- predictive metrics + metric-tensor capture.

`model.py` collects recorded predictive tensors after the old log-prob forward, delegates train-pass mode changes to the controller, and gates predictor optimizer groups when a train step has no valid predictive data.

`actor.py` runs a single train pass over each rollout batch and selects the predictive mode per mini-step via `get_predictive_train_mode_for_step(step_id)`: mini-step `step_id=0` runs in `SKIP_PREDICTIVE` (matches paper Algorithm 3 line 3 condition `i=1`), mini-steps `step_id>=1` run in `COMPUTE_PREDICTIVE_LOSS`. The replay read indices and predictive microbatch cursor are reset before the compute steps begin.

## 2. Data flow

1. Old actor log-prob pass runs with `RouterPredictiveAction.RECORD`.
2. Each patched router records local old inputs/logits and logs the predicted bias into the routing replay artifact.
3. `model.forward_only(...)` collects per-layer recorded tensors and packs them into a `RecordedPredictiveMicrobatch`.
4. Inside the single actor train pass, the predictive mode is set per mini-step:
   - mini-step `step_id=0`: `skip` — pure on-policy update, no predictive loss (paper Algorithm 3 line 3, condition `i=1`)
   - mini-step `step_id>=1`: `compute` — loads the recorded predictive microbatch into router-local state
5. During the compute mini-steps the patched routers compute the predictor loss and top-k metrics, then normal actor optimization proceeds.

## 3. Artifact layout

`router_replay_artifacts.py` is the shared naming/loading layer.

| File | Purpose |
|---|---|
| `{step}_tp{tp_rank}_pp{pp_rank}.pt` | Main routing-replay payload |
| `{step}_tp{tp_rank}_pp{pp_rank}_predictive_metrics.json` | Per-layer predictive metrics |
| `{step}_tp{tp_rank}_pp{pp_rank}_predictive_metric_tensors.pt` | Per-layer captured tensors (input/logits/applied_delta) |

`router_replay_saver.py` writes these via the shared protocol.

---

## 4. Algorithm

### 4.1 Paper PR² (reference)

**Rollout phase** (paper Eq. 4–6):

$$\hat{\rho}^{(l)}_t = \mathrm{Softmax}\!\left(p^{(l)}_{\mathrm{old},t} + h^{(l)}_{\mathrm{old},t} W^{(l)}_p\right), \quad \hat{\mathcal{I}}^{(l)}_t = \mathrm{TopK}(\hat{\rho}^{(l)}_t, k)$$

The predictor delta is added to base logits and routed via softmax + top-k. The predicted index $\hat{\mathcal{I}}$ is cached for replay.

**Training phase** (paper Eq. 7–8 + Algorithm 3):

$$\mathcal{L}_{\mathrm{PR}^2} = \sum_{l=1}^L \mathbb{E}_t\!\left[ D_{\mathrm{KL}}\!\left(\langle\rho^{(l)}_t\rangle \,\big\|\, \hat{\rho}^{(l)}_t\right) \right]$$

Uniform token weighting, stop-grad on the current router's $\rho^{(l)}_t$ as teacher. Mini-step $i=1$ skips the predictive loss; predictor uses a dedicated learning-rate multiplier $\alpha$.

### 4.2 Miles enhancements

Miles adds a stabilization/safety layer + sample reweighting that the paper does not specify. See §6 for parameter semantics.

Miles keeps the stabilization layer deliberately small: a single rollout-side
gate plus a single training-side reweighting. Both default OFF, so omitting
them reproduces the paper exactly.

**Rollout-side stabilization** (`stabilize_predictive_delta_logits`) — **depth-aware layer gating**:

Form $\tilde{b}^{(l)}_t = h^{(l)}_{\mathrm{old},t} W^{(l)}_p$, then scale by a depth-dependent gate:

$$b^{(l)}_t \leftarrow \gamma_l \cdot \tilde{b}^{(l)}_t, \quad \gamma_l \in \{\mathrm{none}, \mathrm{linear}_\downarrow, \sqrt{}_\downarrow, \cos_\downarrow\}(l/L)$$

The intuition: predictor error compounds with depth, so later layers' deltas can be damped toward a floor $\gamma_{\min}$ while early layers stay near $\gamma_l = 1$.

**Training loss with sample reweighting**:

$$\mathcal{L}^{\mathrm{Miles}}_{\mathrm{PR}^2} = \sum_{l=1}^L \frac{\sum_t w^{(l)}_t \cdot D_{\mathrm{KL}}\!\bigl(\langle\rho^{(l)}_t\rangle \,\big\|\, \hat{\rho}^{(l)}_t\bigr)}{\sum_t w^{(l)}_t}, \quad w^{(l)}_t = w^{\mathrm{bdry},(l)}_t$$

where the **boundary weight** (`PREDICTIVE_BOUNDARY_LOSS_MAX_WEIGHT=w_max`, `PREDICTIVE_BOUNDARY_LOSS_MIN_MARGIN=m_min`) is $\min(w_{\max}, 1/\max(m^{(l)}_t, m_{\min}))$ with $m^{(l)}_t = p^{(l)}_{\mathrm{old},t,(k)} - p^{(l)}_{\mathrm{old},t,(k+1)}$ the top-k boundary margin. This concentrates supervision on the near-boundary tokens whose route the predictor is most likely to get wrong. When $w_{\max}$ is unset, all tokens are weighted equally (paper-faithful).

`compute_predictive_loss` then computes the chosen loss form (`kl` or `kl-post`) over the stabilized $\hat{\rho}$.

**Synchronization loss** (`build_synthetic_predictive_loss`): when a DP rank holds no valid predictive tokens, construct $(W_p \cdot x_{\mathrm{any}}).\mathrm{sum}() \cdot 0$ so $W_p$ retains a backward graph and the collective stays in sync.

---

## 5. Configuration parameters

All flags are exposed under the `Predictive Routing Replay (PR²)` argparse
group; run `python3 train.py --help` to see them in context. The tables
below list each by group with semantics and defaults.

### 5.1 Core (required when PR² is on)

| CLI flag | Default | Meaning |
|---|---|---|
| `--enable-predictive-routing-replay` | off | Master switch |
| `--bias-predictor-loss-type` | `kl-post` | `kl-post` = paper Eq. 8 objective; `kl` = Miles experimental delta-distribution KL |
| `--bias-predictor-lr-mult` | `1e3` | Predictor LR multiplier $\alpha$. Paper Appendix E.1 sweeps $\alpha \in \{5\times10^1, 10^2, 10^3, 10^4\}$ across models — Qwen3-30B off-{2,4,8}: $\{10^4, 10^3, 10^3\}$; OLMoE: $10^3$; Moonlight: $5\times10^1$–$10^2$. The default $10^3$ matches Qwen3 off-4/off-8 + OLMoE. |
| `--predictive-storage-dtype` | `fp32` | Cache dtype: `fp32` / `bf16` / `fp16`. **Miles divergence**: paper Table 3 uses mixed `bf16` for hidden features + `fp32` for router logits; Miles applies a single dtype to both. |

`--use-routing-replay` and `--use-miles-router` are upstream prerequisites,
not part of the PR² group.

### 5.2 Data sub-sampling

| CLI flag | Default | Meaning |
|---|---|---|
| `--predictive-downsample-batch-size` | unset (= keep all) | Sub-sample rate over rollout batch |
| `--predictive-downsample-max-len-limit` | unset | Per-sample token cap |
| `--predictive-max-total-tokens` | unset | Hard cap on packed-microbatch token count |

> **Miles divergence**: paper Appendix E.1 uses a per-response feature-cache length $T_c$ with uniform-along-length sub-sampling (Qwen3-30B: $T_c=2\text{K}$ with $T=16\text{K}$; OLMoE / Moonlight: $T_c=T=1\text{K}$, no sub-sampling). Miles instead exposes three per-microbatch caps and defaults to all unlimited. Leaving these flags unset reproduces the paper's OLMoE / Moonlight setting; reproducing the paper's Qwen3-30B $T_c=2\text{K}$ cap requires setting `--predictive-downsample-max-len-limit 2048` plus a corresponding `--predictive-max-total-tokens` value.

### 5.3 Miles enhancements (all default OFF — paper-faithful when omitted)

| CLI flag | Default | Meaning |
|---|---|---|
| `--predictive-layer-scale-schedule` | `none` | `none` / `linear_decay` / `sqrt_decay` / `cosine_decay` — depth-aware gate $\gamma_l$ |
| `--predictive-layer-scale-min` | `1.0` | Floor for $\gamma_l$ (1.0 = no decay) |
| `--predictive-boundary-loss-max-weight` | unset | Boundary weight cap $w_{\max}$, e.g. `4.0` |
| `--predictive-boundary-loss-min-margin` | `1e-4` | Denominator floor $m_{\min}$ |

The two mechanisms are independent: `--predictive-layer-scale-*` shape the rollout-side delta, `--predictive-boundary-loss-*` shape the training-side loss. A common Miles config is `--predictive-layer-scale-schedule sqrt_decay --predictive-layer-scale-min 0.5 --predictive-boundary-loss-max-weight 4.0`.

### 5.4 Router-logits artifact saving (optional, default OFF)

These flags control whether per-rollout router-logit artifacts are written
to disk (useful for offline analysis but expensive — per-rollout, per-TP-rank,
per-PP-rank tensors plus a JSON sidecar add significant I/O).

| CLI flag | Default | Meaning |
|---|---|---|
| `--router-logits-path` | unset | Directory to write artifacts to. Leave unset to disable. |
| `--router-logits-save-freq` | `0` | Save every N rollouts. `0` (default) disables saving even when `--router-logits-path` is set — explicitly set a positive integer to opt in. |
| `--router-logits-max-tokens` | unset | Cap saved artifacts to the first N tokens per step. Leave unset to save all tokens. |

### 5.5 Logged metrics

Per-layer metrics are emitted both to wandb (under `train/`) and to a JSON
sidecar `{step}_tp{tp_rank}_pp{pp_rank}_predictive_metrics.json`.

| Metric | Source |
|---|---|
| `predictive_loss` | The KL value (`kl` or `kl-post`), after sample reweighting |
| `predictive_topk_accuracy` | $|\hat{\mathcal{I}} \cap \mathcal{I}_{\mathrm{current}}|/k$ |
| `predictive_layer_gate_scale` | Depth gate $\gamma_l$ applied to the predicted delta |
| `predictive_stabilizer_scale` | Net scale applied by the stabilization layer ($=\gamma_l$ here) |
| `predictive_raw_bias_to_logits_ratio` | $\overline{\|\tilde b\|}/\overline{\|p_{\mathrm{old}}\|}$ before gating |
| `predictive_stabilized_bias_to_logits_ratio` | $\overline{\|b\|}/\overline{\|p_{\mathrm{old}}\|}$ after gating |
| `predictive_boundary_margin_{mean,min}` | Top-k boundary margin of the recorded route |
| `predictive_boundary_loss_weight_{mean,max,gt1_fraction}` | Stats of the boundary sample weights |

These metrics let you tell whether a Miles enhancement is changing routing
behavior in the way the corresponding flag intends.

---

## 6. Comparison with the PR² paper

| Aspect | Paper PR² | Miles |
|---|---|---|
| Rollout delta handling | Use $hW_p$ as-is | Optional depth-aware layer gate $\gamma_l$ |
| Token-level KL expectation | Uniform $\mathbb{E}_t$ | $\mathbb{E}_t[w_t \cdot D_{\mathrm{KL}}] / \mathbb{E}_t[w_t]$ with boundary weight $w_t$ |
| Layer weighting | Sum over $l$ | Sum + depth schedule $\gamma_l$ |
| Skip $i=1$ predictive loss | Yes (Algorithm 3) | Yes (state-machine controlled) |
| Loss form | `kl-post` (Appendix G ablates delta-matching) | Both (`kl` / `kl-post`); default `kl-post` |
| Empty-rank synchronization | Not specified | Synthetic-loss path |

When both Miles enhancements are off (`--predictive-layer-scale-schedule none`, `--predictive-boundary-loss-max-weight` unset), Miles reduces to the paper-faithful implementation. The `scripts/run-qwen3-30B-A3B-pr2-paper.sh` example exercises this configuration.
