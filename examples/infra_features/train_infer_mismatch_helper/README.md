# Rollout Correction Methods

Rollout correction (e.g, TIS, MIS) through algorithmic methods.


## Quick Takeaway

This function is used to solve offline scenarios through algorithmic adaptations, e.g. TIS/MIS.

We included 3 rollout correction algorithms:

1. decoupled, 3-policies PPO with rollout importance sampling
2. direct rollout policy overwriting in the standard PPO
3. pure REINFORCE loss (without PPO clipping) with rollout importance sampling


`--use-tis`: use this flag to **turn on TIS/MIS** for rollout correction (details in **Algorithms**).
You may specify the **IS/RS configs** with a config file using `--custom-config-path`.

`--use-rollout-logprobs`: When use this flag, the logprobs will **not** be recomputed by training engine - rollout log probs will be directly used in PPO/GRPO loss.

`--get-mismatch-metrics`: When you don't want to add TIS/MIS, but still want to monitor the mismatch-related metrics (e.g. rollout-training KL). It will **only return mismatch metrics** but not change the loss in any way.


## Algorithms

We give examples of the algorithms for solving the training-inference mismatch issue.

### [Baseline: No Mismatch Correction] Standard PPO

This is the basic PPO algorithm with potentially training-inference mismatch issue when the output of SGLang and Megatron does not exactly match.

$$
L_{\text{PPO}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
  \min \left(
    \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{blue}{\text{Megatron}}}(y \mid x)} A_t,
    \mathrm{clip}\left(
      \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{blue}{\text{Megatron}}}(y \mid x)},
      1 - \epsilon,
      1 + \epsilon
    \right) A_t
  \right)
\right].
$$

### Bypassing PPO importance sampling

Like REINFORCE, we directly use the rollout engine's log probs as the old policy in offline PPO's importance sampling, rather than the recomputed log-probs from the training engine.

$$
L_{\text{PPO-bypass}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
  \min \left(
    \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{red}{\text{SGLang}}}(y \mid x)} A_t,
    \mathrm{clip}\left(
      \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{red}{\text{SGLang}}}(y \mid x)},
      1 - \epsilon,
      1 + \epsilon
    \right) A_t
  \right)
\right].
$$

Advantages: 

- Efficiency: skip `log_prob` recomputation on training engine. Reduce one expensive forward pass on all the generated trajectories.

### Decoupled, 3-policy PPO Importance Sampling  

[Decoupled PPO](https://arxiv.org/pdf/2110.00641) achieves batch-independent PPO by decoupling two roles: Proximal Policy (anchor policy for PPO clipping, control update size) and Behavior Policy (for off-policy correction in importance sampling). Therefore, there are totally 3 roles engaged in this mode, **target policy** $\pi_\theta$, **proximal policy** $\pi_{\textcolor{blue}{\text{old}}}$, and **behavior policy** $\pi_{\textcolor{red}{\text{SGLang}}}$. $\pi_{\textcolor{blue}{\text{old}}}$ is recomputed with Megatron at the beginning of each training step.

$$
L_{\text{PPO-decoupled}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
    \frac{\pi_{\textcolor{blue}{\text{old}}}(y \mid x)}{\pi_{\textcolor{red}{\text{SGLang}}}(y \mid x)}
  \min \left(
    \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{blue}{\text{old}}}(y \mid x)} A_t,
    \mathrm{clip}\left(
      \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{blue}{\text{old}}}(y \mid x)},
      1 - \epsilon,
      1 + \epsilon
    \right) A_t
  \right)
\right].
$$

Advantages:

- Achieves batch size invariance and efficient stale data utilization
- Enables accurate off-policy metrics monitoring

## APIs of Algorithms

You may choose from above algorithms with the two command-line flags below. They are the
only CLI flags this feature adds; everything in **Configs and Recommended Settings** is a
key in the YAML file you pass to `--custom-config-path`.

`--use-rollout-logprobs`: True if only use `rollout_log_probs` to compute the loss, bypassing old_log_probs calculated by training engine;

`--use-tis`: True if apply importance sampling/rejection sampling to loss.

| `use_rollout_logprobs` | `use_tis` | Algorithm | Policies |Compute old_log_probs | Batch Invariant | Recommended TIS Mode |
|-----------------|-------------|-----------|--------------|---------------|-----------------|----------------------|
| False | False | Standard PPO (Algorithm 0) | 2 ($\pi_\theta$, $\pi_{\textcolor{blue}{\text{old}}}$)|Yes | No | N/A |
| True | False | Bypassing PPO (Algorithm 3) | 2 ($\pi_\theta$, $\pi_{\textcolor{red}{\text{SGLang}}}$) |🚀 Skipped | No | N/A |
| False | True | Decoupled PPO (Algorithm 2) | 3 ($\pi_\theta$, $\pi_{\textcolor{blue}{\text{old}}}$, $\pi_{\textcolor{red}{\text{SGLang}}}$)  |Yes  | Yes | token/seq/geo |

## Configs and Recommended Settings

When choosing to use importance sampling or rejection sampling for mismatch correction
(`--use-tis` enabled, Algorithm 2 & 3), you may specify the IS modes and applied levels.

### Config keys

These are **not command-line flags**. They live in the YAML file the run points at with
`--custom-config-path`, and `mis.py` reads them off the parsed config. The reference file
is [`mis.yaml`](mis.yaml), wired up like this:

```bash
--use-tis
--custom-config-path examples/infra_features/train_infer_mismatch_helper/mis.yaml
--custom-tis-function-path examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
```

`use_tis`: Enable importance sampling. The IS weight will be multiplied by the policy gradient loss.

- `tis_mode`: Mode for IS. Allowed mode: **truncate**, **clip**, **mask**.
- `tis_lower_bound`, `tis_upper_bound`: Bounds for IS weights.
- `tis_level`: Allowed levels: **token**, **sequence**, **geometric**. See explanations below.
- `tis_batch_normalize`: Normalize IS weights to mean=1.0 across batch


`use_rs`: Enable rejection sampling. When choosing to use rejection sampling, the tokens/sequences with an IS weight out of threshold will be directly masked. Those rejected tokens/sequences will not be considered for loss averaging.

- `rs_lower_bound`, `rs_upper_bound`: Bounds for RS. Unset falls back to the `tis_` bounds.
- `rs_level`: Allowed levels: **token**, **sequence**, **geometric**. See explanations below.
- `rs_veto_threshold`: Sequence-level rejection threshold for catastrophic mismatches

### Importance Sampling

For both IS and RS, we provided 3 levels: **token**, **sequence**, **geometric**.

**Token Level (default)**:

Computes importance weights independently for each token:
$w_i = \exp\left(\log \pi_{\text{train}}(x_i) - \log \pi_{\text{rollout}}(x_i)\right)$

Characteristics: Biased but computationally simple, suitable for most scenarios

**Sequence Level**:

Uses the product of all token weights as the sequence weight:
$w_{\text{seq}} = \exp\left( \sum_i \left( \log \pi_{\text{train}}(x_i) - \log \pi_{\text{rollout}}(x_i) \right) \right)$

Characteristics: Unbiased but high variance, suitable for sequence-level optimization

**Geometric Level**:

Uses geometric mean to compute sequence weight:
$w_{\text{seq}} = \exp\left( \frac{1}{n} \sum_{i=1}^{n} \left( \log \pi_{\text{train}}(x_i) - \log \pi_{\text{rollout}}(x_i) \right) \right)$

Characteristics: Biased but low variance, balances bias and variance

### Rejection Sampling

**Token Level**: Reject tokens with IS weight out of threshold

**Sequence Level:** Reject sequences with mean IS weight out of threshold

**Geometric Level:** Reject sequences with geometric mean IS weight out of threshold

- Extremely selective: Requires near-perfect policy match
- High rejection rate: Only suitable for very slight distribution shifts

**Veto Mechanism**:

Veto mechanism rejects sequences with catastrophically low token probabilities.
Reject entire sequence if $\exists t \in T$ such that $\rho_t < C_{\text{veto}}$

- Prevents catastrophic updates from tokens with near-zero probability under $\pi_{\text{old}}$
- Independent of IS/RS settings

*Typical values: $10^{-4}$ to $10^{-6}$*

## Mismatch Metrics

These metrics help diagnose policy divergence and guide hyperparameter tuning. Which
ones you get depends on the correction function in use. The built-in one
(`vanilla_tis_function`, used when `--custom-tis-function-path` is unset) reports only
`tis`, `tis_clipfrac` and `tis_abs`. Everything below comes from `mis.py`, so it needs
the `--custom-tis-function-path` wiring shown above.

All names below are logged under the `train/` namespace, and every key `mis.py` produces
carries a `mis_` prefix that its wrapper adds on the way out — so `training_log_ppl`
reaches wandb as `train/mis_training_log_ppl`. The two exceptions, marked in the tables,
come from miles itself and are not prefixed.

### Mismatch Monitoring Metrics

These metrics quantify the difference between training and rollout policies. `mis.py`
computes them whenever `rollout_log_probs` are available, whether or not IS/RS correction
is actually applied.

| Metric Name | Description |
|------------|-------------|
| `mis_training_log_ppl` | Negative mean log probability under training policy: $-\mathbb{E}[\log \pi_{\text{train}}]$ |
| `mis_training_ppl` | Perplexity of training policy: $\exp(-\mathbb{E}[\log \pi_{\text{train}}])$ |
| `mis_rollout_log_ppl` | Negative mean log probability under rollout policy: $-\mathbb{E}[\log \pi_{\text{rollout}}]$ |
| `mis_rollout_ppl` | Perplexity of rollout policy: $\exp(-\mathbb{E}[\log \pi_{\text{rollout}}])$ |
| `mis_kl` | Forward KL divergence estimator: $\mathbb{E}[\log \pi_{\text{rollout}} - \log \pi_{\text{train}}]$ |
| `mis_k3_kl` | K3 KL estimator: $\mathbb{E}[\exp(r) - r - 1]$ where $r = \log \pi_{\text{train}} - \log \pi_{\text{rollout}}$ |
| `mis_log_ppl_diff` | Log perplexity difference|
| `mis_log_ppl_abs_diff` | Absolute log perplexity difference |
| `mis_ppl_ratio` | Perplexity ratio |
| `mis_chi2_token`, `mis_chi2_seq` | Token- and sequence-level $\chi^2$ divergence between the two policies |
| `train_rollout_logprob_abs_diff` (miles, unprefixed) | Token-level absolute log probability difference |

**Usage**: These metrics help you monitor policy drift. Large values indicate a significant mismatch between the training and rollout engines.

### IS/RS Correction Metrics

These metrics track importance sampling weights and corrections. They are only computed when `--use-tis` is enabled.

When using `--custom-tis-function-path` pointing to MIS implementation (e.g., `mis.py`), additional fine-grained metrics become available. Under the shared `mis_` prefix, the `tis_` and `rs_` parts say which stage produced the number.

| Metric Name | Description | Emitted when |
|------------|-------------|--------------|
| `ois` (miles, unprefixed) | On-policy importance sampling ratio: $\exp(\log \pi_{\text{train}} - \log \pi_{\text{old}})$ | `--use-tis` (Algorithm 2 only) |
| `mis_tis_weight_before_bound` | Raw IS weights before any bounding: $\exp(\text{log-ratio})$ | `use_tis` |
| `mis_tis_weight_after_bound` | IS weights after the `tis_mode` bounding | `use_tis` |
| `mis_tis_truncate_fraction` | Fraction of weights truncated | `tis_mode: truncate` |
| `mis_tis_clip_fraction_low` / `mis_tis_clip_fraction_high` | Fraction of weights clipped below / above the bound | `tis_mode: clip` |
| `mis_tis_mask_fraction_low` / `mis_tis_mask_fraction_high` | Fraction of tokens rejected below / above the bound | `tis_mode: mask` |
| `mis_rs_mask_fraction_low` / `mis_rs_mask_fraction_high` | Fraction of tokens rejected by rejection sampling | `use_rs` |
| `mis_rs_catastrophic_token_fraction` | Fraction of catastrophic tokens (below the veto threshold) | `rs_veto_threshold` set |
| `mis_rs_catastrophic_seq_fraction` | Fraction of sequences holding a catastrophic token | `rs_veto_threshold` set |
| `mis_is_ratio_mean_after_tis_rs` | Mean IS weight after both TIS and RS | `use_tis` |
| `mis_batch_norm_factor` | Batch normalization factor applied to weights (1.0 when off) | `use_tis` |
| `mis_is_ratio_mean_final` / `mis_is_ratio_min_final` / `mis_is_ratio_max_final` | Final IS weight statistics actually multiplied into the loss | `use_tis` |

## Reference

We thank the materials below for their excellent findings and theories.

1. [Mathematical Formulations of Rollout Correction Methods in verl (Yingru Li)](https://github.com/szrlee/verl/blob/yingru/rollout_correction/docs/advance/rollout_corr_math.md).
2. [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl)
3. [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
