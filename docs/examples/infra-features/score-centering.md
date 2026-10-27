---
title: "Score Centering"
description: "Center off-policy policy gradients using the sampler's top-k probabilities, with optional TIS or MIS weights."
# Generated from examples/infra_features/score_centering/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Correct off-policy score drift using the sampler's top-k probabilities, with optional truncated or masked importance weights.

This implements [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807), including the efficient top-k approximation in Appendix A.

## Enable it

Add these arguments to an existing text-only GRPO training recipe:

```bash
--loss-type score_centering \
--advantage-estimator grpo \
--rollout-top-logprobs-num 128 \
--score-centering-is none \
--rollout-temperature 1.0 \
--rollout-top-p 1.0 \
--rollout-top-k -1 \
--use-rollout-logprobs \
--disable-grpo-std-normalization \
--calculate-per-token-loss
```

Selecting `--loss-type score_centering` automatically enables `--sglang-enable-fp32-lm-head` for rollouts. This avoids rounding LM-head logits to lower precision before computing sampling probabilities. It can increase serving compute and memory costs, depending on the SGLang version.

Keep reward mean subtraction enabled. Disabling standard-deviation normalization gives the paper's group-centered rewards. This is a separate REINFORCE-style loss: PPO clipping parameters do not apply. Existing batch size and update scheduling still control how many updates consume a rollout batch; choose them explicitly when reproducing an experiment.

Choose `--score-centering-is tis` for weights clipped at `--score-centering-tis-clip` (default 2). Choose `mis` to retain ratios in `[--score-centering-mis-low, --score-centering-mis-high]` (defaults 0.5 and 5), setting other weights to zero. These weights are centered together with the score. Use these options instead of `--use-tis` or a custom TIS function.

The existing entropy and reference-KL loss options remain available. They are separate regularizers; the score-centering identity applies to the policy-gradient term.
With filtered sampling, sampling-support replay rules prohibit reference KL; entropy is computed over the recorded support.
The loss keeps full-vocabulary actor scores for reference KL, matching the reference forward even if called directly with replayed candidates. This does not remove the startup restriction above.
Reference-KL tokens with non-finite probabilities or absolute log-probability ratios above 40 are excluded before exponentiation to keep gradients finite. With unbiased KL, this also applies to the train/rollout ratio. `train/kl_invalid_fraction` reports the excluded fraction.

Training logs include `train/train_rollout_logprob_abs_diff` and `train/train_rollout_kl`. The latter uses the same masked, sampled-token k3 estimator of KL(rollout || train) as the policy loss. It is a detached diagnostic and is emitted even when reference-KL regularization is disabled.

## Sampler KL (demo)

A main motivation is to help maintain policy entropy while limiting drift from
the generating policy. The preliminary observations below suggest this effect
in our math/Adam setup; the task, reward, optimizer, learning rate, precision
and sampler/KL schedule can all affect it.

Add `--sampler-kl-coef 0.03` to optimize `KL(q_sampler || p_trainer)` alongside
SFC. The coefficient defaults to zero and is constant when enabled, including
fresh rollouts. Sampler probabilities are detached; the trainer remains
differentiable. KL is independent of reward advantages, TIS/MIS and reference
KL, and uses the existing token or equal-response reduction.

Choose `--sampler-kl-estimator`:

- `head_tail` (default): exact transported top-k head plus a sampler tail modeled
  as proportional to the trainer tail. This approximates an arbitrary true tail.
- `sampled_k1`: `-log_r`, where `log_r = log(p[token]) - stop_gradient(log(q[token]))`.
- `sampled_k3`: `expm1(log_r) - log_r`, with gradients through the uncapped ratio.

Both sampled choices use actual generated-token probabilities. Masked tokens are
zeroed before arithmetic; active nonfinite sampled ratios or k3 overflow raise
an error (including the companion k3 calculation in k1 mode). A single k1 value
can be negative and its gradient need not vanish when p=q; both sampled
estimators match fixed-history KL in expectation over q. k3 has zero per-token
loss and gradient when p=q. SFC still needs its candidate head in all modes.

`train/sampler_kl_loss` and `train/sampler_kl_weighted_loss` report the raw and
weighted term. `train/train_rollout_kl` remains a separate detached diagnostic.
Inspired by the [SFC + trainer-generator KL proposal](https://huggingface.co/blog/NormalUhr/sfc-kl);
this demo uses constant beta rather than the proposal's age-dependent schedule.

### Preliminary 32-update experiments

Qwen3.6-35B-A3B (BF16 trainer / FP8 sampler), INTELLECT-2 verifiable math,
SFC K128, equal-response reduction, GBS256 (32 prompts x 8 responses),
Adam (LR 1e-6, betas 0.9/0.98, weight decay 0.1), seed 1234. Each arm starts
fresh and uses the same first 1,024 training prompts once (8,192 responses).
Greedy held-out evaluation uses 128 questions at updates 0/16/32. Where enabled,
TIS is detached min(p/q, 2), and sampler KL has constant beta 0.03 including
age zero. The combined arm adds KL independently of the TIS-weighted SFC term.

| Recipe | Update 0 | Update 16 | Update 32 |
| --- | ---: | ---: | ---: |
| SFC + TIS | 75/128 (58.59%) | 75/128 (58.59%) | 84/128 (65.62%) |
| SFC + sampler KL (top-k) | 80/128 (62.50%) | 86/128 (67.19%) | 91/128 (71.09%) |
| SFC + sampler KL (k1) | 76/128 (59.38%) | 79/128 (61.72%) | 91/128 (71.09%) |
| SFC + sampler KL (k3) | 75/128 (58.59%) | 85/128 (66.41%) | 90/128 (70.31%) |
| SFC + TIS + sampler KL (top-k) | 77/128 (60.16%) | 83/128 (64.84%) | 87/128 (67.97%) |

Entropy is the native masked equal-response trainer entropy in nats. The
entropy regularization coefficient is zero in every arm. Compare nonoverlapping
eight-update means to reduce batch noise:

| Recipe | Updates 1-8 | Updates 25-32 | Relative change |
| --- | ---: | ---: | ---: |
| SFC + TIS | 0.12993 | 0.11728 | -9.7% |
| SFC + sampler KL (top-k) | 0.13115 | 0.12766 | -2.7% |
| SFC + sampler KL (k1) | 0.13214 | 0.13221 | +0.1% |
| SFC + sampler KL (k3) | 0.13090 | 0.13302 | +1.6% |
| SFC + TIS + sampler KL (top-k) | 0.13275 | 0.12637 | -4.8% |

Sampler KL appears to help maintain entropy here: its arms show smaller drops
or nearly flat entropy over this horizon compared with SFC + TIS. The combined
SFC + TIS + top-k KL arm has a smaller drop than SFC + TIS and reaches 87/128
at update 32 versus 84/128. This is a preliminary observation with differing
generated histories, not evidence of a universal improvement or causal proof.
The [reference proposal](https://huggingface.co/blog/NormalUhr/sfc-kl) motivates
the feature; these task/Adam/fresh-sampler results do not establish that its
empirical conclusions transfer unchanged, or that TIS must always be removed.

These are single-seed observations from E011 F/G/L/M/K, with independent
responses and different baseline scores. F/G use the first 32 updates of
128-update runs; K uses the first 32 of 96; k1/k3 stopped at 32. The experiments
used the historical experiment fork, not this rebased demo. [Aggregate records,
entropy windows and terminal-receipt hashes](sampler_kl_32_updates.json) preserve
the reported points. [Per-update metrics](https://github.com/radixark/miles/blob/main/examples/infra_features/score_centering/sampler_kl_32_updates.csv) include all
32 normal updates per arm: raw/weighted KL, PG/total loss, entropy, mismatch and
gradient norm. Missing KL fields in the control are blank.

## How it works

When the rollout distribution differs from the current trainer, even a constant reward can produce an unwanted average policy update. Score centering subtracts the expected weighted score under the rollout distribution.

Let `p` be the current trainer distribution, `q` the distribution that actually sampled the token, `H` the stored candidates, `A` the detached advantage, and `f` the selected importance-weight function. The implementation computes:

```text
rho   = max(1 - sum(q[H]), 1e-6) / max(1 - sum(p[H]), 1e-6)
alpha = rho * f(1 / rho)
loss  = -A * (stop_gradient(f(p[token] / q[token])) * log(p[token])
             - sum(stop_gradient(q[H] * f(p[H] / q[H]) - alpha * p[H]) * log(p[H])))
```

Outside `H`, the approximation models `q` as `rho * p`. The sampled token always uses its recorded `q[token]`, including when it is outside `H`. All weights and correction coefficients are detached. Full-distribution centering cancels the expected constant-reward gradient; the top-k version approximates the true tail and does not guarantee exact cancellation for an arbitrary tail.

`--rollout-top-logprobs-num K` plays a different role in the two sampling modes:

- **Unfiltered sampling** (`--rollout-top-p 1.0 --rollout-top-k -1`): Miles automatically uses `selected` mode. The sampler draws from the full vocabulary, so Miles records only its top `K` tokens as `H` and the tail model covers the rest. `K` is a real truncation and must be set; a larger `K` leaves less to the tail model.
- **Filtered sampling** (top-p/top-k): Miles automatically uses `support` mode when candidate recording is enabled. The sampler draws only from the realized support, and Miles records that whole support with its post-filter probabilities. `H` is then the full sampling distribution and nothing is approximated. `K` only sizes the arrays: it must be at least `--rollout-top-k` and hold the realized support, which cutoff ties can make larger; a larger support fails validation. The trainer renormalizes its log probabilities over the same support.

SGLang reports three kinds of log probability:

| SGLang field | Distribution | Filtered | Sums to 1 on the support |
| --- | --- | --- | --- |
| `output_token_sampling_logprobs` (selected or support mode) | `q'`: renormalized after top-k/top-p filtering | yes | yes |
| `output_token_logprobs`, `output_top_logprobs` | `p_T = softmax(logits / T)` over the full vocabulary | no | no, less than 1 under filtering |
| `output_token_logprobs`, `output_top_logprobs` with `SGLANG_RETURN_ORIGINAL_LOGPROB=1` | `log_softmax(logits)`, without the temperature | no | — |

The first row is the sampler under filtering and the second the sampler without filtering. Miles-managed rollout workers set `SGLANG_RETURN_ORIGINAL_LOGPROB=0`, so the third row does not occur. Each configuration reads:

| Configuration | Sampling-support replay (`append_sampling_metadata`) | Candidates (`append_rollout_topk_logprobs`) | SGLang source |
| --- | --- | --- | --- |
| Sampling-support replay only (`--rollout-top-logprobs-num 0`) | stores the support; returns the sampled-token log probability | nothing (`K = 0`) | `output_token_sampling_mask`, `output_token_sampling_logprobs` |
| Score centering, unfiltered | not called (no `return_sampling_mask`) | stores the top `K` candidates | `output_top_logprobs`, or its flat arrays on the session path |
| Score centering, filtered | stores the support; picks the sampled-token log probability from its row | stores `[n, K]`: the same support IDs with their row log probabilities | both read the same `output_token_sampling_mask`, `output_token_sampling_logprobs` |

For support replay without reference KL, the trainer gathers only sampled and support logits, normalizes over that support, and saves activations proportional to the candidate count rather than the vocabulary size. Other paths retain full-vocabulary normalization, reducing normalization scalars and selected logits across tensor-parallel ranks. It excludes padded vocabulary entries and computes probabilities in float32 for BF16/FP16 models. `--log-probs-chunk-size` controls the temporary computation size; `--recompute-loss-function` can trade computation for saved activations. No full vocabulary is gathered across ranks.

## Rollout and data contract

Native SGLang generation, the legacy rollout path, and both session-server versions request candidate probabilities at generation time when `--rollout-top-logprobs-num` is positive. Miles derives the logprob mode at startup from `--rollout-top-p` and `--rollout-top-k`. On these training requests, the configured count and derived mode override any client-supplied `top_logprobs`, `top_logprobs_num`, or `sampling_logprobs_mode`. A count of zero disables candidate recording without requesting support-wide probabilities. Session-server training requests ask for the unfiltered candidates as base64 `[n, K]` arrays (`output_top_logprobs_{val,idx}_flat_b64`) instead of nested `output_top_logprobs` rows; Miles decodes either form. Each `Sample` carries:

- `rollout_topk_token_ids`: int32 array shaped `[response_length, k]`.
- `rollout_topk_log_probs`: float32 array of the same shape.
- `rollout_log_probs`: the actual sampled-token log probabilities.

Evaluation requests skip this collection and may use independent sampling settings, including greedy decoding. The built-in agentic producer marks evaluation sessions when creating them; custom session clients should create them with `POST /sessions` with JSON body `{"evaluation": true}`.

Unused candidate slots and non-trained observation rows contain token ID `-1` and log probability `-inf`. Tool-observation masks, multi-turn merging, retries, trailing-token trimming, and truncation preserve row alignment. Session serialization retains both arrays. For score centering in support mode, training batches share the recorded support IDs when every sample has exactly the same candidate order and prefix padding. A per-row candidate count preserves observation rows and masked generated rows; any mismatch keeps the original arrays for the whole batch. The trainer reconstructs only its context-parallel rows. The source Samples and session payloads still retain both representations.

Custom rollout producers must supply these fields with probabilities from the actual generation call and no repeated non-negative token ID in a row. Conversion always rejects missing candidate IDs, candidate log probabilities, or sampled-token log probabilities, including from custom producers. With `--ci-test`, Miles also validates the complete sample before training, rejecting duplicate candidates, sampling-support mismatches, and disagreeing sampled/candidate probabilities. This full-sample validation is skipped in normal training because sorting and support matching are expensive on long responses. Rescoring old rollouts with newer weights is not a substitute. With the feature disabled, requests and the session wire format remain unchanged.

## Supported configurations and limits

- The shared loss is wired into Megatron and FSDP. Candidate selection supports tensor parallelism, packed (`thd`) and padded (`bshd`) zigzag context parallelism, and packed all-gather context parallelism.
- Sampling requires a fixed positive temperature and `min_p=0` on every call. Filtered sampling, including top-p filtering, automatically uses support mode and requires a positive `top_k`, for example `--rollout-top-p 0.9 --rollout-top-k 64 --rollout-top-logprobs-num 128`. Global filtered rollout settings automatically enable sampling-support replay; per-request overrides are checked when each request is built. See the [sampling-support replay guide](/advanced/sampling-support-replay) for its request and server requirements.
- Filtered sampling requires SGLang with support log probabilities (SGLang PR [#40932](https://github.com/sgl-project/sglang/pull/40932), included in the `sglang-miles` branch by [#41047](https://github.com/sgl-project/sglang/pull/41047), merge commit [`ae04cb14046896b6d453758c5769d639deedd353`](https://github.com/sgl-project/sglang/commit/ae04cb14046896b6d453758c5769d639deedd353) or a descendant containing it). External servers must set `SGLANG_RETURN_ORIGINAL_LOGPROB=0` like Miles-managed workers. OpenAI session responses must expose the SGLang fields above in `choices[0].meta_info`; a generic OpenAI-compatible server without that metadata is insufficient.
- The SGLang router must include [sgl-router-for-miles #21](https://github.com/radixark/sgl-router-for-miles/pull/21) (`df2c790`; see `Git Commit` in `smg --version-verbose`), which raises the chat `top_logprobs` limit from 20 to 128 and forwards `sampling_logprobs_mode`.
- Compact candidate transport on the session path requires SGLang PR [#42890](https://github.com/sgl-project/sglang/pull/42890), included in the `sglang-miles` branch by [#42838](https://github.com/sgl-project/sglang/pull/42838), and the router change [sgl-router-for-miles #24](https://github.com/radixark/sgl-router-for-miles/pull/24). Without either, SGLang returns the larger nested rows, where a `-inf` candidate arrives as `null` and is stored as NaN.
- Constrained/custom sampling, speculative decoding, true-on-policy mode, OPD, multi-LoRA/Tinker losses, sequence masking, custom policy-loss reducers, custom train-data converters, and logprob recomputation via prefill are rejected. Multimodal token expansion is not supported. The initial advantage estimator is GRPO.
- Retaining `k=128` uses about 1 KiB per response position for the two arrays, before transport overhead. Eligible support-mode training batches replace the duplicate ID array with one int32 count per response position, saving `4 * (k - 1)` bytes per position in training transport and storage. Source Samples and session payloads keep the original storage cost. Larger `k` improves the unfiltered tail approximation at additional storage and compute cost.

## Metrics and verification

Training logs include `sc_correction`, `sc_train_head_mass`, `sc_rollout_head_mass`, `sc_tail_ratio`, `sc_importance_weight`, `train_rollout_kl`, and `train_rollout_logprob_abs_diff`, under the usual `train/` namespace. Small head mass means more of the distribution is approximated by the tail model. Large tail ratios indicate a substantial mismatch in remaining mass.
For filtered sampling, the head covers the full support, so both head masses and the tail ratio should be approximately one.

The numerical tests compare gradients against an independent dense-distribution oracle for all three weighting modes, including sampled tokens outside the head, constant rewards, and tiny tails. Pipeline tests exercise real native/session producers, serialization, data-parallel splitting, masks, advantage computation, regularization, and checkpointed loss scaling. Run the focused tests in the repository's test environment:

```bash
python -m pytest tests/fast/backends/training_utils/test_score_centering.py \
    tests/fast/backends/training_utils/test_score_centering_pipeline.py \
    tests/fast/backends/training_utils/test_score_centering_filtered.py \
    tests/fast/backends/training_utils/test_score_centering_support.py
python -m pytest \
    tests/fast/backends/training_utils/test_score_centering_distributed.py
MILES_TEST_CUDA_DISTRIBUTED=1 python -m pytest \
    tests/fast/backends/training_utils/test_score_centering_distributed.py
```

The distributed tests use four CPU/Gloo processes or four CUDA/NCCL processes (TP=2, CP=2), all four context layouts and all weighting modes, compact support replay with reference KL on and off, plus BF16 selected-probability gradients. Support-scoring tests check saved tensor shapes and float32/64/BF16 gradients against a dense oracle. The independent dense-gradient oracle also runs on CUDA when available. These are correctness tests, not a reproduction of the paper's GPU training results.

For a real SGLang server, also run the opt-in protocol probe:

```bash
MILES_LIVE_SCORE_CENTERING_ENDPOINT=http://127.0.0.1:30000 \
MILES_LIVE_SCORE_CENTERING_MODEL=/path/to/model \
MILES_LIVE_SCORE_CENTERING_SERVED_MODEL=your-served-model \
python -m pytest --confcutdir=tests/manual tests/manual/test_score_centering_live.py
```

Set `SGLANG_RETURN_ORIGINAL_LOGPROB=0` on the server before starting it. The probe checks unfiltered native and OpenAI response metadata using the production candidate collector and validator, and checks temperature scaling at 0.7, 1.0 and 1.3. It warms the shared prompt first so that cached and uncached prefills do not confound the temperature comparison. Set `MILES_LIVE_SCORE_CENTERING_ARTIFACT_DIR` to keep the raw responses.
