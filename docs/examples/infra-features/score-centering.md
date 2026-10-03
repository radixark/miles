---
title: "Score Centering"
description: "Center off-policy policy gradients using rollout candidate probabilities, with optional TIS or MIS weights."
# Generated from examples/infra_features/score_centering/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Correct off-policy score drift using probabilities recorded during rollout, with optional truncated or masked importance weights.

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

Use the SGLang router from `sgl-router-for-miles` with [#21](https://github.com/radixark/sgl-router-for-miles/pull/21) included. It supports the 128-candidate recipe above without `--use-miles-router`; see [Rollout and data contract](#rollout-and-data-contract) for the request limits and installed-build check.

Keep reward mean subtraction enabled. Disabling standard-deviation normalization gives the paper's group-centered rewards. This is a separate REINFORCE-style loss: PPO clipping parameters do not apply. Existing batch size and update scheduling still control how many updates consume a rollout batch; choose them explicitly when reproducing an experiment.

Choose `--score-centering-is tis` for weights clipped at `--score-centering-tis-clip` (default 2). Choose `mis` to retain ratios in `[--score-centering-mis-low, --score-centering-mis-high]` (defaults 0.5 and 5), setting other weights to zero. These weights are centered together with the score. Use these options instead of `--use-tis` or a custom TIS function.

The existing entropy and reference-KL loss options remain available. They are separate regularizers; the score-centering identity applies to the policy-gradient term.
With filtered sampling, sampling-support replay rules prohibit reference KL; entropy is computed over the recorded support.
The loss keeps full-vocabulary actor scores for reference KL, matching the reference forward even if called directly with replayed candidates. This does not remove the startup restriction above.
Reference-KL tokens with non-finite probabilities or absolute log-probability ratios above 40 are excluded before exponentiation to keep gradients finite. With unbiased KL, this also applies to the train/rollout ratio. `train/kl_invalid_fraction` reports the excluded fraction.

Training logs include `train/train_rollout_logprob_abs_diff` and `train/train_rollout_kl`. The latter uses the same masked, sampled-token k3 estimator of KL(rollout || train) as the policy loss. It is a detached diagnostic and is emitted even when reference-KL regularization is disabled.

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

The sampling filter and candidate recording width are separate settings:

| Setting | Effect |
| --- | --- |
| `--rollout-top-k` | Filters the distribution used to generate tokens; `-1` disables top-k filtering. `--rollout-top-p` controls top-p filtering separately. |
| `--rollout-top-logprobs-num K` | Sets the width of the recorded candidate arrays without changing sampling. The default `0` disables candidate recording; score centering requires a positive value. |

`K` plays a different role in the two sampling modes:

- **Unfiltered sampling** (`--rollout-top-p 1.0 --rollout-top-k -1`): Miles derives `selected` mode and records up to the top `K` rollout tokens as `H`. Sampling still uses the full vocabulary; `K` truncates only the stored candidates. The tail model covers the remaining probability mass, and the sampled token may lie outside `H`.
- **Filtered sampling** (top-p/top-k): Miles derives `support` mode when candidate recording is enabled and records every token in the realized support `H`, with its post-filter probability. The trainer normalizes over that same fixed support, so `p` in the loss is the support-conditioned policy `p(. | H)`. Centering covers this conditional distribution without a missing-support tail; it does not recover the full-vocabulary trainer objective.

For filtered runs, `K` is storage capacity: it must be at least the configured and per-request `top_k` and fit every realized support. Cutoff ties can retain more than `top_k` tokens. The support must fit both `K` and SGLang's `--sglang-sampling-mask-max-tokens`; exceeding either fails instead of silently truncating the support. Increasing `K` alone neither expands the sampling support nor improves an already complete support sum.

SGLang's logprob fields differ in both distribution and returned shape:

| SGLang field | Returned values per generated token | Distribution |
| --- | --- | --- |
| `output_token_sampling_logprobs`, `selected` mode | One sampled-token scalar | Post-filter, support-normalized sampler |
| `output_token_sampling_logprobs`, `support` mode | A list aligned with `output_token_sampling_mask` | The complete post-filter distribution; exponentiated entries sum to one |
| `output_token_logprobs`, `output_top_logprobs` with `SGLANG_RETURN_ORIGINAL_LOGPROB=0` | Sampled-token logprob and requested top-candidate entries | `log_softmax(rollout_logits / T)` over the full vocabulary, before top-k/top-p filtering |
| `output_token_logprobs`, `output_top_logprobs` with `SGLANG_RETURN_ORIGINAL_LOGPROB=1` | Same shapes as above | `log_softmax(rollout_logits)`, before temperature and filtering |

Miles-managed rollout workers set `SGLANG_RETURN_ORIGINAL_LOGPROB=0`. A sampled-token scalar or a top-`K` subset does not describe the entire distribution; only the complete support row provides every post-filter probability. Each configuration reads:

| Configuration | Sampling-support replay (`append_sampling_metadata`) | Candidates (`append_rollout_topk_logprobs`) | SGLang source |
| --- | --- | --- | --- |
| Sampling-support replay only (`--rollout-top-logprobs-num 0`) | stores the support; returns the sampled-token log probability | nothing (`K = 0`) | `output_token_sampling_mask`, `output_token_sampling_logprobs` |
| Score centering, unfiltered | not called (no `return_sampling_mask`) | stores the top `K` candidates | `output_top_logprobs` |
| Score centering, filtered | stores the support; picks the sampled-token log probability from its row | stores `[n, K]`: the same support IDs with their row log probabilities | both read the same `output_token_sampling_mask`, `output_token_sampling_logprobs` |

For support replay without reference KL, the trainer gathers only sampled and support logits, normalizes over that support, and saves activations proportional to the candidate count rather than the vocabulary size. Other paths retain full-vocabulary normalization, reducing normalization scalars and selected logits across tensor-parallel ranks. It excludes padded vocabulary entries and computes probabilities in float32 for BF16/FP16 models. `--log-probs-chunk-size` controls the temporary computation size; `--recompute-loss-function` can trade computation for saved activations. No full vocabulary is gathered across ranks.

## Rollout and data contract

Native SGLang generation, the legacy rollout path, and both session-server versions request candidate probabilities at generation time when `--rollout-top-logprobs-num` is positive. Miles derives the logprob mode at startup from `--rollout-top-p` and `--rollout-top-k`. On these training requests, the configured count and derived mode override any client-supplied `top_logprobs`, `top_logprobs_num`, or `sampling_logprobs_mode`. A count of zero disables candidate recording without requesting support-wide probabilities. Each `Sample` carries:

- `rollout_topk_token_ids`: int32 array shaped `[response_length, k]`.
- `rollout_topk_log_probs`: float32 array of the same shape.
- `rollout_log_probs`: the actual sampled-token log probabilities.

Evaluation requests skip this collection and may use independent sampling settings, including greedy decoding. The built-in agentic producer marks evaluation sessions when creating them; custom session clients should create them with `POST /sessions` with JSON body `{"evaluation": true}`.

The Miles SGLang router fork includes the required protocol fields at [commit `df2c790`](https://github.com/radixark/sgl-router-for-miles/commit/df2c790d70179995adc37af4f10bc4e118c30a1e) (#21), or a descendant containing it. Its typed chat requests preserve `input_ids`, `return_meta_info`, `return_sampling_mask`, and `sampling_logprobs_mode`.

- **Unfiltered sessions:** Miles sends `logprobs=true` and `top_logprobs=K`. This fork accepts `top_logprobs` from 0 through 128, so score-centering sessions can use `1 <= K <= 128`; values above 128 are still rejected.
- **Unfiltered native `/generate`:** Miles sends `top_logprobs_num=K`. This field is forwarded independently of the chat validator; SGLang backend limits still apply.
- **Filtered chat and `/generate`:** Miles sends `return_sampling_mask=true` and `sampling_logprobs_mode="support"`, omitting `top_logprobs` and `top_logprobs_num`. The fork forwards these fields on both endpoints. The chat `top_logprobs` cap does not limit support-mode arrays; the support capacities above apply.

Older router builds can reject chat requests above 20 candidates or drop `sampling_logprobs_mode`. For prebuilt images or independently installed routers, run the unfiltered protocol probe below through the router URL to verify the requested candidate count. Check filtered requests separately for complete, aligned support probabilities on both endpoints. A source revision alone does not identify the installed wheel's behavior.

Unused candidate slots and non-trained observation rows contain token ID `-1` and log probability `-inf`. Tool-observation masks, multi-turn merging, retries, trailing-token trimming, and truncation preserve row alignment. Session serialization retains both arrays. For score centering in support mode, training batches share the recorded support IDs when every sample has exactly the same candidate order and prefix padding. A per-row candidate count preserves observation rows and masked generated rows; any mismatch keeps the original arrays for the whole batch. The trainer reconstructs only its context-parallel rows. The source Samples and session payloads still retain both representations.

Custom rollout producers must supply these fields with probabilities from the actual generation call and no repeated non-negative token ID in a row. Conversion always rejects missing candidate IDs, candidate log probabilities, or sampled-token log probabilities, including from custom producers. With `--ci-test`, Miles also validates the complete sample before training, rejecting duplicate candidates, sampling-support mismatches, and disagreeing sampled/candidate probabilities. This full-sample validation is skipped in normal training because sorting and support matching are expensive on long responses. Rescoring old rollouts with newer weights is not a substitute. With the feature disabled, requests and the session wire format remain unchanged.

## Supported configurations and limits

- The shared loss is wired into Megatron and FSDP. Candidate selection supports tensor parallelism, packed (`thd`) and padded (`bshd`) zigzag context parallelism, and packed all-gather context parallelism.
- Sampling requires a fixed positive temperature and `min_p=0` on every call. Filtered sampling, including top-p filtering, automatically uses support mode and requires a positive `top_k`, for example `--rollout-top-p 0.9 --rollout-top-k 64 --rollout-top-logprobs-num 128`. Global filtered rollout settings automatically enable sampling-support replay; per-request overrides are checked when each request is built. See the [sampling-support replay guide](/advanced/sampling-support-replay) for its request and server requirements.
- Filtered sampling requires SGLang with support log probabilities (SGLang PR [#40932](https://github.com/sgl-project/sglang/pull/40932), included in the `sglang-miles` branch by [#41047](https://github.com/sgl-project/sglang/pull/41047), merge commit [`ae04cb14046896b6d453758c5769d639deedd353`](https://github.com/sgl-project/sglang/commit/ae04cb14046896b6d453758c5769d639deedd353) or a descendant containing it). External servers must set `SGLANG_RETURN_ORIGINAL_LOGPROB=0` like Miles-managed workers. OpenAI session responses must expose the SGLang fields above in `choices[0].meta_info`; a generic OpenAI-compatible server without that metadata is insufficient.
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
MILES_LIVE_SCORE_CENTERING_TOP_K=128 \
python -m pytest --confcutdir=tests/manual tests/manual/test_score_centering_live.py
```

Set `SGLANG_RETURN_ORIGINAL_LOGPROB=0` on the server before starting it. The probe checks unfiltered native and OpenAI response metadata using the production candidate collector and validator, and checks temperature scaling at 0.7, 1.0 and 1.3. `MILES_LIVE_SCORE_CENTERING_TOP_K` sets the recorded candidate count (default 128), not a sampling filter. The probe does not exercise filtered support-mode requests.

It warms the shared prompt first so that cached and uncached prefills do not confound the temperature comparison. Set `MILES_LIVE_SCORE_CENTERING_ARTIFACT_DIR` to keep the raw responses.
