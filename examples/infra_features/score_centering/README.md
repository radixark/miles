# Score Centering

Correct off-policy score drift using the sampler's top-k probabilities, with optional truncated or masked importance weights.

This implements [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807), including the efficient top-k approximation in Appendix A.

## Enable it

Add these arguments to an existing text-only GRPO training recipe:

```bash
--loss-type score_centering \
--advantage-estimator grpo \
--score-centering-top-k 128 \
--score-centering-is none \
--rollout-temperature 1.0 \
--rollout-top-p 1.0 \
--rollout-top-k -1 \
--use-rollout-logprobs \
--disable-grpo-std-normalization \
--calculate-per-token-loss
```

Keep reward mean subtraction enabled. Disabling standard-deviation normalization gives the paper's group-centered rewards. This is a separate REINFORCE-style loss: PPO clipping parameters do not apply. Existing batch size and update scheduling still control how many updates consume a rollout batch; choose them explicitly when reproducing an experiment.

Choose `--score-centering-is tis` for weights clipped at `--score-centering-tis-clip` (default 2). Choose `mis` to retain ratios in `[--score-centering-mis-low, --score-centering-mis-high]` (defaults 0.5 and 5), setting other weights to zero. These weights are centered together with the score. Use these options instead of `--use-tis` or a custom TIS function.

The existing entropy and reference-KL loss options remain available. They are separate regularizers; the score-centering identity applies to the policy-gradient term.

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

The trainer computes only the requested log probabilities from each vocabulary shard, reducing normalization scalars and selected logits across tensor-parallel ranks. It excludes padded vocabulary entries and computes probabilities in float32 for BF16/FP16 models. `--log-probs-chunk-size` controls the temporary computation size; `--recompute-loss-function` can trade computation for saved activations. No full vocabulary is gathered across ranks.

## Rollout and data contract

Native SGLang generation, the legacy rollout path, and both session-server versions request candidate probabilities at generation time. Each `Sample` carries:

- `rollout_topk_token_ids`: int32 array shaped `[response_length, k]`.
- `rollout_topk_log_probs`: float32 array of the same shape.
- `rollout_log_probs`: the actual sampled-token log probabilities.

Evaluation requests skip this collection and may use independent sampling settings, including greedy decoding. The built-in agentic producer marks evaluation sessions when creating them; custom session clients should create them with `POST /sessions?evaluation=true`.

Session rollouts with more than 20 candidates require `--use-miles-router`. The SGLang Rust router caps OpenAI `top_logprobs` at 20, while MilesRouter forwards the larger request to SGLang unchanged. This includes the default `k=128` with either session-server version. To use the SGLang router for session rollouts, set `--score-centering-top-k 20` or less. Native `/generate` rollouts support either router.

Unused candidate slots and non-trained observation rows contain token ID `-1` and log probability `-inf`. Tool-observation masks, multi-turn merging, retries, trailing-token trimming, and truncation preserve row alignment. Session serialization and data-parallel sharding retain both arrays. Candidates stay on CPU until the trainer selects its context-parallel rows.

Custom rollout producers must supply these fields with probabilities from the actual generation call. Missing candidates, duplicates, malformed padding, or disagreeing sampled/candidate probabilities fail validation. Rescoring old rollouts with newer weights is not a substitute. With the feature disabled, requests and the session wire format remain unchanged.

## Supported configurations and limits

- The shared loss is wired into Megatron and FSDP. Candidate selection supports tensor parallelism, packed (`thd`) and padded (`bshd`) zigzag context parallelism, and packed all-gather context parallelism.
- Sampling requires a fixed positive temperature, `top_p=1`, `top_k=-1`, and `min_p=0` on every call. Native candidate logprobs are computed before top-p/top-k filtering, so filtered probabilities cannot be used as `q` here.
- Miles-managed rollout workers set `SGLANG_RETURN_ORIGINAL_LOGPROB=0`. External servers must do the same and return native `output_top_logprobs` alongside sampled-token logprobs. OpenAI session responses must expose them in `choices[0].meta_info`; a generic OpenAI-compatible server without that metadata is insufficient.
- Constrained/custom sampling, speculative decoding, true-on-policy mode, OPD, multi-LoRA/Tinker losses, sequence masking, custom policy-loss reducers, and logprob recomputation via prefill are rejected. Multimodal token expansion is not supported. The initial advantage estimator is GRPO.
- Retaining `k=128` uses about 1 KiB per response position for the two arrays, before transport overhead. Larger `k` improves the tail approximation at additional storage and compute cost.

## Metrics and verification

Training logs include `sc_correction`, `sc_train_head_mass`, `sc_rollout_head_mass`, `sc_tail_ratio`, `sc_importance_weight`, and `train_rollout_logprob_abs_diff`, under the usual `train/` namespace. Small head mass means more of the distribution is approximated by the tail model. Large tail ratios indicate a substantial mismatch in remaining mass.

The numerical tests compare gradients against an independent dense-distribution oracle for all three weighting modes, including sampled tokens outside the head, constant rewards, and tiny tails. Pipeline tests exercise real native/session producers, serialization, data-parallel splitting, masks, advantage computation, regularization, and checkpointed loss scaling. Run the focused tests in the repository's test environment:

```bash
python -m pytest tests/fast/backends/training_utils/test_score_centering.py \
    tests/fast/backends/training_utils/test_score_centering_pipeline.py
MILES_TEST_DISTRIBUTED=1 python -m pytest \
    tests/fast/backends/training_utils/test_score_centering_distributed.py
MILES_TEST_CUDA_DISTRIBUTED=1 python -m pytest \
    tests/fast/backends/training_utils/test_score_centering_distributed.py
```

The distributed tests use four CPU/Gloo processes or four CUDA/NCCL processes (TP=2, CP=2), all three context layouts and weighting modes, plus BF16 selected-probability gradients. The independent dense-gradient oracle also runs on CUDA when available. These are correctness tests, not a reproduction of the paper's GPU training results.

For a real SGLang server, also run the opt-in protocol probe:

```bash
MILES_LIVE_SCORE_CENTERING_ENDPOINT=http://127.0.0.1:30000 \
MILES_LIVE_SCORE_CENTERING_MODEL=/path/to/model \
MILES_LIVE_SCORE_CENTERING_SERVED_MODEL=your-served-model \
python -m pytest --confcutdir=tests/manual tests/manual/test_score_centering_live.py
```

Set `SGLANG_RETURN_ORIGINAL_LOGPROB=0` on the server before starting it. The probe checks native and OpenAI response metadata using the production candidate collector and validator, and checks temperature scaling at 0.7, 1.0 and 1.3. It warms the shared prompt first so that cached and uncached prefills do not confound the temperature comparison. Set `MILES_LIVE_SCORE_CENTERING_ARTIFACT_DIR` to keep the raw responses.

### Fixed-rollout ablation

Record behavior probabilities with the rollout-only debug mode, then replay the same
recordings with the training-only debug mode. For a matched control, add
`--disable-score-centering-correction`: only the correction is disabled; candidate
computation, importance weighting, regularization, and probability diagnostics remain
the same. This control is a weighted policy-gradient ablation, not the PPO loss.
Never replace recorded behavior probabilities with probabilities from the updated actor.
Evaluate learned checkpoints separately: the rewards of recorded responses are fixed.

### Frozen rollout collection

With `--debug-rollout-only`, collection skips training-batch postprocessing,
partitioning, and trainer dispatch. Empty waves are allowed and the data cursor
is saved after every wave. This does not relax the training data requirements
when the saved corpus is later replayed.
