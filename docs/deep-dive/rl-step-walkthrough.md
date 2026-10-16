---
title: One RL Step, End to End
description: A code-grounded walk through one synchronous train.py step - actors, logprobs, advantages, loss normalization, weight sync.
---
This page follows **one training step of `train.py`** from the moment the driver asks for a
rollout until SGLang is serving the new weights. It is written for readers who know RL
(PPO, GRPO, importance sampling) but have not yet traced how Ray, SGLang and Megatron are
wired together inside Miles. It answers, with `path:line` references:

- which processes exist and which GPUs they sit on;
- who computes each logprob (rollout, old-policy, reference) and when a forward pass is skipped;
- where the train/inference mismatch comes from and which flags address it;
- exactly how advantages and the loss are computed and normalized across micro-batches, DP and CP;
- how the weights get back into SGLang.

Read these pages alongside it rather than instead of it:
[Architecture Overview](/developer/architecture), [Core Concepts](/user-guide/concepts),
[Training Backends](/user-guide/training-backend), [Argument Groups](/user-guide/argument-groups),
[True On-Policy](/examples/infra-features/true-on-policy),
[Rollout Correction Methods](/examples/infra-features/train-infer-mismatch-helper). Async training
is covered separately in [Async and Off-Policy](/deep-dive/async-and-off-policy).

Line numbers were checked against commit `edf2c7ae3e` (rebased onto upstream `e30d9a8ec6`,
October 2026). They drift; search by function name when they do.

**Running example.** `scripts/run_qwen3_4b.py` (non-debug mode): one node, 8 GPUs, `--colocate`,
`--rollout-batch-size 32 --n-samples-per-prompt 8 --global-batch-size 256` (so exactly one
optimizer step per rollout), `--advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28`,
`--rm-type math`, `--use-dynamic-batch-size --max-tokens-per-gpu ...`, Megatron backend.

---

## 1. The cast

Unlike open-instruct, the driver in Miles is not a thin launcher: `train.py` itself sequences
every phase of the step and awaits each one. What it awaits are worker handles, not local
objects. `init_orchestration_script` (`miles/utils/orchestration_utils.py:16`) starts a
`RayWorkerManager` that launches every pool `compute_specs` declares
(`miles/ray/specs/entrypoint.py:8`), and a call such as `await rollout_executor.get(rollout_id)`
is a Ray actor call, or an HTTP RPC with `--worker-comm-backend rpc` (the default under
`--cluster-backend kubernetes`). The table describes the default Ray backend.

| Component | Kind / count | Where it runs | Owns |
|---|---|---|---|
| `train()` in `train.py:25` | driver coroutine (`asyncio.run(with_disposer(train, args))`) | Ray driver process, CPU | the loop; handles to the two controllers and the `RolloutExecutor`; the `EvalDispatcher` |
| `InferenceController` (`miles/ray/rollout/inference_controller.py:46`) | 1 Ray actor, 1 CPU, 0 GPUs (`spec_inference_controller`, `miles/ray/specs/inference.py:45`) | any node, or the head with `--pin-rollout-manager-to-head` | the SGLang server cells: health, offload/onload tags, generation pause, the engine set handed to weight sync (`start_update_weights`, `:233`) |
| `RolloutExecutor` (`miles/ray/rollout/rollout_executor.py:65`) | 1 Ray actor, 1 CPU, 0 GPUs (`spec_rollout_executor`, `miles/ray/specs/rollout.py:15`) | any node, or the head with `--pin-rollout-manager-to-head` | the **data source**, the rollout function (default `InferenceRolloutFn`), reward calls, Sample to train-dict conversion, DP sharding |
| Data source (`--data-source-path`, default `RolloutDataSource`, `miles/rollout/data_source.py:40`) | Python object inside `RolloutExecutor` | same process as the executor | prompt dataset cursor/epoch. `--partial-rollout` switches the default to `LegacyRolloutDataSourceWithBuffer` (`:142`, `arguments.py:4377`), which adds a buffer for partial-rollout groups |
| Router (`specs_router`, `miles/ray/specs/inference.py:149`) | 1 process per model, 0 GPUs | Ray node (head if pinned) | HTTP load balancing over engines. **Default is SGLang's `sglang_router.launch_router`**; `--use-miles-router` swaps in the FastAPI `MilesRouter` (`miles/router/router.py:36`) |
| SGLang engines (`specs_inference_engine`, `miles/ray/specs/inference.py:269`) | one `sglang.launch_server` **subprocess** per engine (per node of a multi-node engine), spawned by a `CommandActor` (`miles/utils/workers/command_actor.py:13`) | `rollout` placement-group slots, `--rollout-num-gpus-per-engine` GPUs each | KV cache, sampling, `output_token_logprobs`; receiving weights |
| Session server (`spec_session_server`, `inference.py:219`) | 0 or `--session-server-workers` (default 32) processes | any node; the head with `--pin-rollout-manager-to-head` or `--session-server-external-host` | only with `--use-session-server` (`v1` linear, `v2` tree-serving; agentic/OpenAI-style multi-turn); not on the default path |
| Trainer ranks (`MegatronTrainRayActor` / `FSDPTrainRayActor` / `TorchtitanTrainRayActor`, `miles/ray/specs/train.py:42`) | 1 Ray actor per training GPU | `actor` placement-group slots | the policy, optimizer, **reference / teacher / old-actor weights (as host copies)**, the weight updater |
| `TrainerController` (`miles/ray/train/group.py:52`) | 1 Ray actor per trainer, 1 CPU, 0 GPUs (`specs/train.py:115`) | any node | fans calls out to trainer "cells" (`TrainerCell`, `miles/ray/train/cell.py:29`); with `--indep-dp` (turned on by `--ft-components train`) there is one cell per DP replica, otherwise exactly one (`compute_trainer_num_cells`, `specs/train.py:181`) |
| Critic (`role="critic"`) | second `TrainerController`, same GPUs | same bundles as actor (`placement_group.py:160-161`) | only when `--advantage-estimator ppo` (`args.use_critic`, `arguments.py:3747`) |

**Reward model hub.** There is no reward actor. `async_rm` (`miles/rollout/rm_hub/__init__.py:42`)
runs as an asyncio call inside the `RolloutExecutor` right after each sample is generated.
It dispatches on `sample.reward_spec`, then `sample.metadata["rm_type"]`, then `--rm-type`
(`_resolve_reward_config`, `:31`): `math`, `dapo`, `deepscaler`, `f1`, `gpqa`, `ifbench`,
`remote_rm` (HTTP POST to `--rm-url`), `boxed_*` prefix to extract `\boxed{}` first, or
`--custom-rm-path` to bypass all of them.

**Reference model / old actor.** There is no separate ref actor on either backend.

- Megatron: the `TrainerController` spec gets `with_ref = kl_coef != 0 or use_kl_loss`
  (`miles/ray/specs/train.py:59`). Each trainer rank loads `--ref-load` into the *same*
  Megatron model and snapshots it into a pinned host buffer (`TensorBackuper`,
  `miles/backends/megatron_utils/actor.py:242`; `load_other_checkpoint`, `:975`). Before the ref
  forward, the ref weights are copied into the live parameters (`_switch_model("ref")`,
  `actor.py:539`, `:699`), then the actor weights are copied back. Teacher (`--use-opd --opd-type
  megatron`) and `--keep-old-actor` work the same way (`_load_auxiliary_checkpoints`,
  `actor.py:421-437`).
- FSDP: a second FSDP2 model with `CPUOffloadPolicy`, loaded to GPU on demand
  (`_create_ref_model`, `miles/backends/fsdp_utils/actor.py:358`; `_ref_context`, `:293`).

### 1.1 GPU layout

`_get_placement_group_layout` (`miles/ray/placement_group.py:122`) sizes **one** placement
group of `{"GPU": 1, "CPU": 1}` bundles, sorted by node IP then GPU id (`sort_key`, `:54`):

```text
disaggregated:  [ actor GPUs 0..A-1 | rollout GPUs A..A+R-1 | eval GPUs ]   rollout_offset = A
--colocate:     [ max(A, R) GPUs shared by both ]                           rollout_offset = 0
```

In colocated mode the same bundle hosts a trainer worker requesting 0.4 GPU
(`_NUM_GPUS_PER_TRAINER_WORKER`, `specs/train.py:48`) and an engine worker requesting 0.2 GPU
(`specs/inference.py:342`); the fractions only make Ray co-schedule them. Memory sharing is
done by `torch_memory_saver` (trainer, `specs/train.py:259-279`) and SGLang's
release/resume-memory tags (engines), driven by `--offload-train`, `--offload-rollout` and
`--offload-rollout-level` (defaults both on under `--colocate`). `train_async.py` accepts
`--colocate` only together with `--fully-async` (`train_async.py:24`).

---

## 2. One training step, pseudocode

This is `train.py:103-171` with every callee inlined, for the default (non-legacy) rollout
path, Megatron backend, GRPO. Each `await` on the driver is a call into another actor.

```python
# ---------------- DRIVER: train.py:103 ----------------
for rollout_id in range(start_rollout_id, num_rollout):
    await inference_controller.prepare_rollout(rollout_id)          # resume health checks
    pack = await rollout_executor.get(rollout_id)                   # blocks until data is sharded

# ---------------- ROLLOUT EXECUTOR (CPU actor): rollout_executor.py:164 ----------------
def get(rollout_id):
    samples = InferenceRolloutFn._call_train(...)                   # inference_rollout_common.py:225
    #   generate_rollout_async (inference_rollout_train.py:99):
    #   while len(kept_groups) < rollout_batch_size:
    #       groups = data_source.get_samples(over_sampling_batch_size)   # n copies of each prompt
    #       for group: asyncio task -> generate_and_rm_group
    #            each sample: POST router/generate {input_ids, sampling_params, return_logprob=True}
    #            -> tokens += output ids; rollout_log_probs += SGLang logprobs; loss_mask += 1s
    #            -> sample.reward = await async_rm(args, sample)
    #       as groups finish: drop if any ABORTED / missing reward / dynamic filter says no
    #   abort remaining in-flight requests (discarded unless --partial-rollout)
    data, metadata = postprocess_rollout_data(...)                  # flatten, trim to multiple of GBS
    train_data = convert_samples_to_train_data(...)                 # rewards -> group-normalized
    return split_train_data_by_dp(...)                              # one object-store ref per DP rank

# ---------------- DRIVER, colocated handoff: train.py:107-118 ----------------
    if offload_rollout: await inference_controller.offload(tags=[cuda_graph, kv_cache, weights])

# -------- TRAINER: TrainerController.train (group.py:169) -> each alive cell -> --------
# -------- MegatronTrainRayActor.train (megatron_utils/actor.py:579) -> _train_actor :661 --------
    await actor_model.train(rollout_id, pack)
def train_actor(rollout_data):
    wake_up()                                                       # torch_memory_saver.resume
    iters, n_mb = get_data_iterator(...)                            # micro-batch schedule
    if with_ref:  switch_model("ref");  rollout_data |= forward_only(ref)      # ref_log_probs
    switch_model("old_actor" if keep_old_actor else "actor")
    if not use_rollout_logprobs or get_mismatch_metrics:
        rollout_data |= forward_only(actor)                         # log_probs  (the "old" policy)
    compute_advantages_and_returns(args, rollout_data)              # loss/objective.py:28
    for step in range(num_steps_per_rollout):                       # model.py train()
        zero_grad()
        for mb in micro_batches[step]:                              # Megatron pipeline schedule
            logits = model(mb.tokens, packed_seq_params)            # grad on
            loss = loss_function(...)                               # PPO-clip (+TIS, +KL, -entropy)
            loss.backward()
        optimizer.step();  lr_scheduler.step(increment=num_rollouts)
    weights_backuper.backup("actor")                                # host copy of new weights

# ---------------- DRIVER, handoff back + weight sync: train.py:141-160 ----------------
    await actor_model.offload()                                     # trainer -> CPU
    await inference_controller.onload_weights()
    await update_weights(args, actor_model, rollout_executor, inference_controller)  # CUDA IPC when colocated
    await inference_controller.onload_kv()
    if eval_due: await eval_dispatcher.dispatch(rollout_id)
```

The step is strictly sequential: generation for step `k+1` does not start until weights from
step `k` are in SGLang. This is what makes `train.py` on-policy up to numerics.

### 2.1 Who computes which logprob

Four logprob evaluations can exist per response token:

| Name in `rollout_data` | Who | When | Used for |
|---|---|---|---|
| `rollout_log_probs` | SGLang, during sampling (`meta_info.output_token_logprobs`, `generate_endpoint_utils.py:103-124`; under sampling-support replay, `meta_info.output_token_sampling_logprobs` instead, `:109-115`) | always (`return_logprob: True`, `:78`) | PPO denominator under `--use-rollout-logprobs`; TIS/MIS; the `train_rollout_logprob_abs_diff` / `train_rollout_kl` metrics |
| `log_probs` | trainer, **forward-only** with the pre-update actor (`actor.py:720-737`) | unless `--use-rollout-logprobs` without `--get-mismatch-metrics`, or `--skip-actor-forward-only` | default PPO denominator ("old policy"); KL-in-reward |
| `ref_log_probs` | trainer, forward-only with ref weights swapped in (`actor.py:697-707`) | only if `kl_coef != 0 or use_kl_loss` | KL loss / KL-in-reward |
| current `log_probs` in the loss | trainer, with grad, inside `policy_loss_function` (`losses.py:118`) | every micro-batch | the numerator of the ratio |

Both trainer forwards divide logits by `--rollout-temperature` before the softmax
(`logit_processors.py:282`, `_upcast_chunk_to_fp32`, `math_utils.py:961-967`), except under
`--true-on-policy-mode` where the scaling is applied earlier in `_iter_response_chunks`
(`logit_processors.py:97-99`).

SGLang's `output_token_logprobs` are on the same scale: they are **post-temperature** and
cover the full vocabulary, **before** any top-p/top-k/min-p truncation. This was verified
against the SGLang 0.5.21 release (production images run 0.5.22.dev46, not checked line by
line), with paths relative to `sglang/python/sglang/srt/`. Miles sends
`temperature=--rollout-temperature` with every training request
(`inference_rollout_common.py:46`, `sglang_rollout.py:95`), and the sampler does the following:

- Penalties, logit bias and grammar masks are applied to the logits first
  (`model_executor/model_runner.py:1957-1983`), then any custom logit processor
  (`layers/sampler.py:132-139`).
- The logits are divided in place by the per-request temperature (`layers/sampler.py:248`)
  and softmaxed (`:266`). The returned log-prob is `log(probs)` of that full softmax
  (`:275-280`), or `log_softmax` of the scaled logits under deterministic or true-on-policy
  inference (`:212-222`, `:233-245`, `:255-263`). Top-k/top-p filtering happens later, inside
  the sampling kernel, on its own copy (`_sample_from_probs`, `:324-465`), so it never touches
  the returned value.
- `SGLANG_RETURN_ORIGINAL_LOGPROB=1` switches the returned log-prob to the *pre-temperature*
  `log_softmax` (`layers/sampler.py:74`, `:208-210`, `:283-285`; `environ.py:1323`). Miles
  forces it to `0` on every engine it launches, overriding an exported value
  (`compute_inference_engine_env_vars`, `miles/ray/specs/inference.py:422-424`). Only engines
  Miles does not launch (`--rollout-external-engine-addrs`, a custom
  `--custom-inference-engine-provider-path`) can still run with it set; there
  `train_rollout_kl` becomes a temperature artifact whenever `--rollout-temperature != 1`.
- Greedy requests (`temperature < 1e-6`, which SGLang turns into `top_k=1` at `T=1`,
  `sampling/sampling_params.py:217-220`) return the raw `log_softmax`
  (`layers/sampler.py:185-200`). This only affects eval, which does not feed the mismatch
  metrics.
- Speculative decoding's accepted-token log-probs are also temperature-scaled
  (`layers/logprob_processor.py:386-432`).

So at any `--rollout-temperature` the mismatch metrics compare like with like. A large
`train_rollout_kl` after a temperature change is real mismatch, not a scaling artifact,
unless an external engine runs with `SGLANG_RETURN_ORIGINAL_LOGPROB=1` or
`--recompute-logprobs-via-prefill` is on (see item 3 below). With top-p/top-k, see item 2 below.

Notes:

- The old-policy pass runs **once for the whole rollout, before the first optimizer step**.
  With `--num-steps-per-rollout > 1` the later steps still clip against the pre-update
  actor, which is proper PPO.
- With `--use-rollout-logprobs` (and no `--get-mismatch-metrics`) the Megatron backend skips
  the old-policy forward entirely (`actor.py:721-723`). The torch-native backends (FSDP,
  torchtitan) always run it (`training_utils/torch_native/actor.py:192-193`).
- `--skip-actor-forward-only` (Megatron only, one optimizer step) reuses the detached
  training-pass logprobs as "old", so without `--use-rollout-logprobs` the ratio is exactly 1
  (`losses.py:131-132`).

### 2.2 Where the train/inference mismatch comes from

In `train.py` the rollout and the old-policy forward use **the same weights**: SGLang was
updated at the end of the previous step, and the trainer has not stepped since. So with the
default flags `ratio = exp(logp_theta - logp_old)` is 1 on the first micro-batch of the first
step, the clip never fires, and the rollout-vs-trainer gap is invisible to the loss. The gap
still exists, from:

1. **Kernels and numerics.** SGLang (paged attention, CUDA graphs, fused MoE, its own GEMMs,
   bf16 sampling) vs. Megatron/TE (or HF + FSDP2) with TP/CP/packing. MoE routing can differ
   token by token, which is what `--use-rollout-routing-replay` (R3) fixes by replaying the
   rollout's expert choices (`rollout_routed_experts`).
2. **Sampling-support truncation.** With `--rollout-top-p < 1` or `--rollout-top-k > 0` the
   rollout samples from a truncated distribution. Miles then turns on sampling-support replay
   (`args.use_sampling_support_replay`, `arguments.py:3500`). It asks SGLang for
   `return_sampling_mask` (`generate_endpoint_utils.py:73-83`), and the engine returns the
   realized support plus `output_token_sampling_logprobs`. Those are
   `log(p_T(token) / sum of p_T over the kept support)`, the log-prob renormalized over the
   truncated, temperature-scaled distribution (`sglang/python/sglang/srt/layers/sampler.py:500-592`,
   `:530-535`). With `--rollout-top-logprobs-num` set, Miles requests
   `sampling_logprobs_mode="support"` instead and gets a row per support token, from which it
   picks the sampled one (`rollout_topk_logprobs.py:29-33`, `sampling_mask.py:117-125`).
   Miles stores those as `rollout_log_probs` (`sampling_mask.py:92-133`), and the trainer masks its
   logits to the same support before the log-softmax (`math_utils.py:313-325`,
   `rollout_sampling_mask_*` fields), so both sides are truncated and renormalized. The request
   validator rejects penalties, logit bias and custom logit processors in this mode, because
   the trainer cannot replay them (`sampling_mask.py:61-74`). Entropy is still computed on the
   full, unmasked softmax (`math_utils.py:999-1003`).
3. **Prefill vs. decode paths in SGLang itself** for some models;
   `--recompute-logprobs-via-prefill` (requires `--true-on-policy-mode`) re-scores responses
   with prefill after generation (`inference_rollout_train.py:204`). That re-score sends
   `temperature: 0` (`prefill_logprobs.py:33-46`) and reads `input_token_logprobs` (`:91-95`).
   SGLang's input-logprob path applies no temperature at all: it is a plain `log_softmax` of the
   raw logits (`sglang/python/sglang/srt/layers/logprob_processor.py:730-736`, or the
   equivalent fast row normalizer at `:708-728`), and `LogitsMetadata.temperature`
   (`layers/logits_processor.py:314`) is never read. The
   true-on-policy trainer still divides by `--rollout-temperature` (`logit_processors.py:97-99`).
   By code reading, the re-scored `rollout_log_probs` therefore match the trainer only at
   `--rollout-temperature 1`. The shipped true-on-policy recipes all use that temperature
   (`examples/infra_features/true_on_policy/run_simple.py:37`,
   `tests/e2e/fsdp/test_qwen3_4B_fsdp_true_on_policy.py:38`). The other temperatures have not
   been checked by a run.
4. **Staleness** only in async (see [Async and Off-Policy](/deep-dive/async-and-off-policy)) and with
   `--partial-rollout`, where resumed groups carry tokens from older weights.

Knobs, from least to most invasive:

| Knob | Effect |
|---|---|
| `--get-mismatch-metrics` + `--custom-tis-function-path` | logs the MIS metrics; with `--use-rollout-logprobs` it also brings back the trainer forward so the mismatch is measured against an independently scored policy. (`train_rollout_logprob_abs_diff` and `train_rollout_kl`, low-var k3 of rollout vs. trainer, are logged whenever rollout logprobs exist, `losses.py:346-368`.) Note it still routes `pg_loss` through the custom function (`losses.py:234-265`); the shipped `mis.py` passes it through unchanged when `--use-tis` is off, but a custom function must do the same |
| `--use-rollout-logprobs` | PPO denominator = SGLang logprobs, so the ratio sees the mismatch and clipping acts on it ("bypass" mode). Incompatible with `--use-tis` (`arguments.py:3681-3682`) |
| `--use-tis [--tis-clip 2.0 --tis-clip-low 0]` | decoupled PPO: multiply the clipped per-token loss by `clamp(exp(logp_old_trainer - logp_rollout), low, high)` (`corrections.py:7`) |
| `--custom-tis-function-path ...mis.compute_mis_weights_with_cp` | token/sequence/geometric IS, rejection sampling, veto (config in `mis.yaml`) |
| `--use-opsm --opsm-delta` | drop whole sequences with negative advantage whose mean `logp_old - logp_theta` exceeds delta (`math_utils.py:183`) |
| `--true-on-policy-mode` | bitwise-aligned kernels on both sides, so the ratio is exactly 1 and `train_rollout_logprob_abs_diff` is 0; see [True On-Policy](/examples/infra-features/true-on-policy) and `miles/true_on_policy/` |

---

## 3. The rollout, in detail

### 3.1 Prompts to samples

`RolloutDataSource.get_samples(n)` (`data_source.py:83`) returns `n` **groups**; each group is
`n_samples_per_prompt` deep copies of one dataset row, with a shared `group_index` and a unique,
monotonically increasing `index`. At an epoch boundary it wraps and reshuffles (with
`--rollout-shuffle`). Under `--partial-rollout` the default becomes
`LegacyRolloutDataSourceWithBuffer`, which first drains its buffer (partial-rollout groups put
back after an abort) through `--buffer-filter-path` (default `pop_first`).

`generate_rollout_async` (`inference_rollout_train.py:99`) keeps submitting batches of
`--over-sampling-batch-size` groups (default `rollout_batch_size`, `arguments.py:4060-4061`)
until `rollout_batch_size` groups have passed the filters. Concurrency is bounded by a
semaphore of `sglang_server_concurrency * rollout_num_gpus / rollout_num_gpus_per_engine`
requests (`inference_rollout_common.py:41-43`).

For each sample, `generate_and_rm` (`inference_rollout_common.py:60`) calls the generate
function (`--custom-generate-function-path`, default `generate_hub/single_turn.py:16`), then
the reward (unless `--group-rm`, which scores the whole group at once, `:185-186`). Under
`--sglang-enable-deterministic-inference` each sibling gets `sampling_seed = rollout_seed + idx`
(`:164-165`); otherwise the router and SGLang pick seeds.

### 3.2 Filters and dynamic sampling

As each group completes, `apply_preput_filters` (`filter_hub/common_filters.py:70`) rejects
it if any sample is `ABORTED`, if any reward is `None`, or if `--dynamic-sampling-filter-path`
says no. The DAPO-style filter is
`miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter` (`:95`, drops a
group whose reward std is `<= 1e-8`). Drops are logged as
`rollout/dynamic_filter/drop_<reason>`.

Once enough groups are kept, the remainder are aborted (`abort`, `inference_rollout_train.py:27`).
**Without `--partial-rollout` the aborted groups and the filtered groups are discarded**, and
their prompts are consumed from the epoch. After that:

- `--rollout-sample-filter-path fn(args, data)` can set `sample.remove_sample = True`, which
  zeroes that sample's loss mask (`train_data_conversion.py:111-112`) but keeps it in the group
  statistics used for reward normalization;
- `--rollout-all-samples-process-path` sees every group, including filtered ones.

### 3.3 Multi-turn and loss masks

`generate_hub/multi_turn.py:26` is the reference tool loop (`--generate-max-turns`,
`--generate-tool-specs-path`, `--generate-tool-call-parser`,
`--generate-execute-tool-function-path`). Each model turn appends SGLang's token ids with
`loss_mask += [1]*n` and its logprobs (`generate_endpoint_utils.py:118-136`). Each tool
response is tokenized with a dummy-prefix diff trick (`tool_call_utils.py:80`, it asserts the
chat template is append-only) and appended with `loss_mask += [0]*n` and
`rollout_log_probs += [0.0]*n` (`tool_call_utils.py:60-69`). So in the trainer:

```text
tokens        = prompt | model turn 1 | tool obs 1 | model turn 2 | ...
response part =          |<--------------- response_length --------------->|
loss_mask     =          1 1 1 1 1 1   0 0 0 0 0   1 1 1 1 1 1
```

`--partial-rollout --mask-offpolicy-in-partial-rollout` would zero the mask of tokens produced
before an abort, but the class-based rollout path rejects that combination
(`arguments.py:87-91`); it needs `MILES_USE_LEGACY_ROLLOUT_V1=1`.

### 3.4 Samples to the train dict

`postprocess_rollout_data` (`rollout_data_conversion.py:10`) flattens `list[list[Sample]]` and
trims the count to a multiple of `--global-batch-size` (or computes a one-step dynamic GBS
with `--use-dynamic-global-batch-size`); compact rollouts and `--disable-rollout-trim-samples`
skip the trim (`:21-39`). Then `convert_samples_to_train_data`
(`train_data_conversion.py:61`) builds the dict (section 6) and normalizes rewards.

**GRPO reward normalization happens here, on the rollout side, before the DP split**
(`_normalize_rewards_by_rollout`, `train_data_conversion.py:294`). For estimators `grpo`,
`gspo`, `reinforce_plus_plus_baseline` and unless `--disable-rewards-normalization`:

```text
for each prompt group g (by Sample.group_index):
    r_hat_i = r_i - mean_g(r)
    if estimator in {grpo, gspo} and grpo_std_normalization and |g| > 1 and std_g(r) > 0:
        r_hat_i = r_hat_i / (std_g(r) + 1e-6)          # torch .std(): unbiased, N-1
```

`--disable-grpo-std-normalization` is Dr. GRPO. With `--n-samples-per-prompt 1` std
normalization is turned off automatically (`arguments.py:4053-4054`). A zero-variance group yields
all-zero advantages; it is still trained on (costing compute) unless the dynamic filter drops it.
Compact / sub-agent rollouts that return several `Sample`s for one trajectory must share
`rollout_id` and a single reward, and are normalized as one unit.

### 3.5 DP sharding and the micro-batch schedule

`split_train_data_by_dp` (`train_data_conversion.py:361`) produces one object-store entry per
data-parallel rank (`dp_size` = Megatron intra-DP size, excluding CP). Two paths:

- **Rollout-side schedule** (`split_train_data_by_dp_scheduled_raw`, `:389`): used when the
  backend advertised `dp_size, cp_size, vpp_size, microbatch_group_size_per_vp_stage`
  (Megatron without `--indep-dp`), the run is neither multi-LoRA nor multimodal, and there are
  at least GBS rollouts (`can_schedule_on_rollout_side`, `:375`). `build_dp_schedule`
  (`miles/utils/dp_schedule.py:31`) keeps all samples of one rollout in the same step and
  precomputes `micro_batch_indices`, `num_microbatches` and `num_rollouts` per step.
- **Legacy split** (`:436`): round-robin, or Karmarkar-Karp token balancing with
  `--balance-data`; the trainer builds the micro-batch schedule itself in `get_data_iterator`
  (`training_utils/data/rollout.py:455`). The torch-native backends (FSDP, torchtitan) always
  take this path.

With `--use-dynamic-batch-size`, samples are packed into as few micro-batches as fit
`--max-tokens-per-gpu * cp_size` tokens, the count is all-reduced to the DP max, and samples are
balanced across micro-batches (`data/rollout.py:532-561`); `--micro-batch-size` is then ignored.
Inside a micro-batch Megatron uses `qkv_format=thd`: sequences are concatenated with
`cu_seqlens` and padded to a multiple of `tp_size * --data-pad-size-multiplier`
(`get_batch`, `data/rollout.py:168`). This is Miles' sequence packing; there is no separate
`pack_length`.

The trainer rank fetches only its own shard (`process_rollout_data`, `miles/utils/data.py:293`);
with `--delay-split-train-data-by-dp` (turned on by `--ft-components train`) every rank fetches the
whole batch and splits locally.

---

## 4. The trainer step

### 4.1 Advantages (`compute_advantages_and_returns`, `loss/objective.py:28`)

Only the last pipeline stage computes them. Inputs: `rewards` (already group-normalized where
applicable), the old-policy logprobs (`rollout_log_probs` under `--use-rollout-logprobs`,
otherwise `log_probs`, `objective.py:56`), `ref_log_probs`, and `values` for PPO. A per-token
`kl = compute_approx_kl(logp_old, logp_ref, --kl-loss-type)` is computed only when
`--kl-coef != 0` (`objective.py:86-98`). Then `compute_advantages` (`loss/hub/advantages.py:16`):

```text
grpo / gspo                     A_t = r_hat                         (broadcast over response tokens)
reinforce_plus_plus             G_t = sum_{s >= t} gamma^(s-t) * ( -kl_coef * kl_s + r * [s == last masked token] )
                                A_t = G_t                           (r is the raw reward)
reinforce_plus_plus_baseline    A_t = r_hat - kl_coef * kl_t        (r_hat = group-mean-subtracted, no std)
ppo                             token rewards = -kl_coef * kl_t, terminal reward r; GAE(gamma, lambd) on critic values
```

References: `get_grpo_returns` (`math_utils.py:462`), `get_reinforce_plus_plus_returns`
(`:472`), `get_reinforce_plus_plus_baseline_advantages` (`:536`), GAE (`:566`).

Then, in order:

- OPD: `A_t -= opd_kl_coef * (logp_student_old - logp_teacher)` (`loss/hub/opd.py:8`), on top of
  any estimator.
- `--normalize-advantages`: masked whitening with global mean/var over the DP group
  (`advantages.py:108`, `distributed_masked_whiten`). Required for both REINFORCE++ variants
  (`arguments.py:3675-3679`).

Things that are easy to get wrong here:

- **`--kl-coef` is a no-op for GRPO and GSPO.** `get_grpo_returns` only uses `kl` for its
  shape. Nothing rejects the combination, and because `with_ref` is derived from
  `kl_coef != 0`, it still loads the reference and pays a ref forward every step. PPO
  forbids `--kl-coef` outright (`arguments.py:3755`). Use `--use-kl-loss --kl-loss-coef` for GRPO.
- `--use-kl-loss --kl-loss-coef 0` (the Qwen3-4B recipe with `--use-kl-loss`) loads the ref and
  logs `kl_loss` but adds nothing to the loss (`losses.py:337`).
- In REINFORCE++ the backward sum runs over *all* response positions, including tool-output
  tokens with `loss_mask = 0`, so their `-kl_coef * kl` still flows into earlier tokens' returns
  (`math_utils.py:512-523`). It also asserts every sample has at least one unmasked token
  (`:514`), so a fully masked sample (`remove_sample` from `--rollout-sample-filter-path`)
  crashes `reinforce_plus_plus`.
- `normalize_advantages` reduces over `effective_dp`, which excludes CP. With CP > 1 each CP
  rank whitens its own token slice with statistics from the same-CP-rank DP peers, so tokens of
  one sequence on different CP ranks are whitened with slightly different mean/std. Treat
  `--normalize-advantages` with CP > 1 as approximate.

### 4.2 The policy loss (`policy_loss_function`, `loss/hub/losses.py:64`)

Per micro-batch, with `logp` the current logprobs (grad on), `logp_old` as in 2.1, and
`m_t` the loss mask:

```text
log_ratio_t = logp_t - logp_old_t                               # ppo_kl = -log_ratio (losses.py:189)
    GSPO: log_ratio_t = mean_masked(logp - logp_old) over the whole sequence, broadcast (math_utils.py:224)
r_t   = exp(log_ratio_t)
l_t   = max( -r_t * A_t,  -clip(r_t, 1 - eps_clip, 1 + eps_clip_high) * A_t )          # compute_policy_loss, math_utils.py:254
    dual clip (--eps-clip-c c > 1): for A_t < 0, l_t = min(l_t, -c * A_t)
l_t  *= opsm_mask_i                                             # --use-opsm
l_t  *= w_t                                                     # --use-tis: w = clamp(exp(logp_old_trainer - logp_rollout), tis_clip_low, tis_clip)
loss  = Reduce(l) - entropy_coef * Reduce(H) + kl_loss_coef * Reduce(KL(logp, logp_ref))
```

`--eps-clip-high` defaults to `--eps-clip` (`arguments.py:3716-3717`). Masked positions have their
`ppo_kl` and advantage zeroed and NaN/inf scrubbed before the clip (`losses.py:200-209`).

**KL loss** (`--use-kl-loss`, `losses.py:318`), with `x = logp - logp_ref`:

```text
k1          = x
k2          = x^2 / 2
k3          = exp(-x) - 1 + x                  # Schulman's unbiased, non-negative estimator
low_var_kl  = k3 with the log-ratio clamped, result clamped to [-10, 10]
--use-unbiased-kl:  KL *= exp(logp - logp_old)   (the ratio is not detached, DeepSeek-V3.2 style)
```

**KL in reward** (`--kl-coef`) uses the same estimator but with the **old** logprobs against
ref, and only matters for REINFORCE++ variants (see 4.1). The two coefficients are mutually
exclusive: a nonzero `--kl-coef` with a nonzero `--kl-loss-coef` is rejected (`arguments.py:3673`).

**Entropy**: computed only when `--entropy-coef != 0` or `--observe-training-entropy`; in the
latter case it is detached (`losses.py:309-316`).

**Alternative losses**: `--loss-type sft_loss` (`losses.py:465`), `score_centering`
(`loss/hub/score_centering_loss.py`), `custom_loss` via `--custom-loss-function-path`, and
`value_loss` for the critic (`:407`).
`--custom-pg-loss-reducer-function-path` replaces the reducer for `pg_loss` only
(e.g. `examples/experimental/DrGRPO/custom_reducer.py`).

### 4.3 Loss normalization across micro-batches, DP and CP

This is the part most worth reading carefully when comparing numbers across frameworks.
`Reduce` is `get_sum_of_sample_mean` (`training_utils/data/context_parallel.py:247`), and the
dispatcher `loss_function` (`loss/objective.py:133`) rescales it for Megatron.

**Default (`--calculate-per-token-loss` off): per-sample mean, averaged over samples.**

```text
Reduce(x) over a micro-batch = sum_i  ( sum_t m_it * x_it ) / max(M_rollout(i), 1)
                               M_rollout(i) = rollout_mask_sums[i]   # sum of loss_mask over the whole rollout of sample i
Megatron: loss *= num_microbatches / num_rollouts_in_step * |DP x CP|      (objective.py:220)
          Megatron then divides by num_microbatches and averages grads over DP x CP
Net gradient objective per optimizer step:
    L = (1 / N_step) * sum_{i in step} (1 / M_rollout(i)) * sum_t m_it * l_it
```

`N_step` is the step's rollout count (`num_rollouts`, `data/rollout.py:448`), which is GBS unless
rollouts are compact. For ordinary samples `rollout_id = index`, so `M_rollout(i)` is just that
sample's own masked token count, and this is the original GRPO "mean over tokens, then over
samples" objective: long responses get the same total weight as short ones. Under CP each rank
sums its local tokens but divides by the full-sequence denominator, and the DP x CP gradient
all-reduce adds the pieces back up, so the CP split is exact (under `--true-on-policy-mode` with
Ulysses CP the factor is the DP size alone, `objective.py:215-219`). The LR scheduler also advances by
`num_rollouts` samples, not by 1 (`model.py:663`).

**`--calculate-per-token-loss`: token mean over the whole step (DAPO-style).**

```text
Reduce(x) = sum_i sum_t m_it * x_it           (sum_of_token)
normalizer returned to Megatron = sum_i max(sum_t m_it, 1)          (objective.py:172)
Megatron all-reduces the normalizer over DP and divides grads by it; loss *= cp_size compensates
for every CP rank reporting the full-sequence token count (objective.py:225-226)
Net:  L = sum_{step} m * l / sum_{step} max(|m_i|, 1)
```

The `clamp_min(..., 1)` means a fully masked sample (e.g. `remove_sample`) still adds 1 to the
denominator.

**TIS / rejection sampling** rebuild the reducer with the modified masks for the numerator but
keep `rollout_mask_sums` as the denominator (`losses.py:267-277`). So tokens rejected by MIS
`use_rs` still count in the per-sample denominator. The mismatch-helper page says rejected
tokens "will not be considered for loss averaging"; in the default Megatron path they are
removed from the numerator only. (On FSDP and the other torch-native backends, which have no
`rollout_mask_sums`, the rebuilt reducer falls back to the *modified* mask sums, so rejected
tokens leave the denominator too; `context_parallel.py:260-261`.)

**FSDP differences.** FSDP (and torchtitan) train through the shared torch-native actor
(`training_utils/torch_native/actor.py:240-256`, `:321-332`), which calls `loss_function` with
`apply_megatron_loss_scaling=False`, so the default mode is `loss / GBS * dp_size`
(`objective.py:222`) followed by FSDP's gradient averaging, the same objective as above. Under
`--calculate-per-token-loss` each optimizer step first all-reduces its token count
`sum_i max(|m_i|, 1)` over DP (`_count_step_loss_tokens`, `torch_native/actor.py:307-318`), and
`loss_function` divides by it and multiplies by the DP size (`objective.py:227-229`). After
FSDP's averaging that is the same step token mean as on Megatron; the returned normalizer is
still discarded (`:324`) but no longer needed. The torch-native backends do not fetch
`rollout_mask_sums` (`TRAIN_KEYS`, `torch_native/actor.py:55-64`), so they use each sample's own
mask sum as the denominator (identical except for compact rollouts and rejection sampling).

### 4.4 Optimizer step

`train_one_step` (`megatron_utils/model.py:562`): zero grads, run the pipeline
forward/backward over the step's micro-batches, then `optimizer.step()` with Megatron's grad
clipping (`--clip-grad`, default 1.0) and skip-on-NaN/inf, then
`opt_param_scheduler.step(increment=num_rollouts)`. `--reset-optimizer-states` clears Adam
state at the start of every rollout (`model.py:779-782`, in `train`). After a normal step the new
weights are copied to the host `"actor"` backup (`actor.py:791-794`) so that weight sync and model
switching can read them while the GPU copy is offloaded. That backup exists only with a
ref/teacher/old actor, `--colocate` or `--offload-train` (`_enable_weight_backup`, `actor.py:527`).
`--ref-update-interval N` copies the actor into the `"ref"` slot every N rollouts
(`actor.py:799-807`).

---

## 5. Weight sync and the colocated handoff

`update_weights` (`placement_group.py:296`) is a driver function that is called once before the
loop (`train.py:48`, so SGLang always serves trainer weights even if it booted from a different
HF checkpoint) and then after every step, except after the last step when no eval is due
(`train.py:143`).

```text
update_weights (driver)                                     placement_group.py:296
  info = inference_controller.start_update_weights()        # inference_controller.py:233: takes the controller
                                                            #   lock, pauses health checks, waits for cells;
                                                            #   returns the current engine set + GPU layout
  TrainerController.update_weights(info)                    group.py:391
    first alive cell: MegatronTrainRayActor.update_weights  actor.py:919
      (re)connect if the engine set changed                 # cell hashes, weight_update/conn_status.py
      WeightUpdater.update_weights                          training_utils/weight_update/updater.py:124
          rank 0: maybe_pause_engines(--pause-generation-mode) [+ flush_cache unless in_place]; begin_weight_update
          for bucket in HF-converted weights (--update-weight-buffer-size bytes):
              protocol.send_bucket(bucket)
          rank 0: end_weight_update; set_weight_version(v); maybe_resume_engines
  inference_controller.end_update_weights(...)              # marks new cells weights-ready, releases the lock
                                                            #   (abort_update_weights instead on failure)
  rollout_executor.set_weight_version(v)
```

The `InferenceController` holds its context lock (`miles/utils/context_lock.py`) from
`start_update_weights` to `end_update_weights`, so offload/onload, generation pause, cell
health ticks and cell add/remove all wait until the sync is over. The engine-side session RPCs
(`pause_generation`, `begin_weight_update`, `end_weight_update`, `update_weight_version`,
`continue_generation`) are in `training_utils/weight_update/session.py`.

The protocol is chosen in `get_weight_transfer_protocol` (`weight_update/protocol.py:73`):

| Layout / flag | Protocol | Mechanism |
|---|---|---|
| `--colocate` (always) | `UpdateWeightFromTensor` (`protocols/cuda_ipc.py:34`) | ranks gather CUDA IPC handles to the rank on each engine's first GPU, which calls SGLang `update_weights_from_tensor`; no copy over the network |
| `--update-weight-transfer-mode broadcast` (default) or `broadcast_packed` | `UpdateWeightFromDistributed` (`protocols/broadcast.py`) | one NCCL group per PP stage made of the DP=TP=0 trainer rank plus every engine GPU; metadata over HTTP (`update_weights_from_distributed`), tensors by `broadcast`, one per tensor or, with `broadcast_packed` (Megatron, not colocated), one flattened buffer per bucket |
| `p2p` | `UpdateWeightP2P` | RDMA point-to-point, see [P2P Weight Transfer](/advanced/p2p-weight-transfer) |
| `disk-delta` | `UpdateWeightFromDiskDelta` | diff against the previous snapshot, publish to `--update-weight-disk-dir`, engines pull |

In `train.py` generation has already finished when the sync starts, so the pause mode only
matters for eval traffic and for async. What each mode does inside SGLang (retract frees and
recomputes KV and flushes the radix cache; `in_place` keeps both; requests sent during the
pause wait in the TokenizerManager rather than being rejected) is traced in
[Async and Off-Policy, section 3.3](/deep-dive/async-and-off-policy). The CUDA IPC and
broadcast RPCs are sent with `flush_cache=False` (`sglang_api_client.py:145`, `:394`), so for
those protocols the only cache flush in a sync is the one in `maybe_pause_engines`
(`weight_update/session.py:15-26`). `disk-delta` runs without that session frame: it pauses and
flushes on its own (`protocols/delta.py:400-405`), and its `update_weights_from_disk` call keeps
SGLang's default `flush_cache=True`, so the engine flushes once more after the reload. Under
`--fully-async --colocate` the driver pauses generation itself and the updater skips the pause
(`driver_owns_generation_pause`, `arguments.py:75-76`). With `--keep-old-actor`, the sync also
rotates the host copies (`rollout_actor -> old_actor`, `actor -> rollout_actor`,
`actor.py:956-965`) so the next
old-policy forward uses exactly the weights the engines are about to generate with. That is
only useful in async; in `train.py` it costs memory and changes nothing.

**Colocated order of operations** (`train.py:107-156`, default `--colocate-memory-peak-device cpu`):

```text
rollout done  -> engines release CUDA graphs, KV cache, weights (--offload-rollout-level)
train         -> trainer wake_up (torch_memory_saver.resume), train, backup weights to host
handoff       -> trainer sleep (pause all tags)  -> engines resume weights
sync          -> CUDA IPC from the host "actor" backup   (_get_actor_weights, actor.py:898)
              -> engines resume KV cache
```

Releasing engine memory flushes the cache first on both sides. The Miles client calls
`/flush_cache` before `release_memory_occupation` (`sglang_api_client.py:295-301`), and SGLang
asserts that the engine is idle and flushes again when the `kv_cache` tag is released
(`sglang/python/sglang/srt/managers/scheduler_components/weight_updater.py:318-362`, SGLang
0.5.21). Resuming restores memory and does not touch the cache (`weight_updater.py:364-403`).
Both steps need an idle engine: the client's flush retries for 60 s and then raises
(`sglang_api_client.py:260-276`). Here `abort()` has already emptied the engines.

`--colocate-memory-peak-device gpu` reverses the order so both copies briefly coexist on the
GPU instead of in host RAM.

**Eval.** `EvalDispatcher.dispatch` (`miles/ray/rollout/eval_dispatch.py`) runs
`RolloutExecutor.eval` on the training engines, blocking, after the weight sync. With
`--eval-num-gpus > 0` (or a `CheckpointEvalFn` eval path) it instead exports an HF snapshot and
evaluates it on a separate fleet without blocking the loop. `--eval-interval` and
`--skip-eval-before-train` control cadence.

---

## 6. The `Sample` and the dict the trainer sees

### 6.1 `Sample` (`miles/utils/types.py:165`)

| Field | Meaning |
|---|---|
| `group_index` | prompt-group id; the unit of GRPO normalization |
| `index` | unique per sample, monotonically increasing |
| `rollout_id` | trajectory id; `None` means "= index". Siblings of a compact/sub-agent rollout share it |
| `prompt` | str or chat messages |
| `tokens` | prompt + response token ids, exactly what SGLang saw and emitted (no re-tokenization) |
| `response`, `response_length` | decoded response text; number of response tokens (model + tool) |
| `label` | ground truth passed to the RM |
| `reward` | float, or a dict selected by `--reward-key` |
| `loss_mask` | length `response_length`; 1 = trained on. `None` becomes all ones at conversion |
| `rollout_log_probs` | SGLang logprob per response token; 0.0 on tool tokens |
| `rollout_topk_token_ids`, `rollout_topk_log_probs` | `[response_length, k]` candidate logprobs, recorded with `--rollout-top-logprobs-num` |
| `rollout_sampling_mask` | per-token top-k/top-p support, for sampling-support replay |
| `rollout_routed_experts`, `rollout_indexer_topk` | R3 routing and DSA indexer replay data |
| `weight_versions` | per-call spans of which weight version produced which output tokens (and, as `prefill_spans`, the prompt KV) |
| `teacher_log_probs`, `opd_reverse_kl` | on-policy distillation |
| `remove_sample` | exclude from the loss (mask zeroed), keep in normalization |
| `status` | `PENDING, COMPLETED, TRUNCATED, ABORTED, FAILED` |
| `metadata`, `train_metadata` | free-form; `train_metadata` reaches the trainer as `metadata` |
| `multimodal_inputs`, `multimodal_train_inputs` | raw and processor-encoded multimodal data |
| `reward_spec`, `generate_function_path`, `routing_key` | per-sample RM, generate function, and router key overrides |
| `lineage`, `trainer_model_id`, `kv_cache_namespace` | source sample of a generated output; the policy a multi-policy run trains on it; prefix-cache namespace |
| `spec_info`, `prefix_cache_info`, `non_generation_time` | speculative-decoding, prefix-cache and timing stats |

`Sample.validate` (`types.py:346`) checks each per-token field that is present (`loss_mask`, `rollout_log_probs`, top-k, sampling mask, teacher fields) against `response_length`.

### 6.2 The train dict (`convert_samples_to_train_data`, `train_data_conversion.py:61`)

All lists are per sample, in the same order.

| Key | Content |
|---|---|
| `tokens` | full token ids |
| `response_lengths` | response token counts |
| `rewards` | group-normalized reward (or raw, see 3.4) |
| `raw_reward` | unnormalized reward, for logging |
| `truncated` | 1 if `status == TRUNCATED` |
| `sample_indices`, `rollout_ids` | ids |
| `loss_masks` | response-only masks |
| `rollout_mask_sums` | whole-rollout mask sum, the loss denominator (4.3) |
| `rollout_log_probs` | if present on the samples (always, by default) |
| `rollout_sampling_mask_ids/offsets`, `rollout_routed_experts`, `rollout_indexer_topk` | replay data |
| `rollout_topk_token_ids/log_probs` | candidate logprobs, with `--rollout-top-logprobs-num` |
| `metadata`, `multimodal_train_inputs`, `weight_versions`, `teacher_log_probs`, `opd_reverse_kl`, `round_number`, `prompt_group_sizes`, `dynamic_global_batch_size` | optional |
| `total_lengths` (added at split), `partition`, `num_microbatches`, `micro_batch_indices`, `num_rollouts` | scheduling |

On the trainer, `get_rollout_data` (`training_utils/data/rollout.py:32`) moves tokens and masks to GPU,
slices `rollout_log_probs` for the local CP rank, and the loss pipeline adds `log_probs`,
`ref_log_probs`, `advantages`, `returns` (and `values` for PPO). `get_batch` (`:168`)
builds the packed `tokens`, `cu_seqlens`, `unconcat_tokens`, and `input_loss_masks`: the
response mask left-padded with zeros over the prompt so it aligns with input tokens
(`_prepare_loss_masks`, `:122`). Prompt tokens are never in the loss.

`--custom-convert-samples-to-train-data-path` replaces the whole conversion, including reward
normalization; `--custom-reward-post-process-path` replaces only the reward part.

---

## 7. `train_async.py`, briefly

`train_async.py:81-161` overlaps generation of rollout `k+1` with training on rollout `k`:

```python
next = start(generate(start_id))
for k in range(...):
    data = await next
    next = start(generate(k + 1))                  # runs while we train on k
    await actor.train(k, data)
    if (k + 1) % update_weights_interval == 0:
        data_next = await next                     # wait: never swap weights mid-generation
        update_weights()
```

So each rollout is generated by weights one update behind the actor that trains on it.
Colocation is allowed only together with `--fully-async` (`train_async.py:24`); the driver then
pauses generation, flushes the cache and offloads the engines before training, and resumes them
after the sync (`:93-98`, `:129-146`), so generation and training take turns on the shared GPUs.
Consequences for the loss:

- By default the old-policy forward uses the *current* actor, not the policy that generated
  the data. `validate_async_off_policy_correction` (`arguments.py:4255`) requires one of
  `--use-rollout-logprobs`, `--use-tis` or `--keep-old-actor`, **but only when
  `use_critic` (PPO) is set**. Async GRPO with default flags passes validation and anchors the
  PPO ratio to a policy that never produced the trajectory.
- `--fully-async` replaces the rollout function with a persistent producer
  (`miles/rollout/fully_async_rollout.py`) and adds `--max-weight-staleness`.

Details: [Async and Off-Policy](/deep-dive/async-and-off-policy) and [Fully Async](/user-guide/fully-async).

---

## 8. Corrections to the existing docs

| Page | Claim | What the code does |
|---|---|---|
| [Architecture](/developer/architecture) | Data source is "owned by the trainer"; the sequence diagram has the trainer call `get_samples`, generate and score | The data source, generation and reward all live in the CPU `RolloutExecutor` actor; trainers only receive sharded train dicts |
| Architecture | Rollout sits behind the "Miles Router" | The default router is SGLang's `sglang_router`; `MilesRouter` needs `--use-miles-router` |
| Architecture | `sglang_rollout.py:generate_rollout` is the rollout to read | That is the legacy v1 path (`MILES_USE_LEGACY_ROLLOUT_V1=1`); the default is `inference_rollout/inference_rollout_common.py:InferenceRolloutFn` |
| Architecture, [Concepts](/user-guide/concepts) | Weight sync is `p2p` | Default is NCCL `broadcast`; CUDA IPC when colocated; `p2p` is opt-in |
| Concepts | The reference model is "never updated" | `--ref-update-interval` copies the actor into it |
| Concepts | "Set any three sides; Miles fills in the fourth" | Only `--num-steps-per-rollout` derives GBS (`arguments.py:4041-4049`). Otherwise the sample count is trimmed down to a multiple of GBS (`rollout_data_conversion.py:33-39`) instead of being rejected |
| [Rollout Correction](/examples/infra-features/train-infer-mismatch-helper) | `--get-mismatch-metrics` "will not change the loss in any way" | True with the shipped `mis.py`, but `pg_loss` is passed through whatever `--custom-tis-function-path` returns (`losses.py:234-265`) |
| Rollout Correction | RS-rejected tokens "will not be considered for loss averaging" | On Megatron they leave the numerator only; the denominator stays `rollout_mask_sums` (`losses.py:267-277`). On FSDP and the other torch-native backends (no `rollout_mask_sums`) they leave both |
| `corrections.py:35` docstring | `icepop_function` passes the in-range ratio through "unweighted" | It multiplies `pg_loss` by the in-range ratio, i.e. weighted |

---

## 9. open-instruct `grpo_fast` to Miles

| open-instruct concept | Miles equivalent |
|---|---|
| `PolicyTrainerRayProcess` (DeepSpeed ZeRO-3) | `MegatronTrainRayActor` (`megatron_utils/actor.py`), `FSDPTrainRayActor` (`fsdp_utils/actor.py`) or `TorchtitanTrainRayActor`, selected by `--train-backend` |
| `LLMRayActor` (in-process vLLM) | `sglang.launch_server` subprocess per engine under a `CommandActor`, reached over HTTP through the router |
| `DataPreparationActor` | `RolloutExecutor`: rewards, group normalization, DP split (advantages themselves are computed on the trainer) |
| `ActorManager.should_stop` around weight sync | `maybe_pause_engines` / `maybe_resume_engines` in the weight-update session (`weight_update/session.py`), `--pause-generation-mode` |
| `prompt_Q` / `inference_results_Q` | asyncio tasks inside `generate_rollout_async`, HTTP to the router |
| `num_unique_prompts_rollout` x `num_samples_per_prompt_rollout` | `--rollout-batch-size` x `--n-samples-per-prompt` |
| `num_mini_batches` | `--num-steps-per-rollout` (or `--global-batch-size`) |
| `pack_length` | `--use-dynamic-batch-size --max-tokens-per-gpu` (THD packing per micro-batch) |
| `sequence_parallel_size` (Ulysses) | `--context-parallel-size` (Megatron CP; Ulysses variant under true-on-policy) |
| `use_vllm_logprobs` | `--use-rollout-logprobs` |
| `truncated_importance_sampling_ratio_cap` | `--use-tis --tis-clip` (or MIS via `--custom-tis-function-path`) |
| `clip_lower` / `clip_higher` | `--eps-clip` / `--eps-clip-high` (plus dual-clip `--eps-clip-c`) |
| `beta` + `kl_estimator` | `--use-kl-loss --kl-loss-coef --kl-loss-type {k1,k2,k3,low_var_kl}` (`--kl-coef` is reward-shaping and ignored by GRPO) |
| `load_ref_policy` | implicit: ref is loaded iff `--kl-coef != 0` or `--use-kl-loss` |
| `ref_policy_update_freq` | `--ref-update-interval` |
| `advantage_normalization_type centered / standard` | `--disable-grpo-std-normalization` / default std normalization |
| `filter_zero_std_samples`, `active_sampling` | `--dynamic-sampling-filter-path ...apply_reward_nonzero_std_filter`, `--over-sampling-batch-size` |
| `loss_fn` (dapo / dppo, token-level denominator) | `--calculate-per-token-loss` for token-mean; default is per-sample mean; `--custom-pg-loss-reducer-function-path`, `--advantage-estimator gspo` |
| `async_steps` | `train_async.py` (exactly one step ahead) or `--fully-async` with `--max-weight-staleness` |
| `broadcast_to_vllm` | `update_weights`, `--update-weight-transfer-mode {broadcast,broadcast_packed,p2p,disk-delta}`, CUDA IPC when colocated |
| vLLM on disjoint GPUs | disaggregated layout (default); `--colocate` time-shares the same GPUs |

---

## 10. Flags by stage

| Stage | Flags (`miles/utils/arguments.py` unless noted) |
|---|---|
| Layout | `--actor-num-nodes`, `--actor-num-gpus-per-node`, `--rollout-num-gpus`, `--rollout-num-gpus-per-engine`, `--num-gpus-per-node`, `--colocate`, `--offload-train`, `--offload-rollout`, `--offload-rollout-level`, `--colocate-memory-peak-device`, `--train-backend`, `--use-miles-router` |
| Data | `--prompt-data`, `--input-key`, `--label-key`, `--apply-chat-template`, `--rollout-shuffle`, `--num-rollout` / `--num-epoch`, `--data-source-path`, `--buffer-filter-path` |
| Sampling | `--rollout-batch-size`, `--n-samples-per-prompt`, `--over-sampling-batch-size`, `--rollout-temperature`, `--rollout-top-p`, `--rollout-top-k`, `--rollout-max-response-len`, `--rollout-max-context-len`, `--rollout-stop(-token-ids)`, `--partial-rollout`, `--rollout-function-path`, `--custom-generate-function-path` |
| Reward | `--rm-type`, `--custom-rm-path`, `--rm-url`, `--group-rm`, `--reward-key`, `--custom-reward-post-process-path` |
| Filters | `--dynamic-sampling-filter-path`, `--rollout-sample-filter-path`, `--rollout-all-samples-process-path`, `--disable-rollout-trim-samples` |
| Advantages | `--advantage-estimator`, `--disable-rewards-normalization`, `--disable-grpo-std-normalization`, `--normalize-advantages`, `--kl-coef`, `--gamma`, `--lambd`, `--use-opd`, `--opd-kl-coef` |
| Logprobs | `--use-rollout-logprobs`, `--skip-actor-forward-only`, `--keep-old-actor`, `--ref-load`, `--ref-update-interval`, `--true-on-policy-mode`, `--recompute-logprobs-via-prefill`, `--use-rollout-routing-replay` |
| Loss | `--loss-type`, `--eps-clip`, `--eps-clip-high`, `--eps-clip-c`, `--use-kl-loss`, `--kl-loss-coef`, `--kl-loss-type`, `--use-unbiased-kl`, `--entropy-coef`, `--use-tis`, `--tis-clip`, `--tis-clip-low`, `--custom-tis-function-path`, `--use-opsm`, `--opsm-delta`, `--calculate-per-token-loss`, `--custom-pg-loss-reducer-function-path`, `--get-mismatch-metrics` |
| Batching | `--global-batch-size`, `--num-steps-per-rollout`, `--micro-batch-size`, `--use-dynamic-batch-size`, `--max-tokens-per-gpu`, `--log-probs-max-tokens-per-gpu`, `--balance-data`, `--use-dynamic-global-batch-size`, `--qkv-format`, `--data-pad-size-multiplier` |
| Optimizer | `--lr`, `--clip-grad`, `--reset-optimizer-states`, Megatron optimizer flags |
| Weight sync | `--update-weight-transfer-mode`, `--update-weight-buffer-size`, `--update-weights-interval`, `--pause-generation-mode`, `--check-weight-update-equal` |
| Eval / save | `--eval-interval`, `--skip-eval-before-train`, `--eval-num-gpus`, `--save`, `--save-interval`, `--save-hf` |
