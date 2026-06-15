# Miles SWEGym Debug Notes

## 2026-06-11: wandb entropy logging

- `train/entropy_loss` is logged from the training loss report, not from a standalone entropy monitor.
- Code path:
  - `miles/backends/training_utils/loss_hub/losses.py` builds `reported_loss["entropy_loss"]`.
  - `miles/backends/training_utils/log_utils.py::aggregate_train_losses` aggregates those reported loss values.
  - `miles/backends/training_utils/log_utils.py::log_train_step` prefixes every loss key with `train/` and sends the dict to `tracking_utils.log`.
  - `miles/utils/tracking_utils/base.py::WandbBackend.log` calls `wandb.log(metrics)` without extra processing.
- With `--entropy-coef 0.0`, training loss code calls `get_log_probs_and_entropy(..., with_entropy=False)` and then sets `entropy_loss = 0`, so wandb `train/entropy_loss` is expected to be 0.
- The non-loss entropy logging path is `rollout/entropy`, produced only when actor-side logprob recomputation runs with `--use-rollout-entropy`.
- In `--debug-rollout-only`, actor training returns immediately after `log_rollout_data`, before actor-side logprob recomputation, so `--use-rollout-entropy` alone will not create `rollout/entropy` for that mode.
- `log_correct_samples` currently computes `correct_entropy` after the main rollout wandb log has already been emitted, and there is no second `gather_log_data` call for it. As written, `correct_entropy` is not actually logged to wandb.

Recommended fixes:

1. For normal non-debug training, add `--use-rollout-entropy` if `rollout/entropy` is needed without adding entropy to the train loss.
2. For `--debug-rollout-only`, add a separate debug path that runs actor-side forward-only logprob/entropy recomputation before `log_rollout_data`, or accept that only SGLang sampled-token `rollout_log_probs` are available.
3. Move the `log_correct_samples` block before the main `gather_log_data("rollout", ...)`, or log its computed metrics in a second explicit wandb call.

## 2026-06-15: agentic merge crash on truncated intermediate sample

- Log: `/fs/nlp-intern/yangchengyi/logs/Qwen3.5-27B_agentic_debug_with_over_sample_20260611_155244.log`.
- The visible error is `AssertionError('a.status must be COMPLETED, got Status.TRUNCATED')` in `miles/rollout/generate_utils/sample_utils.py::_merge_sample_pair`.
- Call path:
  - `inference_rollout_train.py::generate_rollout_async` reads a finished group task.
  - `inference_rollout_common.py::generate_and_rm_group` gathers per-sample agent tasks.
  - `agentic_tool_call.py::generate` converts session records into per-turn samples, applies `truncate_samples_by_total_tokens`, then calls `merge_samples`.
  - `sample_utils.py::_merge_sample_pair` permits the final merged segment `b` to be `TRUNCATED`, but asserts every previous accumulated segment `a` is `COMPLETED`.
- Root cause: a long multi-turn agent trajectory can contain an intermediate per-turn sample whose status is already `TRUNCATED` (for example from OpenAI/SGLang `finish_reason == "length"`). The agent/session can still have later records, so `merge_samples` eventually tries to merge another later turn after the truncated accumulated sample, causing `a.status == TRUNCATED`.
- A pure Miles-side `max_seq_len` truncation is less likely to be the only cause because `truncate_samples_by_total_tokens` breaks after adding the first max-seq-truncated sample; that case should make the truncated sample the final `b`, which `_merge_sample_pair` already permits. The safer fix is still to cut the per-turn list at the first `TRUNCATED` sample before merging.
- This is a Miles agentic merge/truncation contract bug, not a CUDA or Harbor-server crash. The rollout loop catches the task exception and continues, so the job may remain `RUNNING` while this group is dropped.
- The current dynamic filter `check_no_aborted_and_reward_nonzero_std` only rejects `ABORTED`, not `TRUNCATED`, so truncated samples can also pass into train data if the group has nonzero reward std.

Recommended fixes:

1. In `agentic_tool_call.generate`, cut the per-turn sample list at the first `TRUNCATED` sample before `merge_samples`.
2. Optionally add debug logging of per-turn `(finish_reason, status, total_tokens, response_length)` before merge so this class is visible without session dumps.
3. Update the dynamic filter if truncated samples should not train: add a `check_no_aborted_or_truncated_and_reward_nonzero_std` variant or extend the current filter to reject both statuses.

Implemented local fix:

- `agentic_tool_call.generate` now calls `_cut_after_first_terminal_turn` before `merge_samples`.
- If a middle turn is terminal, it logs a warning, drops later turns, merges only up to that terminal turn, and writes metadata keys:
  - `agentic_terminal_turn_status`
  - `agentic_terminal_turn_index`
  - `agentic_terminal_turn_dropped_count`
  - `agentic_intermediate_truncated`
- `dynamic_sampling_filters.py` now has `check_no_aborted_or_truncated_and_reward_nonzero_std`.
- The SWEGym sample-param script now uses that new filter, so these samples are dropped before train conversion and logged in wandb as dynamic-filter drops such as `rollout/dynamic_filter/drop_group_has_agentic_intermediate_truncated`.
