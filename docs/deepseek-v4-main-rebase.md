# DeepSeek-V4 Main Rebase

## Scope

- Task: rebase Miles DeepSeek-V4 support from local `deepseek-v4` onto current `upstream/main`.
- Miles base branch: local `deepseek-v4` at `032721cd6`.
- Remote DeepSeek-V4 branch: `origin/deepseek-v4` at `0151cdbbc`; local branch is ahead by one commit.
- Target Miles main: `upstream/main` at `f38996d00` (`radixark/main` points to the same commit).
- Target SGLang runtime/API: `sglang-oss/origin/sglang-miles-v0.5.12` at `244ff926f`.
- Worktree: `/Users/yueming.yuan/radixark/sunrise/miles-deepseek-v4-main-rebase`.

## Log

### 2026-05-21

1. Confirmed the primary `miles-oss/` checkout is dirty on `experiment_runner`; no edits made there.
2. Confirmed `sglang-oss/` is dirty on `deepseek_v4_dump`; it will be used only as historical context.
3. Fetched Miles remotes and verified `origin` is the user fork, while `upstream`/`radixark` are `radixark/miles`.
4. Created branch `deepseek-v4-main-rebase` from local `deepseek-v4`.
5. Created isolated worktree at `/Users/yueming.yuan/radixark/sunrise/miles-deepseek-v4-main-rebase`.
6. Started `git rebase upstream/main`; rebase stopped on `6e340712c DeepSeek V4 RL support`.
7. Resolved the first conflict set by keeping current main infrastructure and replaying only the DeepSeek-V4 semantic additions.
8. Continued rebase; Git dropped `7fb00e5 cleanup` because its patch is already upstream.
9. Rebase stopped on `b6fbf677f add pro model training support`.

## Conflict Notes

### `6e340712c DeepSeek V4 RL support`

Conflicted files:

- `examples/train_infer_mismatch_helper/mis.py`
- `miles/backends/megatron_utils/actor.py`
- `miles/backends/megatron_utils/arguments.py`
- `miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_fp8.py`
- `miles/backends/megatron_utils/model.py`
- `miles/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py`
- `miles/backends/training_utils/cp_utils.py`
- `miles/backends/training_utils/data.py`
- `miles/backends/training_utils/loss.py`
- `miles/rollout/generate_utils/generate_endpoint_utils.py`
- `miles/rollout/sglang_rollout.py`
- `miles/router/router.py`
- `miles/utils/arguments.py`
- `miles/utils/external_utils/command_utils.py`
- `miles/utils/types.py`
- `miles_plugins/mbridge/__init__.py`
- `tools/convert_hf_to_torch_dist.py`

Resolution approach: keep current `upstream/main` infrastructure by default, then re-add only DeepSeek-V4-specific hooks after comparing final old-branch code, current main code, and the target SGLang branch.

Decisions:

- Kept main versions for files where the old branch only carried stale infrastructure or unrelated changes: `examples/train_infer_mismatch_helper/mis.py`, `miles/backends/training_utils/cp_utils.py`, `miles/backends/training_utils/loss.py`, `miles/router/router.py`, `miles/utils/external_utils/command_utils.py`, and `miles/backends/megatron_utils/model.py`.
- Updated replay CP handling in `miles/backends/megatron_utils/actor.py` to use the current `ParallelState` API (`parallel_state.cp.size`, `parallel_state.tp.size`) while preserving V4 `allgather_cp` contiguous slicing.
- Kept `miles/backends/training_utils/data.py` on main's `slice_with_cp` API and added the V4 `allgather_cp`/DSA padding behavior against the current `ParallelState`.
- Added `rollout_indexer_topk` as a `numpy.ndarray` field, matching the target SGLang `sglang-miles-v0.5.12` base64 int32 payload style rather than the old list-of-list shape.
- Combined the routed-experts and indexer-topk response decoding paths in rollout code; indexer replay shape is derived from current args when explicit SGLang metadata is unavailable.
- Combined main linear-attention FP8 quantization entries with DeepSeek-V4 attention/indexer entries.
- Kept the main `get_parallel_state().tp.size`/`etp.size` weight bucket sizing while preserving V4 SGLang-fusion atomic bucket groups.
- Combined MBridge exports from main with optional `DeepseekV4Bridge` registration and config-based dispatch.
- Combined the HF-to-torch-dist conversion fallback from main with the V4 FP32-preserving MBridge weight conversion patch.

### `b6fbf677f add pro model training support`

Conflicted files:

- `tools/convert_hf_to_torch_dist.py`

Decision:

- Adopted the commit's `pipeline_model_parallel_size <= num_layers` check instead of main's stricter `world_size <= num_layers` check. The total world size can legitimately exceed layer count when TP/CP are used, while PP size is the value constrained by layer partitioning.
- Kept main's later automatic PP-size derivation and fallback config loading from the first conflict resolution.

## Fix Notes

- SGLang target `sglang-miles-v0.5.12` exposes replay tensors as base64-encoded int32 buffers; Miles now treats both routed experts and indexer topk as ndarray-backed replay tensors.
- Removed stale old-branch calls to `slice_with_cp(..., parallel_state, ...)` and stale `self.parallel_state.cp_size`/`tp_size`-style access in resolved conflict regions.

## Validation

- `git diff --check` passed after resolving the first conflict set.
- Conflict marker scan passed for Miles/example/tool/doc paths after resolving the first conflict set.
