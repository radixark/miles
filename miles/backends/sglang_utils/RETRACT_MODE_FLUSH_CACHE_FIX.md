# Fix: `flush_cache()` no-ops after `pause_generation(mode="retract")`

Spans two repos: sglang (fork branch `zhichen/retract-flush-cache-fix` off `origin/sglang-miles`,
the branch actually deployed on the validation pod) and miles (this branch, off `feat/torchtitan_exp`
so `torchtitan_utils` is present for the qwen3.5-4b validation run).

## Root cause

`Scheduler.flush_cache()` (`python/sglang/srt/managers/scheduler.py`) only resets `tree_cache`,
`req_to_token_pool`, and `token_to_kv_pool_allocator` when `self.is_fully_idle()` is `True`; otherwise
it logs a warning and returns `success=False`. `is_fully_idle()` requires `len(self.waiting_queue) == 0`.

`pause_generation(mode="retract")` retracts every running request and re-queues it into
`self.waiting_queue` via `_add_request_to_queue`. So immediately after a retract-mode pause,
`waiting_queue` is non-empty by construction — `flush_cache()` called right after **always** no-ops.

Retracted requests already had their KV released (`release_req` → `release_kv_cache(is_insert=False)`)
and are tagged `req.is_retracted = True` (cleared back to `False` on re-admission,
`schedule_batch.py`) — they hold no pool/tree_cache state that a flush needs to protect. The
`waiting_queue` check was never meant to block on requests in this specific state; it just wasn't
written to distinguish them from a genuinely new request that legitimately needs the flush deferred.

## Blast radius (found via workflow-based mapping, both repos)

1. **miles: silent correctness risk.** All weight-update call sites discard `flush_cache`'s
   return value, so the radix tree is never actually reset across a weight update when using
   retract mode. Stale pre-update KV can survive as prefix-cache hits for future requests. Zero
   observed impact on our own qwen3-4b/qwen3.5-4b E2E runs (`prefix_cache_hit_rate` was 0.0
   throughout — no prompt reuse in gsm8k) but real for any workload with prefix reuse
   (shared system prompts, multi-turn, repeated-prompt sampling).
2. **sglang: `release_memory_occupation`'s `assert self.is_fully_idle()`** (`weight_updater.py:329`)
   crashes outright during a retract-pause today.
3. **sglang: `flush_cache_after_weight_update`'s `assert flush_cache_success`** (`weight_updater.py:116-120`,
   called from `update_weights_from_disk`/`_from_distributed`/`_from_tensor`) should also crash —
   gated on `recv_req.flush_cache` (default `True` on `UpdateWeightsFromTensorReqInput`). Our own
   torchtitan/fsdp backends avoid this by explicitly passing `flush_cache=False` on the per-bucket
   transfer call (they rely on the standalone `pause_generation → flush_cache` call instead).
   **`megatron_utils/update_weight/update_weight_from_tensor.py`'s bucket-transfer call site does
   NOT pass `flush_cache`, so it uses the library default of `True`** — meaning the megatron
   backend combined with the (default) `--pause-generation-mode retract` should hit this assert
   and crash today. Not verified empirically in this design pass (out of scope: the validation
   target is qwen3.5-4b on torchtitan, not megatron) — flagged as a real, currently-latent,
   more-severe-than-ours finding that this same fix resolves for free. Worth an empirical check
   by whoever owns the megatron backend.
4. **`megatron_utils/update_weight/update_weight_from_distributed/{delta.py:252,mixin.py:269}`**:
   unrelated latent bug found in passing — `if mode not in ("in_place")` is a parenthesized string,
   not a tuple, so this is a substring check, not membership. Harmless today (no mode string is a
   substring of `"in_place"` except itself) but a footgun. Fix alongside since we're touching these lines.

## Design

Two independent designs were produced and cross-reviewed from three lenses (correctness/safety,
API cleanliness, test feasibility). Consensus: the minimal, opt-in, default-off parameter approach
is safer and cleaner than redefining `is_fully_idle()`'s semantics globally — but two ideas from the
rejected alternative are worth keeping, and reviewers found two real gaps in the accepted design
that must be closed before implementing.

### sglang-side change

`is_fully_idle()` gets one new default-off parameter. Every existing caller (`on_idle`,
`attach/detach_hicache_storage_wrapped`, `process_input_requests`, `SchedulerFlushWrapper`) keeps
calling it with no args and sees byte-identical behavior — this is the load-bearing property that
makes the change safe, and the regression test must assert it explicitly.

```python
def is_fully_idle(self, for_health_check=False, ignore_retracted: bool = False) -> bool:
    waiting_queue = self.waiting_queue
    if ignore_retracted:
        waiting_queue = [r for r in waiting_queue if not r.is_retracted]
    idle = (
        self.running_batch.is_empty()
        and self.chunked_req is None
        ...  # unchanged
    )
    idle &= len(waiting_queue) == 0
    ...  # unchanged
    return idle
```

`flush_cache()` passes `ignore_retracted=True`, **gated on `self._engine_paused`** — this is the
correctness-reviewer's catch: without gating on the pause flag, a transient/accidental retraction
outside of an explicit `pause_generation` call (e.g. a future memory-pressure `retract_decode`
during normal operation) could let a misplaced `flush_cache()` call succeed instead of failing
loudly, silently removing a safety net that currently catches caller bugs.

```python
def flush_cache(self, empty_cache: bool = True) -> bool:
    if self.is_fully_idle(ignore_retracted=self._engine_paused):
        ...  # unchanged reset logic
        success = True
    else:
        logger.warning(...)
        success = False
    return success
```

Apply the identical pattern to `release_memory_occupation`'s assert (`weight_updater.py:329`):
`assert self.is_fully_idle(ignore_retracted=self._engine_paused)`.

**Stolen from the rejected design**: split success handling the way `weight_updater.py` already
does elsewhere (`assert flush_cache_success` is the established internal idiom) — add
`flush_cache_or_raise` for internal callers that require success, keep the bare bool for the
external `/flush_cache` HTTP probe (which shouldn't crash the process on a legitimate "still busy"
response):

```python
def flush_cache_or_raise(self, **kwargs) -> None:
    if not self.flush_cache(**kwargs):
        pending = sum(not r.is_retracted for r in self.waiting_queue)
        raise RuntimeError(f"flush_cache failed: {pending} non-retracted req(s) still pending")
```

Replace `weight_updater.py`'s 4 `assert flush_cache_success` call sites with
`self.flush_cache_or_raise(empty_cache=recv_req.torch_empty_cache)`.

No HTTP/RPC/`grpc_bridge.py`/`engine.py` change needed — `flush_cache`'s `success` bool already
flows back through the existing `FlushCacheReqOutput` → JSON `{"success": ...}` path. The gap is
purely that the miles HTTP client discards it (see below).

### miles-side change

**Correction from the initial workflow-produced design**: `backends/sglang_utils/sglang_engine.py`'s
`flush_cache()` does NOT silently drop the failure signal — `/flush_cache`'s HTTP route already maps
`success` to the status code (`200` on success, `400` via `HTTPStatus.BAD_REQUEST` on failure), and
the client's retry loop already checks `status_code == 200`, retrying up to 60 times and raising
`TimeoutError` if it never succeeds. Verified live on zhichen-tt: racing a real in-flight generation
request against `pause_generation(mode="retract")` + `flush_cache` returns `400 "Flush cache
failed."` pre-fix and `200 "Cache flushed."` post-fix, with debug instrumentation confirming
`retract_all` really does populate `waiting_queue` (`n_retracted=1 n_waiting=1`) in both cases — the
scheduler-side fix alone is what changes the outcome. **No change needed to `flush_cache()`'s
control flow.** The one worthwhile improvement is surfacing *why* it failed in the eventual
exception, since `requests.get()` doesn't raise on a 4xx and the body is currently never read:

```python
def flush_cache(self):
    if self.node_rank != 0:
        return
    for _ in range(60):
        try:
            response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache")
            if response.status_code == 200:
                return
            last_message = response.text
        except NewConnectionError as e:
            raise e
        except Exception as e:
            logger.info(f"Error flushing cache: {e}")
            last_message = str(e)
            time.sleep(1)
            continue
    else:
        raise TimeoutError(f"Timeout while flushing cache: {last_message}")
```

**Considered and dropped**: the correctness reviewer's flagged risk — a raised `flush_cache()`
aborting `update_weights()` mid-body so `continue_generation` never runs — is real, but
`update_weights()` runs identically on every rank of a `torch.distributed` process group, coordinated
via `dist.barrier(group=get_gloo_group())` calls between phases, with the RPC fan-out itself guarded
to `dist.get_rank() == 0`. A naive `try/finally` around rank 0's block only protects rank 0; the other
ranks would still be blocked on the barrier rank 0 never reaches if it raises past that point,
trading today's "job crashes" for a strictly worse "job hangs." Making this safe requires either
broadcasting the failure to all ranks before any of them can hit a barrier, or restructuring the
barrier layout — a real but separate distributed-correctness project, not a same-diff side fix.
Deferred to a follow-up; not bundled here.

**What actually changes on the miles side**: two small, independently-safe fixes surfaced by the
mapping, unrelated to the barrier question above:

1. `experimental/torchtitan_utils/update_weight_utils.py` hardcodes `mode="retract"` instead of
   reading `self.args.pause_generation_mode` like `megatron_utils` does — fixed so users can opt out
   to `abort`/`in_place` for the torchtitan backend too, matching every other backend.
2. `megatron_utils/update_weight/update_weight_from_distributed/{delta.py,mixin.py}`'s
   `if mode not in ("in_place"):` is a parenthesized string, not a tuple — a substring check, not
   membership. Harmless today (no valid mode string is a substring of `"in_place"` except itself, so
   behavior is unchanged for every real value) but a latent footgun — fixed to `mode != "in_place"`
   while touching these lines.

## Test plan

**sglang regression** (`test/registered/unit/managers/`, extending `test_scheduler_pause_generation.py`'s
`_new_scheduler()` fixture, pure `unittest.mock`, no GPU, seconds to run):

```python
def test_flush_cache_after_retract_succeeds(self):
    sched = self._new_scheduler()
    sched.running_batch.reqs = [make_mock_req()]
    sched.pause_generation(PauseGenerationReqInput(mode="retract"))
    self.assertTrue(sched.waiting_queue[0].is_retracted)
    self.assertTrue(sched.flush_cache())
    sched.tree_cache.reset.assert_called_once()

def test_flush_cache_still_blocks_on_genuine_pending_request(self):
    # Closes the vacuous-fix gap both reviewers flagged: a buggy
    # "always return True" implementation must not pass this.
    sched = self._new_scheduler()
    sched.waiting_queue.append(make_mock_req())  # is_retracted=False, a real new request
    self.assertFalse(sched.flush_cache())
    sched.tree_cache.reset.assert_not_called()

def test_is_fully_idle_default_unchanged(self):
    # Regression guard: every other caller's default-arg behavior is untouched.
    sched = self._new_scheduler()
    sched.waiting_queue.append(make_mock_req(is_retracted=True))
    self.assertFalse(sched.is_fully_idle())  # ignore_retracted defaults to False
```

**miles functional check**: `SGLangEngine.flush_cache()` already retries on non-200 and raises
`TimeoutError` after exhausting retries — this was verified correct as-is (see the live-server
confirmation above); no client-side behavior change needed there beyond surfacing the failure
message. Coverage for the two miles-side fixes (torchtitan_utils reading `pause_generation_mode`,
the `delta.py`/`mixin.py` substring fix) is a straightforward read of the changed lines — no new
test infra needed for either.

**Known gap, accepted**: both the unit tests above and the mock-based miles check exercise the
*gating logic* only — they use mocked `tree_cache`/pool objects, so neither proves memory is
*actually* reclaimed on the real KV allocator. That proof comes from the GPU validation run below.

## Validation plan (qwen3.5-4b, zhichen-tt, 8xH200)

Per the test-feasibility review: a full training run against a historical baseline (wandb run
`4zis3xmh`) is too noisy to isolate a flush_cache-sized utilization effect, and is a slow way to
re-prove what the unit tests should already localize. Use it to confirm the fix doesn't regress
*at scale*, not as the primary correctness proof:

1. **Functional correctness**: relaunch the same qwen3.5-4b torchtitan config used for the M7 run,
   with the fix applied on both repos. Compare `train_rollout_logprob_abs_diff`, `raw_reward`,
   `eval/gsm8k` trajectory shape against the M7 baseline — expect no regression (the fix only
   changes whether the cache is *actually* cleared, not the weight-sync numerics).
2. **Stability**: no crashes, no hangs, `ray job status` reaches `SUCCEEDED`.
3. **Utilization, measured precisely instead of via wandb's coarse system panel** (both reviewers'
   point): timestamp `pause_generation`/`flush_cache`/`begin_weight_update`/`continue_generation`
   explicitly in the monitor script, then bracket that window with:
   - `nvidia-smi dmon -s pucvmet` sampled continuously at 1s through the run.
   - sglang's own `/metrics` Prometheus endpoint for queue depth and flush latency specifically
     during the weight-update window.
   - Expect a *small, real* added cost per weight update now that flush_cache actually executes
     (it was previously a no-op) — verify this doesn't meaningfully change per-rollout wall-clock
     versus the M7 baseline's ~45-90s/rollout pace.
4. Iterate on the implementation if any of 1-3 regress; otherwise this closes the mandate.
