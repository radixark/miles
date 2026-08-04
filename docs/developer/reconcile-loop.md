---
title: "Reconcile Loop"
description: "A minimal, level-triggered controller runtime for Miles, and how it maps onto the Kubernetes Go stack."
---

## What this is

- A deliberately small port of the [client-go informer stack](https://github.com/kubernetes/client-go/tree/master/tools/cache) plus [controller-runtime](https://github.com/kubernetes-sigs/controller-runtime).
- Tracks pools of processes — SGLang engine cells, trainer cells — that appear, disappear and restart mid-run.
- The alignment record: **every deviation from Go must appear below with a reason.**

## Modules

All under `miles/utils/workers/reconcile/`.

| Ours | Does | Go | Not 1:1 |
| --- | --- | --- | --- |
| `k8s_api.py` | LIST/WATCH calls; the only module importing `kubernetes_asyncio`, lazily so the loop stays importable without a Kubernetes backend | [typed client `List` / `Watch`](https://github.com/kubernetes/client-go/blob/master/kubernetes/typed/core/v1/pod.go) | |
| `k8s_reflector.py` | Cursor bookkeeping, relist on cursor rejection | [`cache.Reflector`](https://github.com/kubernetes/client-go/blob/master/tools/cache/reflector.go) | |
| `source_event.py` | Reflector-to-loop wire format: `UpsertEvent`, `DeleteEvent`, and a whole-world `ReplaceEvent` | [`watch.Event`](https://github.com/kubernetes/apimachinery/blob/master/pkg/watch/watch.go) + [`Store`'s write methods](https://github.com/kubernetes/client-go/blob/master/tools/cache/store.go) | Go writes to a store by method call; we send the same operations as values |
| `object_store.py` | Cache, parent index, replace with deletion synthesis | [`cache.Store`](https://github.com/kubernetes/client-go/blob/master/tools/cache/store.go) + [`DeltaFIFO.Replace()`](https://github.com/kubernetes/client-go/blob/master/tools/cache/delta_fifo.go) | Absorbs `Replace()`, the only part of `DeltaFIFO` we keep |
| `work_queue.py` | Insertion-ordered dedup with a wakeup, generic over the key type | [`workqueue`](https://github.com/kubernetes/client-go/blob/master/util/workqueue/queue.go) | |
| `retry_scheduler.py` | Per-key exponential backoff, latest-wins deadlines swept by one poll loop, generic over the key type | [rate limiter](https://github.com/kubernetes/client-go/blob/master/util/workqueue/default_rate_limiters.go) + [delaying queue](https://github.com/kubernetes/client-go/blob/master/util/workqueue/delaying_queue.go) | Absorbs both, because latest-wins is one mechanism, not two |
| `source_stream_driver.py` | Open, sync, reopen the stream; pump events into the store | [informer `Run` / `processLoop`](https://github.com/kubernetes/client-go/blob/master/tools/cache/controller.go) | |
| `loop.py` | Lifecycle, the single worker, resync | [controller-runtime `Controller`](https://github.com/kubernetes-sigs/controller-runtime/blob/main/pkg/internal/controller/controller.go) | |

An empty last column means 1:1. Each entry is the shadow of a **Dropped** / **Replaced** row below — the Go class died with its feature and the remainder landed on a neighbor — so the reason lives with that row.

## Decisions per module

### `k8s_reflector.py`

| Upstream | Solves | Decision | Reason |
| --- | --- | --- | --- |
| Reflector | Move remote changes into a local cache reliably | **Kept**, Kubernetes only | Ray / external-URL backends emit in-process: no cursor, no replay window |
| `watchHandler` per-event metadata failure | One malformed frame must not stop the watch | **Kept**: log, skip, advance past it | Tearing down reconnects at the same cursor, replays the same frame, wedges the watch until expiry |
| Bookmarks | Keep an idle cursor fresh | **Kept** | Free server-side, avoids relists |
| `BackoffManager` around reconnects | Survive a watch that keeps dropping | **Kept**, ours only | Always sending `timeout_seconds` is load-bearing: `kubernetes_asyncio` reads its absence as `watch_forever` and then reconnects and retries a 410 behind our back. Sending it keeps every reopen in `watch()` |
| `IsTooLargeResourceVersion` | A cursor from the future is never satisfied | **Kept** as the code 504, **dropped** as a reason string | A rolled-back backend would otherwise freeze the store forever, and a plain gateway timeout costs one LIST. Go finds `ResourceVersionTooLarge` in `Status.Details.Causes[].Type` ([`isTooLargeResourceVersionError`](https://github.com/kubernetes/client-go/blob/master/tools/cache/reflector.go)), never in `Status.Reason`, so matching it as a reason string never fires |
| Watch frame shape | Read a cursor out of whatever the client yields | **Kept**: an attribute, or a camelCase key when the frame is a dict | `kubernetes_asyncio` deserializes into a model unless it has none, and only then leaves the raw JSON, which is camelCase. A dict spelling `resource_version` is neither shape, so it is not accepted |
| `ApiException` shape | Tell a dead cursor from a transient failure | **Kept**: `status` against 410 / 504 only | `ApiException.__init__` sets `status` from `http_resp.status` or the int the watch layer passes, so it is always an int and never carries a separate `code` |
| An `ERROR` frame reaching the consumer | A watch that reports failure in-band | **Kept**, though `kubernetes_asyncio` cannot produce it | `Watch.unmarshal_event` raises `ApiException` on `type == "error"` rather than yielding, so the live path is the exception one. Kept because losing a cursor-death signal freezes the store forever and the dependency is unpinned |
| `ListAndWatch` → `Run` → LIST again | Refresh after every watch ends | **Dropped**; reopen WATCH from the cursor | A LIST per timeout dominates an idle reflector's cost, and the cursor is still valid |
| `BackoffUntil` | Keep a relist storm off the apiserver | **Replaced** by one flat `retry_delay` | One reflector, small label-scoped LIST |
| LIST pagination | Huge collections | **Dropped** | Thousands of pods at most |

### `object_store.py`

| Upstream | Solves | Decision | Reason |
| --- | --- | --- | --- |
| Store | Read without hitting the apiserver | **Kept**, a plain `dict` | Single-threaded asyncio: no locks |
| `Replace()` on relist | Deletions missed while disconnected | **Kept**, store-side | Ghost cells are forever. Store-side also survives a whole stream reopening, which a reflector-side diff cannot remember across. Costs one event type (`ReplaceEvent`) |
| Indexer | Large-scale reverse lookup | **Dropped**; `dict[ObjectKey, ParentKey]` scanned | The parent map is already the index |
| `EnqueueRequestForOwner` | Child event to parent key | **Kept** as `key_map`; an object it cannot map is evicted with an error, not stored | Cells are not Kubernetes objects, so the parent comes from labels, and one bad pod must not stall the pool. Go caches an object it cannot attribute and merely enqueues nothing, but this store is indexed by parent, so an object whose labels stop mapping leaves and re-drives the parent it left. Ignoring the update instead would serve it under that parent forever |
| DeltaFIFO | Delta coalescing | **Dropped** | Reconcile reads a snapshot, so the queue needs parent-key dedup, never a delta chain |

### `work_queue.py`

| Upstream | Solves | Decision | Reason |
| --- | --- | --- | --- |
| workqueue | The scheduling core | **Kept** as a dedup set; delayed retry lives in `retry_scheduler.py` | With one worker, the dirty/processing protocol collapses into the set |
| `ShutDown` vs `ShutDownWithDrain` | Finish in-flight work first | **Dropped**; `stop()` cancels everything, then waits. Awaiting it inside reconcile asserts (use `asyncio.create_task`). It may only follow a `start()` call, but a `start()` that raised leaves no tasks and `stop()` is then a no-op that returns, so an aborted start unwinds carrying its own exception instead of an assertion | Drain exists for many Go workers. One worker means one in-flight key, and reconcile is idempotent, so abandoning it costs a re-derivation. A caller that wraps `start()` in `except: await stop()` is the normal shape, and it must not lose the failure it is unwinding |

### `retry_scheduler.py`

| Upstream | Solves | Decision | Reason |
| --- | --- | --- | --- |
| Delaying queue on earliest `readyAt` | Rate-limited retry | **Replaced** by latest-wins deadlines swept by one `POLL_INTERVAL` loop | A new failure overwrites a deadline instead of cancelling a task, so the delay always matches current state. Go sweeps a heap on an exact timer inside one `waitingLoop` goroutine; polling keeps the single loop and costs a retry up to one interval late, which a backoff heuristic absorbs. The poller is created in `__init__` and cancelled in `shutdown()`, so its lifetime is a straight line and it ticks even when idle, which costs one wakeup a second and buys having no state that decides when to run. Creating it there is why a `ReconcileLoop` must be constructed inside a running event loop |
| Bucket rate limiter | Cap retry pressure | **Dropped** | In-memory accounting behind one worker: no backend to protect |
| Delaying queue *is* a queue (`delayingType` embeds `Interface` and re-`Add`s to itself) | Retry without a second component | **Replaced** by an `on_retry` callback | Nothing here reads the queue, so a scheduler that only pushes needs no queue at all. `WorkQueue` and `RetryScheduler` then know no domain type, and `ReconcileLoop` names the key by instantiating them as `[ParentKey]` |
| `Result{RequeueAfter}` | Timed re-check | **Dropped** | Backoff plus resync covers it |

### `source_stream_driver.py`

| Upstream | Solves | Decision | Reason |
| --- | --- | --- | --- |
| Cache-before-notify ordering | Handlers never read a stale state | **Kept** | Correctness, not volume |
| WaitForCacheSync | Do not decide on a half-filled cache | **Kept**: `run()` + `wait_for_sync()`, the [`SyncingSource`](https://github.com/kubernetes-sigs/controller-runtime/blob/main/pkg/source/source.go) shape; `start()` awaits it | A partial engine list at step 0 would silently shrink the pool |
| `source.Channel` | External event injection | **Dropped** | Miles-internal events are method calls |
| Retry inside vs outside the stream | Recover without losing the cursor | **Kept** at both levels: `KubernetesReflector.retry_delay`, then `ReconcileLoop.source_retry_delay` | The inner one recovers a dropped watch, a failed LIST or an expired cursor without ending the stream. The outer one is the net for a stream that dies for good; in-process registries reach it |
| Closing a stream during teardown | Do not hide a worse error | **Replaced**: every close is logged, none propagates | `run()`'s `finally` closes the stream and clears it before `stop()` gathers the driver, so `aclose()` only ever finds nothing left to close. That leaves `_aclose_logging_failure` as the single close path, and it must swallow: a close failure would otherwise mask an unwinding cancellation, or re-raise the very failure the driver exists to recover from |
| Streaming a listing into the store | Sync a huge collection without holding it in memory | **Dropped**; a listing is one `ReplaceEvent`, and a stream that does not open with one is reopened | client-go buffers too: its watch-list path fills a `temporaryStore` until the `k8s.io/initial-events-end` bookmark, then calls `Replace` once. Ours materializes the page before yielding anyway, so bracketing it would only add a segment FSM and a half-applied state to police |

### `loop.py`

| Upstream | Solves | Decision | Reason |
| --- | --- | --- | --- |
| Resync period | Level-triggered backstop | **Kept**, on by default at Go's 10 h, re-enqueues parents that still have members | Go defaults it on for the same reason we need it: insurance against this controller, or something it depends on, failing to requeue. Go's own limit is kept too, in that a parent which lost its last member is not re-driven. The 10 percent jitter is **dropped**: it spreads the LIST storm of many controllers, and there is one loop here whose resync only adds keys to a dedup set |
| `MaxConcurrentReconciles` | Overlap reconcile I/O | **Dropped**; one worker | controller-runtime's default. No I/O to overlap |
| Predicates | Save reconciles at scale | **Dropped** | Reconcile is cheap |
| Lifecycle | Never leak a running loop | **`async with loop:`** | `start()` and `stop()` stay public, because `stop()` sometimes has to be scheduled from inside reconcile. Every caller that owns both ends writes the `async with` instead, and a caller that hands teardown back to someone else keeps the pairing with `AsyncExitStack.enter_async_context(loop)` |

### Dropped wholesale

| Upstream | Reason |
| --- | --- |
| SharedInformer | One or two consumers per object type |
| Manager, leader election, metrics, webhooks, multi-GVK cache, cached client | In-process objects, not a deployed operator |
| kubebuilder / Operator SDK | One resource type, hand-written |
| Thread-safe store / DeltaFIFO / workqueue | Under asyncio state mutates only between `await` points. Removes the data-race class, not the interleaving class: "check, await, mutate" still needs FSM discipline |

## Test layers

All three run on every PR. `pytest tests/e2e/k8s_apiserver tests/e2e/k8s_kind` self-provisions everything below from a Docker daemon, and the modules that provision it depend on nothing from the reconcile package, so an environment can be verified before anything uses it.

| Layer | Where | Proves | Provisions |
| --- | --- | --- | --- |
| Fakes + fake clock | `tests/fast/utils/workers/reconcile/` | Control flow, timing, shutdown | Nothing |
| Real apiserver, no kubelet | `tests/e2e/k8s_apiserver/` | API semantics: cursors, watch timeouts, real 410, relist | etcd and the apiserver as containers, plus a second apiserver with `--watch-cache=false` — with the cache on, a compacted `resourceVersion` is still served, so nothing can invalidate a live cursor |
| kind cluster | `tests/e2e/k8s_kind/` | What only a kubelet produces: Running, restarts, graceful deletion, bookmarks | A pinned kind binary, downloaded on demand |

| Env var | Effect |
| --- | --- |
| `MILES_K8S_KEEP=1` | Leaves the environment up |
| `MILES_K8S_KUBECONFIG=<path>` | Reuses an existing cluster |
| `MILES_K8S_REQUIRE=1` | Default in CI; missing Docker becomes a failure |
