---
title: Session Server Disk Offload
description: Session-record SQLite storage, ownership, failure handling, and collect cleanup.
---

This page covers session-record disk offload development. Training actor and optimizer offload are covered separately in [Disk Offload](/advanced/disk-offload).

## Configuration and scope

Session servers offload records to SQLite by default. `--disable-session-server-disk-offload` keeps records in memory through the same interface. `--session-server-disk-offload-dir` selects a local directory; when omitted, the server uses `miles-session-records` beneath the system temporary directory. Select a disk directory explicitly if the system temporary directory is memory-backed. The launch builder passes the switch, directory, and run UUID through `SessionServerConfig.disk_offload`, `disk_offload_dir`, and `run_id`.

The choice is fixed for the process lifetime. Disabled mode creates no SQLite backend or writer and never accesses the offload directory. Initialization, encoding, or write failures warn and retain complete records in memory. They do not add a chat admission gate or terminate the server.

Stage 1 has no memory limit, reservation, capacity backpressure, or capacity-based request rejection. Persistent write backlog or memory-only operation can exhaust memory. Existing validation and shutdown gates still apply. This is temporary, process-local storage: restart recovery, cross-worker takeover, archive retention, and locator-only dump formats are outside Stage 1. GET, samples, and existing debug/train dumps retain their complete contents.

## Ownership and interfaces

| Source | Responsibility |
| --- | --- |
| `miles/rollout/session/record/store.py` | Complete snapshots, pending writes, read pins, and logical deletion |
| `miles/rollout/session/record/backends/sqlite/backend.py` | aiosqlite connection, bounded SQL/codec execution, and temporary files |
| `miles/rollout/session/record/codec.py` | Detached canonical records and lossless JSON encoding |
| `miles/rollout/session/record/types.py` | Keys, key-only refs, errors, and backend protocol |
| `miles/rollout/session/recording.py` | Serving checkpoints and commit publication |
| `miles/rollout/session/lifecycle.py` | Active operations, cleanup generations, collect idle retention, and shutdown |

Only the three offload implementation files with `doc-dev:` sentinels bind to this page. Serving and collect callers are integration points; this page does not document general session-server behavior.

`RecordStore` accepts a backend or `None`. Its synchronous `put(session_id, record)` freezes a snapshot, returns a stable `RecordRef`, and schedules optional disk work. `delete(ref)` releases live ownership. `get_many(refs, timeout=...)` synchronously validates and pins every ref, then returns a waiting task. The backend protocol is asynchronous `put/get/delete/close`. Session policy, SQLite settings, paths, and thread management stay outside the store.

`SQLiteBackend` uses `aiosqlite==0.22.1` for its connection thread and SQL queue. Its constructor performs no I/O. Encoding and decoding use `asyncio.to_thread`, bounded together with SQL execution by a semaphore. One sequential store coroutine drains pending writes; pending records do not each enqueue a serialized copy in aiosqlite's internal queue. This bounds concurrent backend work, not total memory.

## Record identity and canonical commit

```text
SessionRecordKey = (run_id, instance_id, session_id, record_id)
```

Allocate `record_id` once for an immutable record. Never derive it from list position, assistant count, or upstream response ID, and never reuse it after rollback. Immutable `INSERT` rejects duplicate keys without replacing existing rows.

Each backend creates a unique process-incarnation directory, with a prefix derived from the run and instance identity. It never reopens an old DB. If the serving `instance_id` is `None`, setup creates one internal storage UUID shared by the store and backend; health and wire identity remain unchanged.

A serving `RecordCheckpoint` pairs the key-only ref with a detached `tools` snapshot. v1 trajectories hold ordered checkpoints; v2 nodes hold one each. Both versions call `commit_record(store, session_id, record)` under their existing gate, after model postprocessing. Inkling and ordinary models use this same entry point and codec. Inkling may change messages, finish reasons, and parser metadata, so raw upstream bytes are not the canonical record.

The helper accepts a ref without awaiting and deletes it if hot-state publication fails. v1 preserves its `closing` and `num_assistant` gates; v2 preserves the pre-generation parent for sibling completions. Hydration never reruns a tokenizer or response parser. The codec preserves the committed record, including replay fields.

## Write and failure boundary

1. **Accept in memory.** Freeze the complete canonical record and publish its serving checkpoint under the existing commit gate. A turn skipped by the gate creates no record.
2. **Write asynchronously.** The backend encodes a JSON BLOB and executes one autocommit INSERT. Chat does not await encoding or commit.
3. **Release after success.** Only successful completion of the actual backend put permits dropping the stored memory snapshot. Failed writes keep the complete snapshot.

SQLite provides transaction atomicity through one aiosqlite connection with `isolation_level=None`, verified `journal_mode=DELETE`, and `synchronous=EXTRA`. Queueing and encoding are not commit acknowledgements. Encoding failure, disk-full errors, and unknown commit outcomes retain memory without undoing a successful chat response. Attempted puts are not automatically retried; a failed record does not disable later healthy writes.

`flush()` waits for actual work retirement, including writes that failed and stayed in memory. It is not a durability acknowledgement. A successful chat response likewise does not guarantee disk durability or restart recovery.

## Read snapshots and errors

GET and samples capture ordered refs and hot metadata under the session lock and start `get_many` before releasing it. v2 also captures leaf paths, tokens, mismatch results, and descriptors. Hydration runs outside the lock. Assembly and synchronous hooks use only the captured metadata and independent returned records.

The store owns the actual read task independently of the waiter's cancellation or 30-second deadline. Source pins end only after independent copies or decoded records exist; those objects keep the result alive through encoding and hooks. Rollback and deletion cannot invalidate a started read. Physical deletion waits for actual reads and writes, and late writes cannot revive deleted refs.

Only completed SQLite BUSY/LOCKED reads may retry, once, within the original pins and read deadline. No entire samples request or hook is retried. Recognized read I/O errors and read timeouts return HTTP 503 with `error.code=session_record_unavailable`. Codec, schema, missing-record, and corruption errors remain explicit storage errors; hooks retain their existing 422 behavior. No partial sample result is returned.

## Collect handoff and cleanup

The tracer performs one samples POST with a 120-second deadline, decodes and merges the reply, then returns `CollectedSamples(reply, generation)`. The wire payload stays unchanged; a successful samples response carries `X-Miles-Session-Generation`. Only the specifically marked storage 503 joins transport errors and timeouts in the driver's ABORTED collection path. Other HTTP and decoding errors propagate.

The agentic caller completes all fallible metadata and output assembly before calling `schedule_cleanup(generation)` and returning without another await. Legal empty replies follow the same handoff rule. Collection failure, cancellation, or failed output assembly does not schedule success cleanup.

Cleanup is asynchronous and guarded. At most 32 DELETE tasks are retained per client process, each with a 30-second deadline. Missing or invalid guards, saturation, scheduling failure, or DELETE failure warn and rely on server retention. DELETE status 204, 404, or 412 ends the attempt. Automatic cleanup never falls back to unconditional DELETE.

Each accepted chat/GET/collect advances `SessionActivity.generation`; rollback, branch positioning, and a previously in-flight chat's later commit also invalidate older generations. Conditional DELETE checks the generation and zero active operations under the session lock before setting `closing`. A mismatch returns 412 without mutation. Explicit manual DELETE without the header retains its existing close semantics; malformed headers are rejected.

Collect enables idle retention at the initial snapshot, before I/O. After the last accepted operation finishes, a 300-second timer starts. New accepted activity cancels the old timer and completion rearms it. Expiry checks the session identity, current timer identity, and absence of active operations. A rejected request does not renew retention, even if rollback or positioning already advanced the generation. This covers lost replies and failed cleanup scheduling; actual store tasks continue protecting records after request cancellation or session expiry.

## Shutdown and temporary files

Setup registers core shutdown before closing the upstream HTTP client. `SessionCore.close` delegates to `SessionLifecycle`, which owns the shared shutdown task. Shutdown stops new serving work, marks sessions closing, cancels retention timers, and retires sessions under their locks. A shared shielded task then drains store work and explicitly closes the aiosqlite connection. Its 30-second waiting deadline includes session-lock waits.

A timeout returns `False` and warns, while the actual shutdown task and files remain owned. A later close can await the same task. Connection-close failure also returns `False` and preserves the namespace; it does not automatically restart a failed shutdown task. The backend removes its directory only after successful connection close. Directory-cleanup failure warns and preserves remaining files.

Failed row deletion keeps a key-only `cleanup_debt` in the backend. Normal shutdown removes the temporary namespace. SQLite may reuse freed pages; deleting rows does not promise immediate filesystem-space reclamation, and Stage 1 does not add online VACUUM.

## Verification boundary

Correctness coverage includes real SQLite commits and hydration, v1/v2 and Inkling canonical records, R3/indexer fields, memory/disk equivalence, failed writes, cancellation, rollback, late commits, guarded cleanup, retention, startup/config/argv, and shutdown. The HTTP integration uses real FastAPI/httpx and SQLite with a fake inference backend; it does not establish real-model correctness or hardware performance.

No particular CPU architecture, CUDA runtime, or NVMe device is required. The selected local filesystem must provide correct SQLite locking and sync semantics. Network filesystem support and performance are not established.

**M4 remains planned and not implemented:** real-model acceptance, RSS and latency measurements, and remaining compatibility verification. The pinned SGLang baseline lacks `convert_to_chat_completion_request`, leaving the Anthropic suite unable to collect. Mintlify rendering has not been verified. Correctness tests do not establish OOM safety or measured offload performance.
