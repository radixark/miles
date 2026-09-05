# Per-replica control proxy — staged, not deployed

The existing public ports remain 30130 (engine/control) and 30132 (generation router). Proposed engine bases are http://192.0.2.10:30130/engines/0 through /engines/3. The default path retains the existing primary engine. The inspected Miles API client appends endpoint paths to server_url, preserving these prefixes; the actual H200 controller fork still needs verification.

`nginx.pending.conf` is a candidate replacement for the 30130 socat frontend after the smoke run and pool acceptance. Do not run it during the smoke reservation. No live service or security group was modified. The nginx binary was extracted from the Ubuntu package without installation or service activation.

`test_proxy.py` tested a localhost nginx against a mock backend: rank query forwarding, POST bodies, authorization, and incremental SSE all passed. This does not prove H200 reachability or actual Miles end-to-end control. Results are in test-results.json.

## Actual Miles client validation

The downloaded Miles client (pin d2fc97ce581577e255e494801d7568747d5a10d7) now passes ten actual API methods through nginx against a mock backend: health, rank/parallelism, server info, flush, read version, pull, pause, disk reload, set version, resume. Test script: exercise_miles_client.py, invoked by test_proxy.py. These are protocol tests, not real weight changes or proof that the H200 controller accepts prefixed worker URLs everywhere.

## Disk-delta coordination audit

At this pin, protocols/delta.py `_reload_engines` pulls all replicas, pauses all, flushes (except in_place), submits all reloads concurrently, checks reload results, then resumes all. It does not explicitly call get_weight_version after reload. A failed result interrupts before resume; blindly adding finally-resume would risk serving mixed versions.

`reload_coordinator.py` is an uninstalled integration helper for the section AFTER all workers have paused/flushed. It bounds reload concurrency (default one), waits for all reload operations to settle, requires success responses, reads every version, and only then resumes. Three mock failure/ordering tests pass. Resume itself is not atomic; a transport failure during resume requires pool reconciliation. The helper must be reviewed against the actual H200 Miles fork and its response schema before integration. It does not implement pulling, router quiescence, or distributed trainer barriers.

The live 512 engine restart already enabled generation-independent health checks. No additional engine, weight, router, security-group, or frontend change was made by these investigations.
