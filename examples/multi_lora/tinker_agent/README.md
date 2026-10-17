# Agent rollouts through official Tinker capture

This independently installable client package uses the pinned official
`tinker_cookbook.capture.proxy`, SDK instrumentation, exporter and capture
store. It imports no Miles code and accepts a sampling client from either
the Miles gateway or the official Tinker service.

```bash
uv pip install ./examples/multi_lora/tinker_agent
```

Wrap the recipe in the official `capture_to_store(run_id, data_dir=...)`.
Pass its sampling client directly to `serve_proxy(renderer, sampling_client,
model_label=..., default_max_tokens=...)`. Give each agent a proxy URL with
`/r/run/<run_id>/group/<group_idx>/traj/<traj_idx>` before `/v1/...`.
For sandbox callers, choose a reachable bind interface and an auth token,
following the official proxy's bind policy.

The official proxy owns rendering, tools, stop semantics and SSE. Official
SDK instrumentation owns capture; the official store is the only sampled-token
record. There is no separate recorder, token schema or episode state machine.

After the group's trials finish, close the proxy and let aiohttp finish its
accepted handlers. `flush_capture()` flushes the official exporter and checks
its loss counters before reading the store. A reported capture loss fails
training rather than silently consuming an incomplete dataset. This is a
training-consumer check, not a change to official capture or request behavior.

`read_samples()` queries the store's public, paginated rows API by run/group/
trajectory and sorts by official call timestamps. `rows_to_trajectory()`
converts those rows to cookbook transitions without rebuilding tokens.
Rewards and trial outcomes come from the harness; cookbook decides which
exact token-prefix continuations can share a Datum. Failed requests remain
in official capture annotations and are handled by the agent/harness.
The recipe does not add sampling retries, refusal matching, parameter
overrides or partial-episode recovery.

The isolated tests exercise the real official proxy → instrumentation →
exporter → store path with a fake sampler, and validate store-row conversion
to training data. The GPU acceptance trains from actual rows in the official
store, then samples from the updated policy.
