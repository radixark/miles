---
title: Streaming Reward and Deserialization
description: How rollout responses stream through msgpack transport, a Ray parser-actor pool, and per-microgroup async reward scoring.
---

Rollout in miles-diffusion is not "generate everything, then score everything". Each microgroup is generated,
deserialized, and scored as an independent asyncio task, so reward computation and tensor decoding overlap with
generation that is still in flight. This page covers the three stages of that pipeline and the knobs that keep each one
off the critical path.

## 1. The per-microgroup pipeline

`miles/rollout/sglang_diffusion_rollout.py` runs, per microgroup:

```mermaid
flowchart LR
    G[generate<br/>POST /rollout/generate] --> D[deserialize<br/>parser actor pool]
    D --> R[reward<br/>rm_hub actor pool]
```



The rollout pipeline decomposes into `generate` / `deserialize` / `reward` components. Because every microgroup is its
own `asyncio` task, the engine starts the next request while previous responses are still being unpacked and scored.
Nothing waits for the full batch.

## 2. Deserialization: msgpack + parser-actor pool

Trajectory tensors are large — for video models the response for one microgroup can be gigabytes, and encoding tensors
into base64 further increases the size. In a naive implementation, tensors were base64-encoded inside a JSON body and
parsed on the main asyncio event loop, one sample at a time.

The current path:

- **msgpack raw-bytes transport.** The engine responds with `application/msgpack`; `post(..., raw=True)` returns the
  body untouched, and tensors decode directly from safetensors raw bytes — no base64.
- **Unpacking runs inside Ray actors, not the event loop.** `RolloutImageResponseParserActor.apply_raw(samples, raw)`
  does `msgpack.unpackb` + tensor decode in a separate process (`miles/utils/diffusion_rollout_response.py`), so the
  rollout event loop never blocks on a multi-GB unpack.
- **One call per microgroup.** A whole microgroup is parsed in a single `apply_raw` call — fewer Ray RPCs, one unpack
  per response.
- **A pool, round-robin dispatched.** `--rollout-parser-num-workers N` spins up N parser actors so multiple microgroups
  deserialize in parallel.

Measured on the LTX-2.3 recipe (H200, same config, averaged over two consecutive RL steps, base64/JSON vs the current
path):


| Metric              | base64/JSON | msgpack + pool | Speedup            |
| ------------------- | ----------- | -------------- | ------------------ |
| `perf/rollout_time` | 157.4 s     | 87.6 s         | **~1.8× (−44 %)**  |
| `perf/step_time`    | 321.9 s     | 252.1 s        | **~1.28× (−22 %)** |




## 3. Reward: scored as soon as a microgroup lands

`generate_and_rm_microgroup` calls `batched_async_rm` immediately after parsing, per microgroup — not once at the end of
the rollout:

```python
microgroup = await generate_microgroup(...)   # generate + deserialize
rewards    = await batched_async_rm(args, microgroup)   # score right away
```

Reward workers are Ray actor pools (see [Rewards](/diffusion/user-guide/rewards) for per-reward flags), so scoring one microgroup
overlaps with generation and deserialization of the others.

## 4. Diagnosing the pipeline

The miles dashboard visualizes the lifetime of every request — the generate / deserialize / reward spans of each
microgroup on one timeline, so overlap (or the lack of it) is visible directly. Launch with `--use-miles-dashboard`
(telemetry lands under `--miles-dashboard-workspace`), then render:

```bash
python -m miles.dashboard.viewer --workspace ./miles_dashboard --out dash.html
```

For quick triage from `perf/*` metrics alone:

| Symptom                                                 | Look at                  | Likely fix                                                                 |
| ------------------------------------------------------- | ------------------------ | -------------------------------------------------------------------------- |
| `perf/rollout_time` high, engines idle between requests | `deserialize` stage time | Raise `--rollout-parser-num-workers`                                       |
| Rollout stalls at the end of each iteration             | `reward` stage time      | More reward workers, or a dedicated reward GPU                             |
| Event loop warnings / slow heartbeat                    | main-process CPU         | Confirm parsing is going through the actor pool (it always does on `main`) |

