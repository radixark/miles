---
title: Migration Guide
description: Sync → async loop, breaking flag changes between releases.
---

# Migration Guide

This page tracks breaking changes between Miles releases and how to update your code or
launch scripts.

## Train loop: sync → async

### What changed

`train.py`, `train_async.py`, and `RayTrainGroup` now use Python `async`/`await`
instead of `ray.get()`-style sync calls.

### Why

Async code is more expressive when you need to overlap work. A concrete example: in
fault tolerance we want to capture Ray actor errors and retry `actor_model.train`
while still letting `critic_model.train` proceed. That's hard to write with synchronous
`ray.get`.

### How to migrate mechanically

#### 1. Make `train` async

```diff
- def train(args):
+ async def train(args):
      ...

  if __name__ == "__main__":
-     train(parse_args())
+     asyncio.run(train(parse_args()))
```

#### 2. Drop `async_` prefixes; `ray.get(x)` → `await x`

```diff
- ray.get(group.async_init(...))
+ await group.init(...)

- ray.get(group.async_train(...))
+ await group.train(...)

- group.save_model(...)
+ await group.save_model(...)

- group.update_weights()
+ await group.update_weights()

- ray.get(rollout_manager.generate.remote(rollout_id))
+ await rollout_manager.generate.remote(rollout_id)
```

Same pattern applies to `offload`, `onload`, `clear_memory`, `connect`,
`set_rollout_manager`.

#### 3. Dispatch handles → eager tasks

```diff
- handle = critic.async_train(...)
- ray.get(actor.async_train(...))
- ray.get(handle)

+ task = await eager_create_task(critic.train(...))
+ await actor.train(...)
+ await task
```

#### 4. `create_training_models` is now async

```diff
- actor, critic = create_training_models(args, pgs, rollout_manager)
+ actor, critic = await create_training_models(args, pgs, rollout_manager)
```

## Other recent breakages

### v0.0.8 → v0.0.9

* `--rollout-router-ip` renamed to `--sglang-router-ip`.
* `--rm-path` renamed to `--custom-rm-path` (legacy alias warns then exits in v0.1.0).
* `--num-rollout` is now required (previously defaulted to 1000).

### v0.0.7 → v0.0.8

* Default `--save-interval` changed from `10` → `100`. Set explicitly if you relied on
  the old cadence.
* `--apply-chat-template` is no longer implicit when `prompt` is a list; pass it
  explicitly.

## Versioning policy

* **Patch (0.1.x):** bug fixes only, no flag changes.
* **Minor (0.x.0):** new features; old flags marked deprecated, alias kept for one minor.
* **Major (x.0.0):** breaking changes; deprecated aliases removed.

We log a clear deprecation warning at startup whenever you use a soon-to-be-removed
flag — pay attention to those, they're the cheapest way to migrate ahead of time.

## Compatibility matrix

| Miles | SGLang | Megatron-LM | Notes |
|---|---|---|---|
| 0.1.0 | 0.4.4+ | mcore-r0.10 | Current stable |
| 0.0.9 | 0.4.3 | mcore-r0.9 | |
| 0.0.8 | 0.4.2 | mcore-r0.8 | Last sync-loop release |

If you're pinned to a specific SGLang or Megatron version, use the matching Miles
release rather than mixing.
