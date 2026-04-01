# Migration Guide

## Async Train Loop (since feat/refactor_dp/7)

`RayTrainGroup` and the train loops are now async. Two mechanical changes:

### 1. Make the train function async

```python
# Before                          # After
def train(args):                  async def train(args):
    ...                               ...

if __name__ == "__main__":        if __name__ == "__main__":
    train(parse_args())               asyncio.run(train(parse_args()))
```

### 2. Apply two rules to every call

**Rule A — Group methods:** drop the `async_` prefix, add `await`.

```python
ray.get(group.async_init(...))  →  await group.init(...)
ray.get(group.async_train(...)) →  await group.train(...)
group.save_model(...)           →  await group.save_model(...)
group.update_weights()          →  await group.update_weights()
# Same for offload, onload, clear_memory, connect, set_rollout_manager
```

**Rule B — Ray ObjectRef:** replace `ray.get(ref)` with `await ref`.

```python
ray.get(rollout_manager.generate.remote(id))  →  await rollout_manager.generate.remote(id)
```

### Actor/critic parallelism

The old pattern dispatched Ray RPCs eagerly via `async_train` (sync function returning ObjectRefs). The new equivalent uses `asyncio.create_task` to run the critic coroutine concurrently:

```python
# Before
handle = critic.async_train(...)        # .remote() calls dispatched immediately
ray.get(actor.async_train(...))
ray.get(handle)

# After
task = asyncio.create_task(critic.train(...))  # coroutine scheduled, runs on next yield
await actor.train(...)                          # actor runs, critic starts at first await
await task
```

### Setup

`create_training_models` is now async:

```python
actor, critic = await create_training_models(args, pgs, rollout_manager)
```
