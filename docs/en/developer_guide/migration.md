# Migration Guide

## Train Loop: from Sync to Async

### What is Changed

The train loop (`train.py`, `train_async.py`) and `RayTrainGroup` now use Python async/await instead of sync `ray.get()`.

### Why it is Changed


### How to migrate

**1. Make the train function async:**

```python
# Before                          # After
def train(args):                  async def train(args):
    ...                               ...

if __name__ == "__main__":        if __name__ == "__main__":
    train(parse_args())               asyncio.run(train(parse_args()))
```

**2. Apply two mechanical rules to every call:**

Rule A — `RayTrainGroup` methods: drop the `async_` prefix, add `await`.
```python
ray.get(group.async_init(...))  →  await group.init(...)
ray.get(group.async_train(...)) →  await group.train(...)
group.save_model(...)           →  await group.save_model(...)
group.update_weights()          →  await group.update_weights()
# Same for offload, onload, clear_memory, connect, set_rollout_manager
```

Rule B — `ray.get(ref)` on Ray ObjectRef: replace with `await ref`.
```python
ray.get(rollout_manager.generate.remote(id))  →  await rollout_manager.generate.remote(id)
```

**3. Actor/critic parallelism:** replace the dispatch-handle pattern with `eager_create_task` + `await`.

```python
# Before
handle = critic.async_train(...)        # dispatches .remote() immediately
ray.get(actor.async_train(...))         # dispatches + waits
ray.get(handle)                         # waits for critic

# After
from miles.utils.async_utils import eager_create_task
task = await eager_create_task(critic.train(...))  # dispatches .remote() eagerly
await actor.train(...)
await task
```

`eager_create_task` is used instead of bare `asyncio.create_task` to ensure the critic's Ray RPCs are dispatched before the actor starts — needed because actor and critic participate in the same distributed collective (`sync_actor_critic_data`).

**4. `create_training_models` is now async:**

```python
actor, critic = await create_training_models(args, pgs, rollout_manager)
```
