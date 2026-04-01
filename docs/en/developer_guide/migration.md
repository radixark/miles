# Migration Guide

## Async Train Loop (since feat/refactor_dp/7)

`RayTrainGroup` methods and the train loop are now fully async. If you have custom train loops based on `train.py` or `train_async.py`, apply the following changes.

### Entry Point

```python
# Before
def train(args):
    ...

if __name__ == "__main__":
    train(parse_args())

# After
async def train(args):
    ...

if __name__ == "__main__":
    asyncio.run(train(parse_args()))
```

### RayTrainGroup API

| Before (sync) | After (async) |
|---|---|
| `refs = group.async_init(args, role, ...)` | `results = await group.init(args, role, ...)` |
| `ray.get(refs)` | *(already awaited inside `init`)* |
| `refs = group.async_train(rollout_id, data)` | `await group.train(rollout_id, data)` |
| `ray.get(refs)` | *(already awaited inside `train`)* |
| `group.save_model(rollout_id)` | `await group.save_model(rollout_id)` |
| `group.update_weights()` | `await group.update_weights()` |
| `group.onload()` | `await group.onload()` |
| `group.offload()` | `await group.offload()` |
| `group.connect(critic)` | `await group.connect(critic)` |
| `group.set_rollout_manager(mgr)` | `await group.set_rollout_manager(mgr)` |

### Ray ObjectRef

`ray.get(ref)` becomes `await ref` (Ray ObjectRef is natively awaitable):

```python
# Before
rollout_data = ray.get(rollout_manager.generate.remote(rollout_id))

# After
rollout_data = await rollout_manager.generate.remote(rollout_id)
```

### Actor/Critic Parallel Training

```python
# Before
critic_handle = critic_model.async_train(rollout_id, data)
ray.get(actor_model.async_train(rollout_id, data))
ray.get(critic_handle)

# After
critic_task = asyncio.create_task(critic_model.train(rollout_id, data))
await actor_model.train(rollout_id, data)
await critic_task
```

### Setup

`create_training_models` is now async:

```python
# Before
actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

# After
actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)
```
