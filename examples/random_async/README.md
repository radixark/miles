# Random fully-async example

Minimal sibling of `examples/fully_async/`. Exercises the entire async
rollout ↔ trainer loop **without any real dataset, real reward model, or
meaningful generation** — useful as an agent infrastructure stress test 
for bigger agentic workloads.

## Quick start

```bash
# default (Qwen3.5-35B-A3B), in_place pause + broadcast weight transfer
python run_random_async_3node.py

# retract pause + p2p weight transfer
python run_random_async_3node.py \
    --pause-generation-mode retract \
    --update-weight-transfer-mode p2p

# swap in a different model
python run_random_async_3node.py \
    --model-name Qwen3.5-35B-A3B --megatron-model-type qwen3.5-35B-A3B
```

## Notes

- `--disable-rollout-global-dataset` is on, so no `--prompt-data` file is
  required. The rollout function ignores the data buffer and constructs
  `Sample` objects from scratch.
- The rollout uses Qwen3.5-35B's vocab size (151643) for the random
  `input_ids`; any model with vocab ≥ that works without changes.
- `ignore_eos=True` in the sampling params means SGLang generates until
  it hits `max_new_tokens` (drawn from `MAX_TOKENS_RANGE`). `Sample.status`
  is set to `COMPLETED`.
