---
title: Data & Datasets
description: Prompt JSONL format, label / metadata schemas, multi-source datasets, on-the-fly filters.
---

# Data & Datasets

Miles consumes JSONL files. There is no built-in dataset registry — you point at a path,
specify which keys to map, and you're done.

## Minimal schema

A single line of `prompt-data` looks like this:

```json
{
  "prompt": "Solve: 4 * (3 + 2) - 7",
  "label":  "13"
}
```

You then tell Miles which keys to use:

```bash
ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
)
```

`--input-key` becomes `Sample.prompt` and `--label-key` becomes `Sample.label` (used by
the reward model).

## Chat-format prompts

If your prompts already include role / content (multi-turn), use `--apply-chat-template`:

```json
{
  "prompt": [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user",   "content": "Solve 12 / 4 + 1"}
  ],
  "label": "4"
}
```

```bash
ROLLOUT_ARGS+=(--apply-chat-template)
```

Miles then runs the tokenizer's chat template before the rollout.

!!! warning "Don't pre-template your data"
    A common bug: data already includes `<|im_start|>...` and you also pass
    `--apply-chat-template`. The model sees the template *twice*, generates garbled output,
    and your gradient norm explodes. Pick one.

## Carrying extra context with `metadata`

For multi-turn / agentic / retrieval workflows you usually need more than `prompt + label`.
Stuff everything else into a `metadata` column:

```json
{
  "prompt": "What's the capital of France?",
  "label":  "Paris",
  "metadata": {
    "session_id": "sess_42",
    "tool_code": "search_engine_v3",
    "user_profile": {"locale": "en-GB"}
  }
}
```

```bash
ROLLOUT_ARGS+=(--metadata-key metadata)
```

In your custom rollout / reward function you'll see `sample.metadata` as a dict.

## Multiple data sources

You can pass multiple datasets, each with a name:

```bash
--prompt-data \
   math   /data/math.jsonl \
   coding /data/coding.jsonl \
   chat   /data/chat.jsonl
```

The names propagate into wandb / logs, so you can inspect per-source reward and loss.

## Dataset filters

Apply runtime filtering with one of:

| Flag | Type | When it runs |
|---|---|---|
| `--dynamic-sampling-filter-path` | per-group | After scoring, before queueing for training. |
| `--buffer-filter-path` | per-rollout | When the trainer pulls from the rollout buffer. |
| `--rollout-sample-filter-path` | per-sample | Before loss computation. |

Example: drop sample groups where every response got the same reward (DAPO-style).

```python
# miles/rollout/filter_hub/dynamic_sampling_filters.py
def check_reward_nonzero_std(args, samples, **kwargs):
    rewards = [s.reward for s in samples]
    return torch.tensor(rewards).std() > 0.0
```

```bash
--dynamic-sampling-filter-path \
   miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

## Eval datasets

Eval reuses the same JSONL format. You usually want one or more dedicated eval files:

```bash
EVAL_ARGS=(
   --eval-prompt-data \
      aime  /root/aime-2024/aime-2024.jsonl \
      gsm8k /root/gsm8k/test.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-temperature 0.0
)
```

Eval is run every `--eval-interval` rollouts.

## Loading non-JSONL data

For arbitrary sources (Parquet, S3 bucket, internal API), implement a `data-source-path`:

```python
# my_data_source.py
from miles.data import Sample

class MyDataSource:
    def __init__(self, args): ...
    def __len__(self): return self.n
    def __getitem__(self, idx) -> Sample:
        ...
        return Sample(prompt=p, label=l, metadata=m)
```

```bash
--data-source-path my_data_source.MyDataSource
```

This gives you full control over how prompts are loaded, sharded, and shuffled.
