---
title: Mixing Datasets
description: Train on several prompt datasets at once with --rollout-data-yaml-config, with a fixed share of every rollout step per dataset.
---
`--rollout-data-yaml-config` replaces `--prompt-data` when a run trains on **several datasets at once**,
for example math, science QA and agentic tasks. Miles draws each rollout step from every dataset in a
**fixed ratio**, so there is no need to merge and shuffle the files by hand.

```yaml
datasets:
  math:
    path: /data/math.jsonl
    sha256: "9f2c0e41"  # quoted, so YAML never reads a prefix like 00001234 as a number
    samples_per_step: 3
  science:
    path: /data/science.jsonl
    samples_per_step: 5
    max_prompt_len: 8192
```

The flag also accepts an inline `base64:<payload>`, like `--eval-config`.

### Fields

| Field | Required | Meaning |
|---|---|---|
| `path` | yes | Prompt file, read with the run's `--input-key`, `--label-key`, `--metadata-key` and `--tool-key`. |
| `samples_per_step` | yes | Weight of this dataset in every request for prompts. |
| `sha256` | no | First 8 hex characters of the file's SHA-256. A mismatch stops the run at startup; when unset, the value to pin is logged. |
| `max_prompt_len` | no | Drop this dataset's longer prompts. At most `--rollout-max-prompt-len`. |
| `custom_rm_path` | no | Reward function for this dataset's samples, instead of `--custom-rm-path`. |
| `rm_type` | no | Built-in reward for this dataset's samples, instead of `--rm-type`. |

Unknown fields are rejected, so a typo fails at startup instead of being ignored.

`custom_rm_path` and `rm_type` become the samples' `reward_spec`, so they follow the per-sample reward rules in
[Customization](customization.md), checked at startup: a function with its own config must be the
`--custom-rm-path` one, and under `--group-rm` a batched `--custom-rm-path` scores every group, so a dataset that
sets `rm_type` or names a different function is rejected. A custom reward function also outranks any `rm_type`,
so with `--custom-rm-path` set, a dataset's `rm_type` needs its own `custom_rm_path` too.

### How prompts are drawn

- **Weights, not counts.** `samples_per_step` values need not add up to `--rollout-batch-size`. Every
  request is split by weight, and the remainder carries over to the next request, so the long-run ratio is
  exact even when fully async rollout asks for one prompt at a time.
- **Mixed by prompt.** Datasets are mixed before each prompt is copied `--n-samples-per-prompt` times, so
  every group still holds the responses to a single prompt.
- **Separate epochs.** Each dataset has its own position, epoch and shuffle, and wraps around on its own.
  The checkpoint saves all of them, and resuming also re-checks every file's SHA-256.

Every sample carries `metadata["dataset_name"]`.

### Limits

- Set `--num-rollout`: an epoch has no single length across datasets that wrap independently, so
  `--num-epoch` is not supported.
- `--partial-rollout` and a custom `--data-source-path` are not supported.
- The ratio holds for the prompts drawn. When dynamic sampling filters out groups, the refill is drawn by
  the same weights, not from the dataset whose groups were dropped.
