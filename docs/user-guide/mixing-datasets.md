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
    max_response_len: 16384
    custom_generate_function_path: my_pkg.science_generate
    agent:
      max_turns: 50
```

The flag also accepts an inline `base64:<payload>`, like `--eval-config`.

### Fields

| Field | Required | Meaning |
|---|---|---|
| `path` | yes | Prompt file, read with the run's `--input-key`, `--label-key`, `--metadata-key` and `--tool-key`. |
| `samples_per_step` | yes | Weight of this dataset in every request for prompts. |
| `sha256` | no | First 8 hex characters of the file's SHA-256. A mismatch stops the run at startup; when unset, the value to pin is logged. |
| `max_prompt_len` | no | Drop this dataset's longer prompts. At most `--rollout-max-prompt-len`. |
| `max_response_len` | no | Response budget (`max_new_tokens`) for this dataset's samples, also sent to an agentic dataset's model calls. At most `--rollout-max-response-len`. |
| `custom_generate_function_path` | no | Generate function for this dataset's samples. Unset: `--custom-generate-function-path`, or Miles' default `miles.rollout.generate_hub.single_turn.generate` when that is unset too. |
| `custom_agent_function_path` | no | Agent function for `agentic_tool_call` on this dataset, instead of `--custom-agent-function-path`. |
| `max_seq_len` | no | Agentic sequence limit for this dataset. At most `--max-seq-len`. |
| `agent` | no | Free-form settings for this dataset's agent, passed as `metadata["agent_config"]`. |
| `custom_rm_path` | no | Reward function for this dataset's samples, instead of `--custom-rm-path`. |
| `rm_type` | no | Built-in reward for this dataset's samples, instead of `--rm-type`. |

Unknown fields are rejected, so a typo fails at startup instead of being ignored. A `custom_generate_function_path` or
`custom_agent_function_path` that declares its own options (`config_class` or `add_arguments`) gets them registered as
flags, just like one passed through `--custom-generate-function-path`, so two agentic datasets can each run
their own generate function and agent. These flags live in one run-wide namespace: two hooks that declare the
same flag identically share it, so every dataset using those hooks sees the same value. A hook that re-declares
only some of another hook's flags, or declares one differently, is rejected at startup. To give two datasets
different values for an option, give their hooks distinct flags (or put the value in the dataset's `agent`
block).

`custom_agent_function_path` and `max_seq_len` are read only by `miles.rollout.generate_hub.agentic_tool_call.generate`, or by
a wrapper around it that sets `reads_sample_agent_fields = True` on itself. A dataset that sets them under any
other generate function, or an agentic dataset with neither `custom_agent_function_path` nor `--custom-agent-function-path`,
is rejected at startup. Eval samples do not come from the mixer, so an agentic eval still needs
`--custom-agent-function-path`.

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

### Resuming with different datasets

A resumed run continues every dataset from its saved position, so `--load` refuses a YAML that lists other
datasets, a changed file, a dataset filtered shorter than its saved position, or any other changed dataset field
(groups drawn before the save still carry the old generate function, agent or reward). Changing only the
`samples_per_step` weights is fine. To switch datasets mid-training, add `--rollout-data-reset-on-load`: the
model, optimizer, rollout step and sample ids still resume from `--load`, while every dataset starts from its
beginning and the prompt groups and untrained batches drawn from the old datasets are dropped. The reset happens
only when the checkpoint does not match the current YAML, so the flag can stay in the launch command: later
resumes from the run's own checkpoints continue in place. The same flag moves a run started with `--prompt-data`
onto a mix. The other way needs no flag: a mix resumed under `--prompt-data` starts that dataset from its first
prompt, with the same drops and the same sample ids.

Every sample carries `metadata["dataset_name"]`, and the rollout metrics add
`rollout/dataset/<name>/{episode_raw_reward,response_len/mean,truncated_ratio,num_training_samples}`.

### Limits

- Set `--num-rollout`: an epoch has no single length across datasets that wrap independently, so
  `--num-epoch` is not supported.
- `--partial-rollout` and a custom `--data-source-path` are not supported.
- The ratio holds for the prompts drawn. When dynamic sampling filters out groups, the refill is drawn by
  the same weights, not from the dataset whose groups were dropped.
