---
title: "Length penalties"
description: "Shape correct rollout rewards using trainable token counts."
---


Enable training reward shaping with `--length-penalty-coeff 1.0`. Only trajectories
whose selected raw reward is exactly `1.0` are penalized. For each prompt group,
let `L` be a trajectory's total loss-mask sum, `L_min` the shortest correct
trajectory's count, `m` the `--length-penalty-min-tokens` floor, and `c` the coefficient:

```text
shaped_reward = max(1, L / max(m, L_min)) ** (-c)
```

The coefficient defaults to zero (disabled) and must be finite and non-negative.
The minimum defaults to one token and must be a positive integer. Incorrect
rewards and groups with no correct trajectories are unchanged. Ties and correct
trajectories at or below the reference length keep reward `1.0`. For example,
with `c=1`, doubling the reference length gives reward `0.5`.

Shaping runs before group reward normalization. Accuracy logging and evaluation
retain raw rewards. The built-in nonzero-standard-deviation sampling filter uses
shaped rewards so all-correct groups with useful length differences can survive.
All-correct groups are mean-centered without standard-deviation scaling, preserving
the coefficient's control over length-only advantages. Mixed-correctness groups
retain the configured standard-deviation normalization. The filter requires a
shaped reward range greater than `1e-6` for all-correct groups.

Length is the sum of the final loss masks across a rollout's retained training
samples. If no mask is supplied, every response token is treated as trainable.
Prompts, tool outputs, discarded retries, truncated output, and tokens masked
off for partial rollouts do not count. In V2, shared prefixes count once through
training-mask ownership. Samples marked `remove_sample` contribute zero tokens.
Mask initialization and removal happen before reward shaping. Retained sibling
samples share one rollout length and shaped reward; custom generators only need
to supply the same loss masks that training should use.

When length shaping is enabled, training-data conversion emits one `Length penalty:` INFO log per session/rollout,
before advantage normalization. It includes `session_id`, `rollout_id`, the total
`generated_tokens` counted toward shaping (the training loss-mask sum),
`multiplier` (`1` means no penalty), `reward_reduction`, `raw_reward`, and
`shaped_reward`. Incorrect sessions report multiplier `1`. Disabled shaping emits no penalty logs and skips length counting.
Candidate filtering does not emit applied-penalty logs.

Nonzero length shaping cannot be combined with custom reward postprocessing or
custom train-data conversion, since those hooks replace the built-in pipeline.
Custom filters must account for shaped rewards themselves if they discard groups
based on raw reward variance.
