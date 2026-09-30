---
paths:
  - "miles/**/arguments.py"
  - "miles/**/*args*.py"
  - "miles/utils/eval_config.py"
  - "miles/utils/chat_template_utils/**/*.py"
  - "miles/backends/sglang_utils/**/*.py"
  - "miles/rollout/**/*.py"
  - "miles/router/**/*.py"
  - "miles/tinker/**/*.py"
  - "miles/ray/rollout/**/*.py"
  - "examples/**/*.py"
---

# Launch And Request Args

Miles configures behavior at two scopes: launch args fixed for the whole run,
and request args sent with each sample, turn or call. Most values also pass
through code that merges defaults, launch args and request values. Follow these
rules when adding or changing a flag, a config field, a per-request field, or
code that merges them.

## Pick the scope from what the value means

- **Launch args carry run-wide behavior constraints.** Engine topology and
  features, training/rollout alignment, replay side channels, the served
  adapter and server policies are fixed when the run starts. Validate and derive
  them once at startup, and treat them as read-only afterwards.
- **Request args carry only the control that has to vary.** Sampling knobs, a
  sample's remaining token budget, per-dataset eval settings and per-turn
  template options belong to the sample, dataset, turn or call. A launch arg may
  supply their default.
- **Decide with one question.** Can two requests in the same run correctly use
  different values? If so, the field is a request field with an optional launch
  default. If the run is only correct when every request uses the same value,
  the field is a launch constraint, and request input must not change it.
- **Do not widen either side.** Do not add a launch flag because one call site
  wants a different value; pass it with the request or the dataset config. Do
  not add a request field that lets a caller change a launch constraint. Do not
  choose a value by branching on a model or dataset name in shared code; put it
  in the layer that owns that identity, such as model-specific rules, the
  dataset config or per-engine-group config.

## Respect the existing override logic

- **Extend the merge that already exists.** Before adding a field, find the code
  that already combines defaults, launch args and request values for that kind
  of value, and add the field there. Do not create a parallel path or a second
  knob for the same value.
- **Know the override order before you change it.** Every merge already decides
  which value wins when defaults, launch args and request values disagree. Read
  that logic before adding a field, and keep the new field consistent with it.
- **Default for a request field:** the request value wins, and a default applies
  only when the request leaves the field unset.
- **Default for a launch constraint:** a request cannot change it; a request
  that sets a different value gets a clear error.
- **Make every departure from these defaults explicit.** When a change makes one
  value override another, ignores input, or forces a value, including a
  miles-derived value replacing a user setting, add a comment at that spot
  stating which value wins and why, and state it in the PR description.
- **Every accepted field must reach its consumer.** A flag, config key or
  request field that parses but is never read turns a user setting into a
  silent no-op. Wire it through or refuse it.
- **Keep train and eval differences in the merge.** When evaluation relaxes or
  tightens a launch constraint, implement that where values are merged, not at
  individual call sites.
- **Never mutate launch state per request.** Do not write to the parsed args or
  to shared default dicts from a request path; copy first.
