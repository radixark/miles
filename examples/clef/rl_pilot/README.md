# Clef RL pilot data

Generate native Clef choice records with deterministic one-hot targets and GPT-6
Luna document framing. API credentials are read from a protected file, never
stored in outputs. Run on a devbox, not a workstation.

```console
uv run --project examples/clef/rl_pilot examples/clef/rl_pilot/generate.py --output /scratch/clef-rl-pilot
```

Default: 2,048 training cases and 256 validation cases, shuffled separately.
Workflow cases comprise 50%, tool selection 25%, retrieval 12.5%, and extraction
12.5%. Every case retains rules, canonical facts, exact answers, generated text,
API usage and blind reviewer outputs. Labels never enter the model input.

Rendering uses explicit fact assignments inserted verbatim by code, followed
by a separate Luna call solving the rendered case without access to the answer.
Disagreements are retried up to four times and retained for inspection. A failed
quota prevents publication. Resume by rerunning the same command; accepted IDs
are preserved. Credential errors retain only exception types.

This pilot has seven fixed scenario families, not broad organic business data.
Train and validation have distinct seeds and records and disjoint semantic
fact/policy groups, ignoring arbitrary case IDs, but share rule families.
Several rendered examples can instantiate the same semantic scenario within a
split; the validation report records the effective unique scenario count.
The reviewer uses the same model and does not establish human-level label
quality. Tool labels model catalog capabilities/preconditions, not real API
execution. No benchmark questions are inputs; semantic decontamination and
step-2048 error mining require separate checks before making benchmark claims.

After generation, run `uv run --project examples/clef/rl_pilot
examples/clef/rl_pilot/validate.py --data /scratch/clef-rl-pilot` to regenerate
every label and probe the exact field and whole-record reward functions.

`probe.py` runs in an existing Miles/SGLang environment with the repository on
`PYTHONPATH`; it checks all records through the actual Clef tokenizer/encoder.
Pass `--endpoint` to measure checkpoint difficulty, Brier loss, exact record
success, and near-one-hot fields. It verifies the endpoint's model path first.
No RL training is performed by these scripts.
# Hard pilot

`python -m examples.clef.rl_pilot.hard --output /path/to/data` generates 2,048
training and 256 validation cases using GPT-6 Luna. Labels come from an
executable revision resolver and business rules; the model supplies document
framing and a blind answer audit with medium reasoning effort. Model training
still consumes the decision head directly and generates no reasoning.

The six families are invoice, support, security, transfer, tool choice, and
policy retrieval. Records include signed revisions, unsigned proposals, future
changes, and unrelated entities. Policies combine arithmetic, inclusive/exclusive
boundaries, exceptions, and competing reasons with explicit priorities. Rare
positive branches are deliberately represented to avoid trivial all-no fields.
Options are shuffled; labels and provenance remain outside the encoded state.

Validation uses new scenarios plus additional policy-clause compositions.
It is frozen before evaluation, not selected based on model mistakes. Task
families and revision-resolution rules remain shared between splits. This is
a controlled synthetic pilot, not an organic workflow benchmark.

`python -m examples.clef.rl_pilot.hard_check` executes independent rule-boundary
fixtures and checks all 2,304 generated schemas. Finalization rechecks exact
targets, fact preservation, reviewer agreement, unique scenarios, and split
separation. `--canonical-only` skips API rendering for development probes;
it must not be confused with the reviewed final dataset.
