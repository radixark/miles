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

Rendering uses mandatory fact placeholders replaced deterministically, followed
by a separate Luna call solving the rendered case without access to the answer.
Disagreements are retried up to four times and retained for inspection. A failed
quota prevents publication. Resume by rerunning the same command; accepted IDs
are preserved. Credential errors retain only exception types.

This pilot has seven fixed scenario families, not broad organic business data.
Train and validation have distinct seeds and records but share rule families.
The reviewer uses the same model and does not establish human-level label
quality. Tool labels model catalog capabilities/preconditions, not real API
execution. No benchmark questions are inputs; semantic decontamination and
step-2048 error mining require separate checks before making benchmark claims.
