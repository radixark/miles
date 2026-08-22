---
title: "Multi Policy Solver and Verifier"
description: "Two policies in one run — a solver answering gsm8k, and a verifier scored on ruling correctly about the solver's answers."
# Generated from examples/multi_policy/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Two policies trained against each other in one run: a solver answering a gsm8k question and a
verifier ruling on its work. Each rollout yields one sample per policy; each policy has its own
trainer and its own inference engines inside the same job.

## Quick Start

Eight GPUs: 2 trainer GPUs and 2 single-GPU engines per policy. Needs a cluster backend and a
namespace, from `MILES_SCRIPT_CLUSTER_BACKEND` / `MILES_SCRIPT_NAMESPACE` or `--cluster-backend` /
`--namespace`:

```bash
python examples/multi_policy/run_solver_verifier_gsm8k.py
```

## Failure Propagation

The first policy task to finish ends the run and cancels the other policy tasks. If that
task failed, its exception remains the run's primary error. Failures raised while the
other tasks clean up are logged; runtimes with `BaseException.add_note` also attach their
tracebacks to the primary exception as notes. Python 3.10 has no `add_note`, so the log is
the secondary failure record there. If the first task finished normally, a cleanup failure
still fails the run.
