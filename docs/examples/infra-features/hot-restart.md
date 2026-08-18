---
title: "Hot Restart Example"
description: "Replaces the orchestration script and rollout executor of a live run, keeping trainers and engines up."
# Generated from examples/infra_features/hot_restart/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Replace the orchestration script and the rollout executor of a running job without taking down
the trainers or the inference engines.

## Quick Start

Start the run under a name the same command can be relaunched with:

```bash
python examples/infra_features/hot_restart/run_qwen3_0_6b_hot_restart.py \
    --cluster-backend kubernetes --namespace <your-namespace> --run-id my-run
```

While it trains, hot restart it with the same command plus one flag:

```bash
python examples/infra_features/hot_restart/run_qwen3_0_6b_hot_restart.py \
    --cluster-backend kubernetes --namespace <your-namespace> --run-id my-run \
    --hot-restart orchestration,rollout_executor
```

## TODO

* Show a hot restart that changes the rollout executor's args and resumes under them; today the
  command is repeated verbatim.
