# Retool v2

This example is an upgraded version of the original retool example, using the updated interfaces provided by the miles framework to implement multi-turn RL training with tool calls in a cleaner way.

## Key Differences from v1

**v1 (retool)** requires manually implementing the full multi-turn conversation loop in `generate_with_retool.py`, directly depending on low-level `GenerateState` and `sglang_rollout` interfaces — resulting in verbose code tightly coupled to the framework internals.

**v2 (retool_v2)** uses the framework's standard plugin interfaces. Users only need to implement three functions and mount them via command-line arguments:

| Argument | Description |
|----------|-------------|
| `--custom-generate-function-path` | Uses the built-in `miles.rollout.generate_hub.multi_turn.generate` — no need to implement the multi-turn loop yourself |
| `--generate-tool-specs-path` | Declare tool definitions (user-implemented) |
| `--generate-execute-tool-function-path` | Implement tool execution logic (user-implemented) |
| `--custom-rm-path` | Implement the reward function (user-implemented) |

Users only need to focus on business logic (tool definitions, tool execution, reward calculation). Multi-turn scheduling, token concatenation, loss masking, etc. are all handled by the framework.

## Files

- `tool_sandbox.py`: Tool definitions (`tool_specs`), tool execution (`execute_tool`), reward function (`reward_func`), and sandboxed safe execution environment
- `run_retool_multi_turn.py`: Training launch script

## Quick Start

```bash
python examples/retool_v2/run_retool_multi_turn.py
```

The launch script prepares everything it needs on its own: it downloads the dapo-math-17k
training set and the aime-2024 eval set, downloads the checkpoint, and converts it to
`torch_dist` before training starts.

### Single GPU, fully async and colocated

```bash
python examples/retool_v2/run_retool_multi_turn.py --fully-async --num-gpus-per-node 1
```

`--fully-async` switches the run to `train_async.py`, where the rollout engines keep
generating between training steps and share the GPUs with the trainer; see
[fully async](https://github.com/radixark/miles/blob/main/docs/user-guide/fully-async.md).
