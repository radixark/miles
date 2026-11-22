# Terminal-Bench 2 training with Harbor and Tinker

This recipe trains a LoRA policy on Terminal-Bench 2 (TB2) with the official Tinker cookbook RL
loop and the Miles Tinker gateway. [Harbor](https://github.com/harbor-framework/harbor) runs
each task's agent in a sandbox. The agent samples through the Tinker capture proxy described in
[Agentic training with Tinker capture](../tinker_agent/README.md), and the task's verifier
scores each trial. The cookbook computes group advantages and applies PPO to the captured
tokens.

The recipe accepts any dataset in Harbor's native task format; TB2 is the reference
configuration.

## Requirements

- **Gateway.** An 8-GPU node running the Miles Tinker gateway: 4 training and 4 sampling
  GPUs. The reference configuration serves Qwen3.6-35B-A3B on H200 with a 65536-token window
  (see [Setup](#setup)).
- **Client.** Python 3.12 or newer on the machine that runs the recipe.
- **Sandboxes.** An E2B account and API key; see
  [Sandbox Providers](../../../docs/user-guide/sandbox-providers.md). The sandboxes must be able
  to reach the client machine, which hosts the proxy.

## Setup

### 1. Start the gateway

Download the checkpoint and serve it with the gateway launcher, overriding the model, LoRA
layout and context window:

```bash
hf download Qwen/Qwen3.6-35B-A3B --local-dir /root/models/Qwen3.6-35B-A3B

python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve \
  --hf-checkpoint /root/models/Qwen3.6-35B-A3B --model-type qwen3.6-35B-A3B_lora \
  --lora-rank 16 --lora-alpha 32 --n-adapters 1 --target-modules attn,mlp,unembed \
  --extra-args "--tinker-base-model Qwen/Qwen3.6-35B-A3B \
    --max-tokens-per-gpu 65536 --max-position-embeddings 65536 --sglang-context-length 65536 \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --qkv-format thd"
```

- `--tinker-base-model` is the model name clients pass as `config.model_name`.
- `--target-modules attn,mlp,unembed` matches the cookbook client's default LoRA layout, which
  trains the output head (`train_unembed=True`). The client's LoRA rank (16) must not exceed
  `--lora-rank`.
- `--max-tokens-per-gpu 65536` and `--sglang-context-length 65536` set the trainer and inference
  limits; with context-parallel size 1, the sampling window is 65529 tokens. See
  [Context window](../README.md#context-window). Tensor parallelism does not raise the
  per-sequence limit.
- The gateway serves on port `10613`.

### 2. Install the client

From a Miles checkout:

```bash
uv pip install './examples/multi_lora/tinker_agent[harbor]' \
  'harbor[e2b] @ git+https://github.com/harbor-framework/harbor@917b3c1be7cf6553a7a9fe25de8adb6f8e639287'
```

The `harbor` extra pins the tested Harbor revision; `harbor[e2b]` adds the E2B provider at the
same revision.

### 3. Download the dataset

```bash
harbor datasets download terminal-bench@2.0 --output-dir /data/harbor-tasks
export TB2_TASKS_DIR=/data/harbor-tasks/terminal-bench
```

The recipe loads every directory under `TB2_TASKS_DIR` that contains a `task.toml`, in sorted
order. Each task keeps its instruction, environment and verifier.

### 4. Write the trial configuration

`trial.json` is a Harbor `TrialConfig` without the `task` field; the recipe supplies each task.
The recipe substitutes two placeholders: `{proxy_url}`, the trajectory's proxy URL, and
`{proxy_api_key}`, the proxy's auth token. The reference configuration runs mini-swe-agent on
E2B:

```json
{
  "trials_dir": "/tmp/harbor-trials",
  "environment": {"type": "e2b"},
  "agent": {
    "name": "mini-swe-agent",
    "model_name": "openai/policy",
    "override_timeout_sec": 5400,
    "env": {
      "OPENAI_BASE_URL": "{proxy_url}/v1",
      "OPENAI_API_KEY": "{proxy_api_key}",
      "MSWEA_COST_TRACKING": "ignore_errors"
    },
    "kwargs": {
      "max_tokens": 8192,
      "max_seq_len": 65536,
      "config": {"model": {"model_kwargs": {"temperature": 0.8}}}
    }
  }
}
```

For `claude-code`, set `agent.env.ANTHROPIC_BASE_URL` to `{proxy_url}` and
`agent.env.ANTHROPIC_API_KEY` to `{proxy_api_key}`. All other fields, including sandbox
credentials, timeouts and verifier settings, follow Harbor's schema.

## Run

```bash
export TINKER_API_KEY=tml-<tenant>
export E2B_API_KEY=<e2b-api-key>
export HARBOR_PROXY_HOST=<client-address-reachable-from-sandboxes>

python -m tinker_agent.train \
  config.base_url=http://<gateway-host>:10613 \
  config.model_name=Qwen/Qwen3.6-35B-A3B config.renderer_name=qwen3_5 \
  config.lora_rank=16 config.learning_rate=1e-6 config.max_tokens=8192 config.loss_fn=ppo \
  config.ttl_seconds=None config.log_path=/tmp/harbor-tinker \
  config.dataset_builder.tasks_dir="$TB2_TASKS_DIR" \
  config.dataset_builder.groups_per_batch=4 config.dataset_builder.group_size=8 \
  harbor.trial_config=trial.json \
  harbor.listen_host=0.0.0.0 harbor.advertised_host="$HARBOR_PROXY_HOST"
```

Each batch runs `groups_per_batch × group_size` trials: 4 tasks with 8 trajectories each.
Add `config.max_steps=1` for a single-batch run. Omit `config.base_url` to train on the hosted
Tinker service instead of the Miles gateway.

## Configuration

`config` is the cookbook's `tinker_cookbook.rl.train.Config`, with its native field names and
defaults. Run `python -m tinker_agent.train --help` for the full list.

| Option | Default | Description |
|---|---|---|
| `config.dataset_builder.tasks_dir` | required | Directory of Harbor task directories. |
| `config.dataset_builder.groups_per_batch` | `4` | Tasks per batch. |
| `config.dataset_builder.group_size` | `4` | Trajectories per task; advantages are computed within a group. |
| `config.dataset_builder.epochs` | `1` | Passes over the dataset. |
| `harbor.trial_config` | required | Path to `trial.json`. |
| `harbor.reward_key` | `reward` | Key in the verifier's rewards used as the trial reward. |
| `harbor.listen_host` | `127.0.0.1` | Address the proxy binds to. Use `0.0.0.0` for remote sandboxes. |
| `harbor.advertised_host` | `127.0.0.1` | Address placed in `{proxy_url}`; it must be reachable from the sandboxes. |
| `harbor.max_parallel_trials_per_group` | `4` | Concurrent trials per group. |

The recipe enforces the following:

- `config.rollout_error_tolerance` is set by the recipe; setting it is rejected.
- `config.temperature` must remain `1.0`. The agent issues its own sampling requests, so the
  sampling temperature is set in `trial.json`.
- `config.max_tokens` is the proxy's default for requests that do not set `max_tokens`.
- `config.renderer_name`, if omitted, resolves to the cookbook's recommended renderer for
  `config.model_name`. Qwen3.6 uses `qwen3_5`.
- `config.ttl_seconds=None` is required with the Miles gateway, which does not implement
  checkpoint expiry.

Each group uses a fresh proxy with a random auth token. Rendering and capture run on the client
machine; captured data is stored under `<log_path>/capture`.

## Limitations

- **Trial outcomes.** A completed trial, an agent timeout, a context-window overflow and an
  output-token limit are scored with the verifier's reward. Any other Harbor failure, a missing
  or non-finite reward, a capture loss or a sample without log-probs raises: it cancels the
  group's remaining trials and stops training. Infrastructure errors are never scored as zero.
- **Failed requests.** A failed model call produces no training sample; the agent and Harbor
  decide whether the trial continues.
- **Algorithm.** Training is Tinker LoRA with the cookbook's PPO and group advantages. The agent,
  sandbox, batch size, sampling temperature, token limits and learning rate match the
  [cloud Harbor example](../../experimental/harbor/README.md), but its full-parameter GRPO, optimizer defaults and KL behavior differ, so
  results are not directly comparable.
- **Recovery.** The recipe provides no partial-episode recovery.
