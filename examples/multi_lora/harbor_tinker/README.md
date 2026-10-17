# Terminal-Bench 2 training through Harbor and the official Tinker proxy

Install the independent client package; running it does not import or require
Miles. The Harbor version is pinned to the same fork used by the Miles Harbor
example. Request/render/stop/tool/SSE behavior comes unchanged from the pinned
official `tinker_cookbook.capture.proxy`.

```bash
uv pip install 'tinker-agent-rollouts[harbor] @ git+https://github.com/radixark/miles@<commit>#subdirectory=examples/multi_lora/tinker_agent'
uv pip install 'harbor[e2b] @ git+https://github.com/harbor-framework/harbor@917b3c1be7cf6553a7a9fe25de8adb6f8e639287'
```

Use the complete Terminal-Bench 2 dataset, as in the other Miles Harbor training
examples. Download its native Harbor task directories:

```bash
harbor datasets download terminal-bench@2.0 --output-dir /data/harbor-tasks
export TB2_TASKS_DIR=/data/harbor-tasks/terminal-bench
```

The recipe loads every task directory containing `task.toml` under
`TB2_TASKS_DIR`; there is no task-specific selection. Each task keeps its original
instruction, environment and verifier. An existing TB2 directory used by another
Miles Harbor example can be reused by setting `TB2_TASKS_DIR` to that directory.
The same interface also accepts other datasets in Harbor's native task format.

Create `trial.json` with Harbor `TrialConfig` fields, omitting `task` (the
recipe supplies each dataset task). This example uses mini-swe-agent with E2B,
matching the cloud Harbor example:

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

Use Harbor's normal schema for agent kwargs, sandbox/provider credentials,
timeouts and verifier settings. The recipe substitutes only `{proxy_url}`
and `{proxy_api_key}` in this JSON. The URL includes a trajectory-specific
`/r/run/<run_id>/group/<group_idx>/traj/<traj_idx>` prefix. For `mini-swe-agent`, put `{proxy_url}/v1` in
`agent.env.OPENAI_BASE_URL` and `{proxy_api_key}` in `agent.env.OPENAI_API_KEY`.
For `claude-code`, use `{proxy_url}` in `agent.env.ANTHROPIC_BASE_URL` and
`{proxy_api_key}` in `agent.env.ANTHROPIC_API_KEY`. Choose a model/renderer
that supports the agent's requested tools and message format.

Run the recipe and its model proxy on the GPU devbox. The devbox must reach
both E2B endpoints, and E2B sandboxes must reach the proxy on that devbox.
For the internal sandbox service, acquire the devbox with `rx devbox acquire
--tailnet` and use its tailnet address as `HARBOR_PROXY_HOST`. On a multi-node
devbox, run the recipe on rank 0, where the platform exposes tailnet access.
The proxy binds to `0.0.0.0` on dynamically allocated ports; verify sandbox
connectivity to that address before starting training.

```bash
export TINKER_API_KEY=tml-your-key
export E2B_API_KEY="$(cat ~/.config/e2b/api_key)"
export HARBOR_PROXY_HOST='devbox-address-reachable-from-e2b'
python -m tinker_agent.train \
  config.model_name=Qwen/Qwen3.6-35B-A3B config.renderer_name=qwen3_5 \
  config.learning_rate=1e-6 config.max_tokens=8192 config.lora_rank=16 \
  config.log_path=/tmp/harbor-tinker config.loss_fn=ppo \
  config.dataset_builder.tasks_dir="$TB2_TASKS_DIR" \
  config.dataset_builder.groups_per_batch=4 config.dataset_builder.group_size=8 \
  config.base_url=http://miles-gateway:8000 config.ttl_seconds=None \
  harbor.trial_config=trial.json \
  harbor.listen_host=0.0.0.0 harbor.advertised_host="$HARBOR_PROXY_HOST"
```

The gateway must serve the same Qwen3.6 checkpoint and support rank-16 LoRA.
Start it with `--target-modules attn,mlp,unembed` to match the cookbook client's
default LoRA layout, including `train_unembed=True`.
Qwen3.6 uses cookbook's `qwen3_5` renderer. Set both the inference context limit
and the gateway's per-Datum training limit to 65536 tokens; tensor
parallelism does not increase the per-Datum limit. Configure E2B credentials and
endpoints as described in [Sandbox Providers](../../../docs/user-guide/sandbox-providers.md).
If the image sets Hugging Face cache variables to a read-only shared model
mount, configure `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE` and
`TRANSFORMERS_CACHE` consistently for the recipe's writable tokenizer cache.
Setting only `HF_HOME` does not override an inherited `HUGGINGFACE_HUB_CACHE`.
The Miles gateway caps each generation request to the remaining context budget
and rejects a full prompt before dispatching it to the engine. It preserves the
prompt and sampled tokens; context errors reach the agent through the official
proxy. `config.ttl_seconds=None` disables checkpoint expiry, which this gateway
does not implement.

This produces 32 trials per full batch (4 tasks, 8 trajectories each). Add
`config.max_steps=1` for a one-batch acceptance run. The dataset remains the full
TB2 set; the first batch uses its first four task directories in sorted order.
The agent, sandbox, batch size, sampling temperature, response/context limits
and learning rate match the cloud Harbor example. Training remains Tinker LoRA
with cookbook PPO/group advantages, rather than the native example's full-parameter
GRPO; optimizer defaults and KL behavior are not equivalent.

Omit `config.base_url` to use the official Tinker service. For remote sandboxes,
`harbor.advertised_host` must resolve to the devbox running this recipe from the sandbox; use
your provider's reachable host address. Each group gets a random proxy auth
token. Rendering and official capture run on the client machine. Agent
sampling parameters come from the agent; `config.max_tokens` supplies the proxy's
default only when the request omits it.

`config` is the official cookbook `rl.train.Config`, with its native names,
required fields and defaults. The CLI supplies only the recipe name and the
Harbor dataset builder type. Dataset options live under `config.dataset_builder`;
`harbor` contains only the trial config path, reward key and proxy connection /
concurrency options. For example, `config.load_checkpoint_path`,
`config.rolling_save_every` and `config.async_config.max_steps_off_policy` pass
directly to cookbook without recipe-specific copies or a field allowlist.
Use `python -m tinker_agent.train --help` for the complete configuration.

The recipe installs `HarborRolloutStrategy` in `config.rollout_error_tolerance`;
setting that field explicitly is rejected. Set sampling temperature in the
Harbor agent config; non-default `config.temperature` is rejected because the
external agent owns its sampling requests. An omitted renderer is resolved
with cookbook's model recommendation. Training defaults, including the loss,
LoRA rank, save cadence and checkpoint TTL, come from cookbook. Set
`config.ttl_seconds=None` explicitly to disable checkpoint expiry.

The recipe wraps cookbook `train.main` in one official `capture_to_store`
context for the entire run, with storage under `<log_path>/capture`. Each
rollout group fixes one sampling client and uses distinct group/trajectory
coordinates in the official proxy URL. After the trials finish, the proxy
closes and drains accepted handlers; the recipe flushes the official exporter
and reads the store's public rows API. Those rows are the sole sampled-token
record, including compaction and repeated prompts. Cookbook computes group
advantages and merges only exact token-prefix continuations into Datums.

A completed trial, agent timeout or token-limit failure with a verifier reward is scored using
`harbor.reward_key` (default `reward`). Missing/non-finite rewards and other Harbor
exceptions abort the group. Exporter loss or rows without training logprobs
also stop training. Failed requests remain official capture annotations;
the agent and Harbor determine whether the trial can continue. There is no
synthetic zero reward for infrastructure errors. Task and agent limits belong
to Harbor; rendering, stops, tools and streaming use official proxy behavior.
There is no separate recorder, episode state machine or partial-episode
recovery.

The isolated CPU tests compare the wrapped and bare official proxy and test
Harbor configuration, trial outcomes and token-to-Datum conversion with a
scripted trial. The registered H100 gateway test exercises real sampling and
PPO training. Scripted trials do not validate live sandbox provisioning or
agent binaries; those require an actual Harbor environment.
