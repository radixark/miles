# Harbor in-process on cloud sandboxes

This example runs [Harbor](https://github.com/harbor-framework/harbor) trials
**inside the rollout worker**: the agent function builds a `TrialConfig` and
calls `Trial.run()` directly, with the task sandbox on a cloud backend the worker
reaches over the network — E2B Cloud or a self-hosted
[AgentENV](../agentenv/README.md) (E2B API), Daytona, Modal, or any other Harbor
`EnvironmentType`. There is no agent server.

Compared with [`examples/swe-agent-harbor-docker`](../../swe-agent-harbor-docker/README.md):

| | agent server (`swe-agent-harbor-docker`) | in-process (this example) |
| --- | --- | --- |
| Where `Trial.run()` runs | a separate host with a Docker daemon | the rollout worker |
| Sandbox backends | `docker` (local), `daytona` via the server's env | any Harbor backend the worker can reach; `HARBOR_ENV_TYPE` is passed straight to Harbor |
| Moving parts | trainer → HTTP → agent server → Harbor | trainer → Harbor |
| Use it when | tasks must run on the local Docker daemon | sandboxes are cloud-hosted |

Everything on the trainer side is the same: TITO, the session server, GRPO, the
reward hook (`generate.py` from the agent-server example).

## 1. Install Harbor in the rollout environment

Harbor now runs where the rollout workers run, so it goes into the Miles
image / environment. Use the `harbor-miles-v0.20.0` branch of
`harbor-framework/harbor` — the terminus-2 truncation policy it carries is
required for TITO (see the agent function's header for the full list) —
with the extra for your backend:

```bash
pip install "harbor[e2b] @ git+https://github.com/harbor-framework/harbor@harbor-miles-v0.20.0"
# or harbor[daytona], harbor[modal], ...
```

mini-swe-agent does not need the fork's patches for correctness; public
`harbor[e2b]` works for it.

## 2. Provision the sandbox backend

Credentials follow the contract every Miles sandbox integration uses: the
worker reads the provider key from its own environment or from a key file; the
launcher forwards only the file's path.

```bash
# E2B Cloud
mkdir -p ~/.config/e2b && echo e2b_... > ~/.config/e2b/api_key
# self-hosted AgentENV instead: point the SDK at it (see ../agentenv/README.md)
export E2B_API_URL=http://<server>:8000 E2B_SANDBOX_URL=http://<server>:8000
# Daytona
mkdir -p ~/.config/daytona && echo dtn_... > ~/.config/daytona/api_key
```

Task directories: `HARBOR_TASKS_DIR` must contain one Harbor task dir per
`metadata.instance_id` in the training data (same as the agent-server example);
put it on a filesystem every worker can read.

**Network.** In-sandbox agents (mini-swe-agent, claude-code) call the model
from inside the sandbox, so the sandbox platform must reach the Miles session
server: `--router-external-host` is the address substituted into the URL the
agent gets, and ports 30000/31000 must route from the sandbox network (for
AgentENV, allow the trainer's subnet in the server's egress config; see the
AgentENV recipe). Host-process agents (terminus-2) call the model from the
worker and need no sandbox egress.

## 3. Prepare data

Same as the agent-server example:

```bash
python examples/swe-agent-harbor-docker/download_and_process_data.py \
    --input /path/to/terminal-bench.jsonl \
    --output /path/to/tb2_train.jsonl \
    --agent-name mini-swe-agent \
    --prompt-key instruction
```

## 4. Launch

```bash
HARBOR_ENV_TYPE=e2b python examples/experimental/harbor/run.py \
    --num-nodes 1 --num-gpus-per-node 8 --skip-prepare \
    --megatron-path /root/Megatron-LM \
    --hf-checkpoint /path/to/GLM-4.7-Flash \
    --ref-load /path/to/GLM-4.7-Flash_torch_dist \
    --save-dir /path/to/checkpoints \
    --prompt-data /path/to/tb2_train.jsonl \
    --harbor-tasks-dir /path/to/harbor_tasks \
    --router-external-host <trainer-address-reachable-from-the-sandboxes> \
    --rollout-batch-size 4 --n-samples-per-prompt 8 --global-batch-size 32 \
    --num-rollout 200 --save-interval 10
```

`HARBOR_ENV_TYPE` has no default: the backend decides whose quota a run spends.
Backend-specific settings go in `HARBOR_ENV_KWARGS` as a JSON object (Harbor's
`EnvironmentConfig.kwargs`), e.g. `'{"auto_snapshot": true}'` for Daytona.

Every remaining knob — timeouts and their layering, failure semantics, the
full env-var reference — is documented in `harbor_agent_function.py`'s header,
next to the code that reads it.

## Validation

The platform round trip (golden agent, real sandbox APIs) passes on e2b and
Daytona via the sandbox smoke, `scripts/sandbox_smoke`. The GPU e2e — one GRPO
step with terminus-2 on real e2b sandboxes,
`tests/e2e/agentic/test_harbor_e2b_training.py` — **passed 2026-09-04** on a
2×H200 devbox against a self-hosted E2B-compatible service (2 trials
submitted, optimizer step completed). `run.py` itself: one-step rollout smoke
pending.
