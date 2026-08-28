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

## Files

| File | Purpose |
| --- | --- |
| `harbor_agent_function.py` | Builds and runs one Harbor trial per sample; maps the result to sample metadata. Header lists which `harbor-miles-*` fork patches this path depends on. |
| `run.py` | GLM-4.7-Flash launcher. |
| `run_glm52_lora_tb2_daytona.py` | Multi-node GLM-5.2 744B-A40B LoRA launcher (bf16 trainer, fp8 rollout) on Daytona with terminus-2. |
| `launch_common.py` | The launchers' Harbor-side wiring: passes `HARBOR_ENV_TYPE` through and forwards the provider credential (by key-file path) to rollout workers. |

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
`harbor[e2b]` works for it, and is the smoke-test configuration.

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
`EnvironmentConfig.kwargs`); `HARBOR_OVERRIDE_STORAGE_MB` / `HARBOR_OVERRIDE_MEMORY_MB`
replace the task's declared limits for any backend.

### Daytona

Daytona accounts have a total-disk quota, so keep concurrent sandboxes times
`HARBOR_OVERRIDE_STORAGE_MB` under it. `'{"auto_snapshot": true}'` in
`HARBOR_ENV_KWARGS` snapshots each task image on first use so later trials skip
the build; `HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER` gives the first build of each
task headroom. With terminus-2 (a host-process agent) the sandboxes need no
route back to the trainer.

```bash
HARBOR_ENV_TYPE=daytona HARBOR_ENV_KWARGS='{"auto_snapshot": true}' HARBOR_OVERRIDE_STORAGE_MB=10240 \
    python examples/experimental/harbor/run.py ... --prompt-data /path/to/tb2_train.jsonl   # data prepared with --agent-name terminus-2
```

For the multi-node GLM-5.2 744B-A40B LoRA shape, `run_glm52_lora_tb2_daytona.py`
defaults to this backend.

## Timeouts and failure semantics

`AGENT_TIMEOUT` is Harbor's per-trial agent budget; `AGENT_TRIAL_TIMEOUT`
(default 7200 s) is the wall-clock cap around the whole trial and must stay
above it. A trial that ends without a verdict scores 0 with a named
`exit_status` (`TimeLimitExceeded`, `SequenceLengthLimitExceeded`,
`AgentError`), the same vocabulary the agent-server path reports. Nothing is
discarded on this path yet; see the
[agentic rollout guide](../../../docs/user-guide/agentic-rollout.md) for which
outcomes should be.

## Status

Offline tests cover the config built per harness, the `HARBOR_ENV_TYPE`
pass-through, and the result mapping. Validation on a live backend
(AgentENV, public `harbor[e2b]`, mini-swe-agent, then a GRPO smoke) is pending
and will be recorded here.
