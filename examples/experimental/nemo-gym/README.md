# SWE-agent training via NeMo Gym

## Introduction

This example trains a SWE agent with Miles using NVIDIA's
[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) as the environment ecosystem:
NeMo Gym's sandbox-backed `mini_swe_agent_2` agent runs the
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) v2 harness inside
per-task SWE-bench containers (via the `nemo_gym.sandbox` provider API) and
grades every episode with the official SWE-bench harness; Miles owns training,
batch orchestration, and lossless token recording.

NeMo Gym plugs in at the **agent function** layer (see the
[Environments guide](../../../docs/user-guide/environments.md)), the same
shape as the Harbor and OpenEnv connectors:

```
Miles trainer ── session server (records every chat-completions turn: token
   │             ids + logprobs + loss masks, no re-tokenization)
   │  per-sample POST /run { task fields, policy_base_url = session URL }
   ▼
NeMo Gym responses-API agent server (mini_swe_agent_2)
   │  runs mini-swe-agent v2 against policy_base_url,
   │  per-task container via a nemo_gym.sandbox provider (docker here;
   │  daytona / apptainer / ecs_fargate / opensandbox also exist)
   ▼
reward (official SWE-bench harness) ──► sample.metadata ──► reward hook
```

- `nemogym_agent_function.py` — the connector: one `/run` POST per sample.
- `nemogym_generate.py` — reward hook (reads the NeMo Gym grade from
  `sample.metadata["reward"]`).
- `eval_nemogym_via_api.py`, `tests/` — no-GPU validation tooling (below).

Run the NeMo Gym server from `main` (>= `fcca3a8`).

## Validation status

Validated end-to-end (2026-07-28) — the commands in this README are the exact
ones used:

- offline contract tests: 7/7 pass;
- golden scan: gold patch through the official
  `swebench/sweb.eval.x86_64.*` container scored **reward 1.0**;
- API-policy scan: DeepSeek drove a full episode through the
  `policy_base_url` override — patch applied, FAIL_TO_PASS 4/5, a legitimate
  reward 0.0;
- **GPU training smoke** (`run.py` defaults, 4x H200,
  Qwen3-4B-Instruct-2507, SWE-bench Verified prompts): 3 synchronous GRPO
  steps completed twice; every episode ran mini-swe-agent v2 in a real task
  container on the NeMo Gym host, the SWE-bench harness executed the task's
  full test suite (e.g. 175/175 PASS_TO_PASS on an unresolved attempt), and
  the grade flowed back into `rollout/raw_reward`.

Two known limitations from the smoke run:

- A 4B policy solves none of these tasks, so rewards were uniformly 0 —
  GRPO then has zero advantage (`rollout/zero_std` fires). That's a model
  capability floor, not a pipeline defect; expect the same until you use a
  stronger policy or an easier task pool.
- `rollout/tito_session_mismatch_rate/v1` reads 1.0 with this model: Qwen3
  chat templates insert an empty `<think></think>` skeleton when re-rendering
  assistant history, which the engine's actual output never contains. It is a
  soft diagnostic — training tokens and loss masks come from the engine's
  recorded token ids, which stay lossless — and is a property of the
  model-family template, not of this connector.

## Setting up the NeMo Gym server

Any docker-capable host works: a CPU box next to the cluster, or a container
beside the trainer (mount `/var/run/docker.sock` and share a docker network
with the trainer — that variant is not validated here). Set `NEMO_GYM_URL` to
wherever the server listens.

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git
cd Gym

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra dev
# Do NOT install the agent's requirements.txt into this venv. `gym env start`
# builds a per-server venv from it automatically; installing it here bumps
# shared pins (e.g. openai) past nemo-gym's caps, and the injected versions
# then make the child venv unresolvable.

# Global config. The model-server entry must boot but receives no policy
# traffic in this setup — every /run carries its own policy_base_url override
# pointing at a Miles session URL. policy_model_name is the model name the
# harness sends on each request (any string Miles' router accepts).
echo "policy_base_url: http://localhost:9/v1
policy_api_key: dummy
policy_model_name: model
default_host: 0.0.0.0" > env.yaml
```

Start the server, composing the agent config, the docker sandbox provider
config, and a model server config:

```bash
gym env start \
    --config responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml \
    --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
    --model-type vllm_model \
    '++mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.port=12000' \
    '++mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.concurrency=16'
```

The first spin-up resolves each server's own venv, which takes a few minutes;
the server is ready when the startup table lists `mini_swe_agent_2` on
port 12000 and uvicorn reports it running.

## Preparing data

On the trainer, download the task instances and convert them to Miles' prompt
data format. The validated smoke run uses **SWE-bench Verified**:

```bash
cd miles/examples/experimental/nemo-gym
python download_and_process_data.py --input princeton-nlp/SWE-bench_Verified \
    --split test --subset verified --output /root/swe_verified.jsonl
```

Each row keeps the full SWE-bench-format instance (`instance_id`, `repo`,
`base_commit`, `problem_statement`, ...) in `metadata`, plus `subset` /
`split` — the agent function forwards all of it in the `/run` body, and
NeMo Gym selects the per-task image from it.

**SWE-Gym caveat**: `--input SWE-Gym/SWE-Gym --subset gym` produces the
training dataset this recipe ultimately targets (per-task images from
`docker.io/xingyaoww/...`), and episodes run fine — but the official
`swebench` package `mini_swe_agent_2` scores with does not carry eval specs
for several SWE-Gym repos (`KeyError: 'getmoto/moto'` at
`make_test_spec`), so those episodes error at grading and score 0. Until
SWE-Gym eval specs are available in that path (upstream gap), train on
SWE-bench-family instances or filter SWE-Gym to repos the `swebench` package
knows.

## Wiring it into training

The launcher is [`run.py`](run.py) (4 GPUs, smoke-scale defaults — scale up
--num-rollout / batch sizes for real training). Its prepare step downloads the
HF checkpoint and converts it to torch_dist on first run (`--skip-prepare` to
skip):

```bash
export NEMO_GYM_URL="http://<nemo-gym-host>:12000"
# Only if the NeMo Gym host cannot resolve the trainer's hostname (e.g. it
# reaches the trainer over a tailnet):
export MILES_ROUTER_EXTERNAL_HOST="<trainer host/IP reachable from that host>"
python examples/experimental/nemo-gym/run.py
```

To wire the connector into a different launch script, the essential pieces
are this example's directory on `PYTHONPATH`, and:

```bash
--prompt-data /root/swe_verified.jsonl
--input-key prompt
--metadata-key metadata
--max-seq-len 16384

--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
--custom-agent-function-path nemogym_agent_function.run
--custom-rm-path nemogym_generate.reward_func
--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
--use-session-server
--session-server-ip 0.0.0.0  # listen on all interfaces for the dial-back
--tito-model qwen3           # match your policy model's TITO family
```

Per sample, `agentic_tool_call` opens a session on Miles' session server and
hands its OpenAI-compatible URL to `nemogym_agent_function.run`, which POSTs
the task to the NeMo Gym server's `/run` with `policy_base_url` set to that
session URL and the sampling settings mapped onto `responses_create_params`
(`temperature`, `top_p`, `max_output_tokens`). NeMo Gym's mini-swe-agent v2
then talks to the policy exclusively through the session URL (litellm chat
completions), so Miles records every turn losslessly — token ids, logprobs,
and loss masks come from the session server, not from re-tokenizing message
text. The episode grade rides back in the `/run` response (`reward`, with the
SWE-bench eval report in `metadata` → `sample.metadata["eval_report"]`) and
enters training through `sample.metadata["reward"]`.

Episodes that fail before the first model call produce no session records; the
sample is marked aborted and `check_no_aborted` drops its group from training.

## Validating without a GPU

Everything except the session server and the training loop can be validated
on CPU-only machines, in three independent layers (all three pass as of
2026-07-28, see [Validation status](#validation-status)):

1. **Offline unit tests** — the `/run` request contract, response mapping,
   failure semantics, and the data conversion. No network, no docker:

   ```bash
   pytest tests/fast/examples/experimental/nemo_gym -q
   ```

2. **Golden scan** — the sandbox + per-task image + SWE-bench harness chain,
   with no model involved at all: start the server with
   `'++mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.run_golden=true'`
   appended to the `gym env start` command, then

   ```bash
   python eval_nemogym_via_api.py --input /root/swe_verified.jsonl --golden --limit 5
   ```

   Every gold patch must score reward 1.0; the script exits non-zero
   otherwise.

3. **API-policy scan** — a real model drives full episodes through the same
   `policy_base_url` override the trainer uses (so this also exercises that
   field end-to-end). Start the server *without* the golden override and with
   `policy_model_name` set to the API model name (e.g. `deepseek-chat`) in
   `env.yaml`, then:

   ```bash
   export DEEPSEEK_API_KEY=...   # or OPENAI_API_KEY
   python eval_nemogym_via_api.py --input /root/swe_verified.jsonl --limit 2 \
       --policy-base-url https://api.deepseek.com/v1
   ```

## Troubleshooting

1. `train.py: error: unrecognized arguments: --max-seq-len
   --custom-agent-function-path`: `MILES_USE_LEGACY_ROLLOUT_V1=1` is set in the
   training environment — the legacy path never registers the agentic flags.
2. `mini_swe_agent_2` dies at spin-up with an unresolvable-dependency error
   (`openai==X` vs `nemo-gym depends on openai<=Y`): the main venv has extra
   packages installed. Recreate it with `uv sync --extra dev` only — see the
   setup note above.
3. Slow episodes are usually docker pulls (each task has its own image,
   fetched on first use) or the in-container SWE-bench evaluation
   (server-side `eval_timeout`, default 1800s). `NEMO_GYM_RUN_TIMEOUT`
   (default 3600s) caps one episode end-to-end on the Miles side.
4. A failed episode surfaces as `sample.metadata["eval_report"]["error"]` with
   a traceback from the NeMo Gym server — check there before digging into
   server logs.
