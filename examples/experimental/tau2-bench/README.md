# τ²-bench

GRPO on [τ²-bench](https://github.com/sierra-research/tau2-bench) customer-service
tasks (airline, retail, telecom). Each training sample is one tau2 task, played end
to end by tau2's own orchestrator, user simulator, environment and evaluator. Only
the agent is Miles': it keeps tau2's `LLMAgent` system prompt and tools, but answers
through the session server's OpenAI-compatible endpoint, so every policy turn is
recorded token by token ([Agentic Rollout (TITO)](../../../docs/user-guide/agentic-rollout.md)).

```text
agentic_tool_call.generate ── session (TITO) ── base_url
        │
        └─ tau2_agent_function.run(base_url, metadata={"domain", "task_id"})
              tau2 Orchestrator
                ├─ SessionAgent ── POST base_url/v1/chat/completions (tools = domain tools)
                ├─ UserSimulator ── litellm (TAU2_USER_LLM)
                └─ Environment ── tool calls, DB
              tau2 run_simulation ── reward → sample.metadata["reward"]
```

The older [tau-bench](../tau-bench) example trains on τ-bench v1 through the
`/generate` endpoint with its own tokenization; this one uses the session server
and the current τ²-bench tasks.

## Setup

Install tau2-bench (Python 3.12 or 3.13) from a source checkout, which also holds
its task data:

```bash
git clone https://github.com/sierra-research/tau2-bench.git /root/tau2-bench
pip install -e /root/tau2-bench
```

With a non-editable install, point `TAU2_DATA_DIR` at the checkout's `data`
directory.

Besides the policy, an episode calls up to two more models, both through
[litellm](https://docs.litellm.ai/docs/providers): the user simulator, and the
judge that scores the parts of a task's reward basis tau2 cannot check
deterministically.

| Domain | Reward basis | tau2's default judge |
|---|---|---|
| airline | DB, COMMUNICATE | `gpt-5.5` (semantic communicate check) |
| retail | DB, NL_ASSERTION | `gpt-5.4-mini` (NL assertions) |
| telecom | ENV_ASSERTION, ACTION | none |

With the defaults (user simulator `gpt-4.1-2025-04-14`) every domain needs
`OPENAI_API_KEY`. To use your own endpoint instead, point both at any
OpenAI-compatible server (SGLang, vLLM, ...) through litellm's `hosted_vllm/`
provider, which needs no key:

```bash
export TAU2_USER_LLM=hosted_vllm/<served-model-name>
export TAU2_USER_LLM_ARGS='{"api_base": "http://<host>:<port>/v1", "temperature": 0.0}'
export TAU2_JUDGE_LLM=hosted_vllm/<served-model-name>
export TAU2_JUDGE_LLM_ARGS='{"api_base": "http://<host>:<port>/v1", "temperature": 0.0}'
```

`TAU2_JUDGE_LLM` replaces both judges. They must answer in plain JSON in
`content`: serve a reasoning model with its reasoning parser so its thinking
stays out of the reply. A different judge also means rewards are no longer
scored exactly as tau2 reports them.

Keys go in the provider's own environment variable (for example
`HOSTED_VLLM_API_KEY` for an endpoint that checks one), not in the `*_ARGS`
JSON: the launcher forwards the `TAU2_*` variables to ray workers by value,
while workers inherit the launcher's environment for everything else. The
launcher refuses to start when a model it would call is missing a key.

## Run

```bash
cd examples/experimental/tau2-bench
python run.py                     # retail, one node of 8 GPUs
python run.py --domain airline
python run.py --domain telecom --mode debug_rollout_only
MILES_SCRIPT_EXTERNAL_RAY=1 python run.py --num-nodes 2   # an already-joined ray cluster
```

The policy is Qwen3.8-27B. It thinks before every reply and its template keeps each
turn's thinking in the history, so the recipe asks for the template's low reasoning
effort (`--reasoning-effort`); higher efforts need a longer `--max-seq-len`.

`prepare` downloads Qwen3.8-27B, converts it to `torch_dist`, and writes
the domain's tau2 `train` and `test` splits to
`<data-dir>/tau2-bench/<domain>_{train,test}.jsonl`, one row per task:

```json
{"prompt": [{"role": "user", "content": "tau2 retail task 0"}], "metadata": {"domain": "retail", "task_id": "0"}}
```

The prompt only labels the sample: the agent loads the task from tau2 by its id,
and tau2 supplies the conversation. Training uses the `train` split and evaluates
on `test`; `run.py` holds the defaults.

## Rewards and failures

`tau2_agent_function.run` returns `reward`, `termination_reason` and the episode's
non-generation time; `reward_func` reads `reward` back.

- **The simulation finishes:** tau2's reward. tau2 already scores every
  termination other than an agent or user stop (max steps, too many tool errors) 0.
- **The policy ends the episode early:** 0. That covers a reply cut at
  `--rollout-max-response-len` (`length`), a conversation reaching `--max-seq-len`
  (`context_window_exceeded`), and an empty reply or a tool call whose arguments
  are not a JSON object (`agent_error`).
- **Anything else fails** (user simulator or judge API, session server): no reward
  (`infrastructure_error`). Training drops the group instead of learning a false 0;
  evaluation reports the share as `eval/<name>-none_reward_ratio`.

The policy needs the SGLang parsers of its family (`qwen3` reasoning, `qwen3_coder`
tool calls for Qwen3.8). Without the tool-call parser, tool calls stay text and every
task ends in replies to the user; without the reasoning parser, thinking reaches tau2
as part of the reply.
