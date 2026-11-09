# Pydantic AI as an agent function

[`pydantic_ai_agent_function.py`](pydantic_ai_agent_function.py) runs a
[Pydantic AI](https://ai.pydantic.dev) agent through the TITO session server. It is a
plain `async def run(...)` that
[`agentic_tool_call.generate`](../../../miles/rollout/generate_hub/agentic_tool_call.py)
calls once per episode, as the
[agentic rollout guide](../../../docs/user-guide/agentic-rollout.md) describes:

- It builds the agent per call, with an `OpenAIChatModel` (Pydantic AI's `openai-chat:`
  model) whose provider points at the call's session URL. A process-wide
  `OPENAI_BASE_URL` would send concurrent episodes to one session.
- The task is arithmetic with one tool, `calculate`. The verdict is programmatic:
  `reward` is 1.0 when the last number the agent writes equals `metadata["answer"]`.
- A 5xx from the model service or an unreachable session raises `InfraAbort`, which
  discards the episode. A 4xx, a tool or output failure the agent did not recover from,
  the model-call cap and the wall-clock cap score 0, since the policy can cause them.
- Cancelling the call cancels the agent run; nothing is caught on the way out.
- Sampling comes from the rollout: Pydantic AI sends no sampling fields here, and the
  session fills the rollout's.

## Install

Pydantic AI goes where the rollout workers run. The Miles image pins `openai==2.8.0`
through SGLang and ships `opentelemetry-api` 1.45, and Pydantic AI 1.30.1, the last
release that accepts openai 2.8, does not import with that OpenTelemetry. 1.64.0 is the
last release that accepts openai 2.11:

```bash
pip install "pydantic-ai-slim[openai]==1.64.0" "openai==2.11.0"
```

## Data

```bash
python examples/experimental/pydantic_ai/make_tasks.py /root/datasets/pydantic_ai/train.jsonl
python examples/experimental/pydantic_ai/make_tasks.py /root/datasets/pydantic_ai/eval.jsonl --num-tasks 128 --seed 1
```

Each row is `{"messages": [...], "metadata": {"answer": ...}}`.

## Flags

Add these to a GRPO launcher, with `examples/experimental/pydantic_ai` on the rollout
workers' `PYTHONPATH`:

```bash
--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
--custom-agent-function-path pydantic_ai_agent_function.run
--use-session-server v2
--session-message-matcher loose_tool_call
--tito-model qwen3
--sglang-reasoning-parser qwen3
--sglang-tool-call-parser qwen25
--prompt-data /root/datasets/pydantic_ai/train.jsonl
--input-key messages
```

- `--use-session-server v2`: v1 keeps one lineage per session and is only right for a
  strictly linear harness.
- `--session-message-matcher loose_tool_call`: Pydantic AI 1.64 replays tool arguments
  as the engine spelled them, but 2.x parses them and resends them compact, and under
  the default `strict` matcher that opens a new root, and a one-turn sample, on every
  turn.
- `--sglang-reasoning-parser` and `--sglang-tool-call-parser`: the parsers registered
  for `--tito-model qwen3`. Without a tool-call parser the engine returns tool calls as
  text, and every episode silently ends after one model call.
- No `--apply-chat-template`: the session server renders the messages.
- No `--custom-rm-path`: the session assigns the agent's `reward` to every sample of the
  episode.

`PYDANTIC_AI_MAX_MODEL_CALLS` (default 8) and `PYDANTIC_AI_EPISODE_TIMEOUT_S` (default
600) bound an episode.
