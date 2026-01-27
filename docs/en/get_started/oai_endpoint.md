# OAI Endpoint Usage

This document explains how to use the OpenAI-format chat endpoint through Miles
Router sessions. For the `/generate` endpoint, see
`docs/en/get_started/gen_endpoint.md`.

## 1. Minimal `run_agent` loop

Your `run_agent` receives a session-scoped `base_url`. Send OpenAI-format chat
requests to `base_url/v1/chat/completions` and pass the `messages` list as the
prompt.

Minimal custom agent example:

```python
from miles.utils.http_utils import post

async def run_agent(base_url: str, prompt, request_kwargs: dict | None = None) -> None:
    payload = {"model": "default", "messages": prompt, **(request_kwargs or {})}
    await post(f"{base_url}/v1/chat/completions", payload)
```

Notes for `run_agent`:

- `base_url` already includes the session path (e.g. `/sessions/<id>`), so you
  should not manually add the session id. Just append the OpenAI route.
- `request_kwargs` already contains the default sampling settings from
  `agentic_tool_call.build_chat_request_kwargs`, so you can directly expand it
  into the chat request payload.
- If you pass rollout sampling params, `max_new_tokens` will be mapped to the
  OpenAI `max_tokens` field before the request is sent.
- If you need structured parsing payloads, use SGLang's
  `ChatCompletionRequest`-compatible format. It is compatible with native OpenAI
  fields, plus extra SGLang parameters.

## 2. OpenAI chat messages and the basic request

The OpenAI-format chat API uses a list of `messages`, each with a `role` and
`content`.

Minimal request shape:

```json
{
  "model": "default",
  "messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Answer with one word: 2+2?"}
  ],
  "logprobs": true,
  "logprob_start_len": 0
}
```

You can pass any OpenAI-compatible parameters in the payload, or any
SGLang-compatible `ChatCompletionRequest` parameters. Note:
`logprobs=True` and `logprob_start_len=0` are required to extract token ids and
logprobs for TITO (see below), and are already set in `request_kwargs`.

## 3. Quickstart index

If you just want something runnable, start here:

Generator entry point:

- `miles/rollout/generate_hub/agentic_tool_call.py`
  - OpenAI-format agent loop via router sessions.

OpenAI-format examples that use `agentic_tool_call.generate`:

- `examples/openai_format/dapo_math.py`
  - Single-turn OpenAI format agent (DAPO math).
- Launcher scripts:
  - `examples/openai_format/run-qwen3-4B-dapo-math.sh`


You can customize generate function like:
```
CUSTOM_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path examples.openai_format.dapo_math.run_agent
)
```

For OpenAI format, do not add `--apply-chat-template`; the
prompt must remain a `messages` list.

More agentic multi-turn examples will come in the future.

## 4. Further customization (OpenAI wrapper generate function)

For OpenAI-format rollout, the key generate function is
`miles/rollout/generate_hub/agentic_tool_call.generate`. It is a thin wrapper
around your custom agent:

1. Create a session on Miles Router and build a session-scoped `base_url`.
2. Call the custom agent (from `--custom-agent-function-path`) to send one or
   more chat requests to `base_url/v1/chat/completions`, typically using
   `prompt` and `request_kwargs`.
3. Collect session records via `OpenAIEndpointTracer`.
4. Convert records into `Sample` objects with
   `compute_samples_from_openai_records`.

If you want general generate-function customization beyond the OpenAI wrapper,
see `docs/en/get_started/gen_endpoint.md`.

## 5. TITO (token-in token-out)

TITO needs two things:

1. Prompt token ids returned by the backend (e.g. `input_logprobs` or
   `input_token_ids`). These can come from tokenizing `messages`, or from a
   provided `input_ids` payload.
2. Output token ids returned by the backend (`logprobs.content[*].token_id`).

By default, the session middleware forwards raw `messages` to SGLang. With
`logprobs=True` and `logprob_start_len=0`, SGLang tokenizes the prompt and
returns prompt token ids along with output token ids, which is sufficient for
TITO. You do not need to provide `input_ids`.

If you prefer to send `input_ids` to SGLang, you can enable token input for chat
completions in the router via
`--miles-router-enable-token-input-for-chat-completions`. The session route
will tokenize `messages` and inject `input_ids` before proxying to SGLang. The
backend still returns prompt token ids, and they should match any `input_ids`
you supplied.

We can save multi-turn samples within a single session, but we still do not
inherit or reuse prompt tokens across turns. Each request is tokenized
independently, regardless of which option you choose.

### Common pitfalls

- Ensure `logprobs=True` in OpenAI chat requests, and ensure
  `logprob_start_len=0` if you rely on SGLang to return prompt token ids.
- Ensure the tokenizer matches `--hf-checkpoint`.
