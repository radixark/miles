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
  "return_prompt_token_ids": true
}
```

You can pass any OpenAI-compatible parameters in the payload, or any
SGLang-compatible `ChatCompletionRequest` parameters. Note:
`logprobs=True` and `return_prompt_token_ids=True` are set in
`request_kwargs` to extract token ids and logprobs for TITO (see below).
Do **not** set `logprob_start_len=0` — it would disable SGLang's prefix
cache.

## 3. Quickstart index

If you just want something runnable, start here:

Generator entry point:

- `miles/rollout/generate_hub/agentic_tool_call.py`
  - OpenAI-format agent loop via router sessions.

OpenAI-format examples that use `agentic_tool_call.generate`:

- `examples/openai_format/dapo_math.py`
  - Single-turn OpenAI format agent (DAPO math).
- Launcher scripts:
  - `examples/openai_format/run-qwen3-4B.sh`


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

TITO needs two things from the SGLang response:

1. **Prompt token ids** — extracted from `response.choices[0].prompt_token_ids`.
   This field is returned by SGLang when the request sets
   `return_prompt_token_ids=True`.
2. **Output token ids and logprobs** — extracted from
   `response.choices[0].logprobs.content[*].token_id` and
   `response.choices[0].logprobs.content[*].logprob`.
   These fields are returned when the request sets `logprobs=True`.

By default, `build_chat_request_kwargs` in `agentic_tool_call.py` sets both
`return_prompt_token_ids=True` and `logprobs=True`. The session middleware
forwards raw `messages` to SGLang, which tokenizes the prompt and returns the
response. Then `_compute_sample_from_openai_record` in
`openai_endpoint_utils.py` extracts prompt token ids and output token ids from
the response and concatenates them into `sample.tokens`. You do not need to
provide `input_ids` yourself.

We can save multi-turn samples within a single session, but currently we
still do not inherit or reuse tokens across turns. Each request is tokenized
independently.

### Common pitfalls

- Ensure `logprobs=True` and `return_prompt_token_ids=True` in OpenAI chat
  requests (both are already set in `request_kwargs`).
- Do **not** set `logprob_start_len=0` — it forces SGLang to compute
  logprobs for every prompt token, which destroys the prefix cache and hurts
  performance. Use `return_prompt_token_ids=True` instead, which returns
  prompt token ids at zero cost without affecting caching.
