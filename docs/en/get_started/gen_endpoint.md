# Gen Endpoint Usage

This document covers generate_hub usage for the `/generate` endpoint. For OpenAI
format usage, see `docs/en/get_started/oai_endpoint.md`.

## 1. What generate_hub is

`miles/rollout/generate_hub/` contains reusable generate functions that plug into
rollout through `--custom-generate-function-path`. They use the refactor
interface (`GenerateFnInput` / `GenerateFnOutput`) and are meant to be composed
with custom agents, tool use, or multi-turn logic.

Key types and entry points:

- `miles/rollout/base_types.py` defines `GenerateFnInput` and `GenerateFnOutput`.
- `miles/rollout/inference_rollout/inference_rollout_common.py` builds a
  `GenerateState` and calls the generate function.
- `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1` enables the new path (see
  `examples/openai_format/*.sh`).

## 2. Generate function basics

The intended abstraction is:

1. The rollout engine provides a `GenerateFnInput` with:
   - `state` (tokenizer, processor, args, sampling defaults)
   - `sample` (prompt, current tokens, response, status)
   - `sampling_params` (max_new_tokens, temperature, top_p, etc.)
2. The generate function focuses only on:
   - turning the sample into a model request
   - executing the request (SGLang `/generate` or OpenAI format)
   - updating the `Sample` with tokens, logprobs, loss mask, and status

Minimal skeleton:

```python
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample

async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = input.sample
    sampling_params = input.sampling_params

    # 1) build request from prompt and sampling params
    # 2) call backend
    # 3) update sample.tokens, sample.response, sample.rollout_log_probs, sample.loss_mask, sample.status

    return GenerateFnOutput(samples=sample)

def _add_arguments(parser):
    parser.add_argument("--your-arg", type=str)

generate.add_arguments = _add_arguments
```

Notes:

- `generate.add_arguments = _add_arguments` is the hook for custom CLI flags.
  Add any arguments you want; they are parsed into `input.args` and can be used
  freely by your generator without touching rollout core code.
- Use `compute_prompt_ids_from_sample` and `compute_request_payload` from
  `miles/rollout/generate_utils/generate_endpoint_utils.py` to build requests
  for the `/generate` endpoint.
- If you want to return multiple samples, set `--generate-multi-samples` and
  return a list.

## 3. /generate endpoint examples

Examples (library side):

- `miles/rollout/generate_hub/single_turn.py`
  - Single-turn generation using `/generate`.
  - Works with text or multimodal prompts.
- `miles/rollout/generate_hub/multi_turn.py`
  - Multi-turn tool calling using `/generate`.
  - CLI flags: `--generate-max-turns`, `--generate-tool-specs-path`,
    `--generate-tool-call-parser`, `--generate-execute-tool-function-path`,
    `--generate-multi-samples`.
- `miles/rollout/generate_hub/benchmarkers.py`
  - Benchmark helper that forces random output sequence length (OSL).

## 4. Radix tree middleware helper (full TITO for `/generate`)

Full TITO caching for the `/generate` endpoint is provided by the radix tree
middleware. This is unrelated to session middleware and works only on the
`/generate` and `/retrieve_from_text` routes.

What it does:

- Caches token ids and logprobs by prompt text in a radix tree.
- Lets `/generate` requests include `input_tokens` and avoids re-tokenization.
- Enables `update_sample_from_response` to fetch tokens via
  `/retrieve_from_text` for training.

How to enable:

```
--use-miles-router \
--miles-router-middleware-paths miles.router.middleware_hub.radix_tree_middleware.RadixTreeMiddleware
```

Make sure `--sglang-router-ip` and `--sglang-router-port` point to the Miles
Router so `/retrieve_from_text` can be reached during rollout.
