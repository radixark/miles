---
title: Rollout Endpoints
description: Use the low-level /generate endpoint from a custom generate function.
---

Use SGLang's stateless `/generate` endpoint when a custom generate function must
control prompt construction and token handling directly. This page documents
that low-level path.

For an agent or environment loop that exchanges OpenAI-compatible chat messages
through Miles' TITO session server, use
[Agentic Rollout (TITO)](/user-guide/agentic-rollout). That guide owns the
wrapper setup, request contract, session behavior, and message/token ownership.

Both paths are selected through `--custom-generate-function-path`.

---

## The `/generate` endpoint

### What `generate_hub` is

`miles/rollout/generate_hub/` ships reusable generate functions that conform to the
refactored rollout interface (`GenerateFnInput` / `GenerateFnOutput`). They compose
with custom agents, tool use, or multi-turn logic.

Key modules:

| Path | Purpose |
|---|---|
| `miles/rollout/base_types.py` | `GenerateFnInput` / `GenerateFnOutput` |
| `miles/rollout/inference_rollout/inference_rollout_common.py` | Builds a `GenerateState` and calls the generate function |
| `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1` | Enables the new path (see `examples/swe-agent-harbor-docker`) |

### Generate function basics

The runtime contract:

1. The rollout engine passes a `GenerateFnInput` containing:
    - `state`: tokenizer, processor, args, sampling defaults.
    - `sample`: the prompt, current tokens, response, status.
    - `sampling_params`: `max_new_tokens`, `temperature`, `top_p`, etc.
2. Your function:
    - Builds a request from the prompt.
    - Executes it against SGLang.
    - Updates the `Sample` with tokens, logprobs, loss mask, status.

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
    # 3) update sample.tokens, sample.response, sample.rollout_log_probs,
    #    sample.loss_mask, sample.status

    return GenerateFnOutput(samples=sample)


def _add_arguments(parser):
    parser.add_argument("--your-arg", type=str)


generate.add_arguments = _add_arguments
```

<Tip>

**Custom CLI flags.** `generate.add_arguments = _add_arguments` registers extra CLI flags. They are
parsed into `input.args` and available everywhere in your generator.

</Tip>

Helpers:

- `compute_prompt_ids_from_sample` and `compute_request_payload` from
  `miles/rollout/generate_utils/generate_endpoint_utils.py` build `/generate` requests.
- A generate function can set `GenerateFnOutput.samples` to a `Sample` or `list[Sample]`.

### Reference generators

- **`single_turn.py`**: single-turn generation via `/generate`. Text or multimodal prompts.
- **`multi_turn.py`**: multi-turn tool calling via `/generate`. Adds CLI flags
  `--generate-max-turns`, `--generate-tool-specs-path`, `--generate-tool-call-parser`,
  `--generate-execute-tool-function-path`.
- **`benchmarkers.py`**: forces random output sequence length for benchmarking.

---

## Next

- [Agentic Rollout (TITO)](/user-guide/agentic-rollout): build an
  OpenAI-compatible agent loop on the session server.
- [Customization](/user-guide/customization): browse every Python hook.
- [Multi-agent example](/examples/multi-agent): follow a complete agentic
  walkthrough.
