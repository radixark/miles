---
title: Generate Endpoint
description: Write a custom generate function against SGLang's raw /generate endpoint, owning prompt construction, tokens, and loss masks.
---

Write a token-level generate function when your rollout logic must control
prompt construction and token handling directly. Your code builds the request,
calls SGLang's stateless `/generate` endpoint, and writes tokens, logprobs,
loss mask, and status back onto the `Sample`.

This is one of two styles of custom generation, both selected through
`--custom-generate-function-path`. The other style exchanges OpenAI-compatible
chat messages instead of tokens — see
[Agentic Rollout (TITO)](/user-guide/agentic-rollout).

---

## The generate-function hook

`--custom-generate-function-path` accepts two forms. The difference is only
the signature you write — `load_generate_function`
(`miles/rollout/inference_rollout/compatibility.py`) adapts an old-form
function automatically at load time:

| | New form (recommended) | Old form |
|---|---|---|
| Signature | `async def generate(input: GenerateFnInput) -> GenerateFnOutput` | `async def generate(args, sample, sampling_params) -> Sample` |
| `--custom-generate-function-path` | `miles.rollout.generate_hub.single_turn.generate`, `miles.rollout.generate_hub.multi_turn.generate` | `miles.rollout.sglang_rollout.generate` |
| Example file | [miles/rollout/generate_hub/single_turn.py](https://github.com/radixark/miles/blob/main/miles/rollout/generate_hub/single_turn.py), [miles/rollout/generate_hub/multi_turn.py](https://github.com/radixark/miles/blob/main/miles/rollout/generate_hub/multi_turn.py) | [miles/rollout/sglang_rollout.py](https://github.com/radixark/miles/blob/main/miles/rollout/sglang_rollout.py) |
| Status | what all `generate_hub` built-ins use | kept for backward compatibility; adapted automatically at load time |

<Note>

`MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1` is off by default — set it yourself
(through `runtime_env` when submitting via Ray). Both generate-function forms
work with or without it.

</Note>

Either way, your function does the same three things:

1. Builds a request from the prompt.
2. Executes it against SGLang.
3. Updates the `Sample` with tokens, logprobs, loss mask, status.

`GenerateFnInput` / `GenerateFnOutput` live in `miles/rollout/base_types.py`.
The input carries:

- `state`: tokenizer, processor, args, sampling defaults.
- `sample`: the prompt, current tokens, response, status.
- `sampling_params`: `max_new_tokens`, `temperature`, `top_p`, etc.
- `evaluation`: whether this call serves an eval rollout.

Minimal skeleton (new form):

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

## Reference generators

`miles/rollout/generate_hub/` ships reusable token-level generate functions
that compose with tool use and multi-turn logic:

- **`single_turn.py`**: single-turn generation via `/generate`. Text or multimodal prompts.
- **`multi_turn.py`**: multi-turn tool calling via `/generate`. Adds CLI flags
  `--generate-max-turns`, `--generate-tool-specs-path`, `--generate-tool-call-parser`,
  `--generate-execute-tool-function-path`.
- **`benchmarkers.py`**: forces random output sequence length for benchmarking.

---

## Next

- [Customization](/user-guide/customization): browse every Python hook.
- [Agentic Rollout (TITO)](/user-guide/agentic-rollout): the message-level
  style of custom generation.
