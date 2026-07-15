# PR 1689 walkthrough

PR: <https://github.com/radixark/miles/pull/1689>

This document tracks the video-rollout change from the dataset boundary through SGLang and back into the training sample. It is intended to be updated whenever the PR's data contract changes.

## Scope

The PR fixes video propagation from Miles to an SGLang rollout engine. It does not add audio support or replace the existing image transport.

Runtime diff against PR base `bf536263` at the time of this revision:

- Production: `+210/-55`
- Tests: `+216`
- Production files changed: 10
- Test files changed: 3

## Bug before this PR

Miles resolved video sources locally with `qwen_vl_utils`, then passed the sampled frames through the Hugging Face processor:

```text
structured prompt
    -> qwen_vl_utils
    -> sampled video tensors
    -> local processor
    -> processor-expanded prompt IDs
```

The rollout request sent those processor-expanded IDs but did not send `video_data`:

```python
{
    "input_ids": processor_expanded_ids,
    # no video_data
}
```

The sampled tensors in `multimodal_inputs["videos"]` are training-side processor inputs, not a JSON-safe HTTP transport. SGLang therefore received a prompt whose IDs represented video content without receiving a video source to process.

Images did not have this gap. Miles already encoded resolved `PIL.Image` objects as PNG data URIs and sent them through `image_data`.

## Current data contract

| Value | Runtime type | Owner | Purpose |
| --- | --- | --- | --- |
| `Sample.multimodal_inputs["images"]` | `list[PIL.Image]` or `None` | Miles/Qwen loader | Local processing and existing image request encoding |
| `Sample.multimodal_inputs["videos"]` | sampled tensors or frame lists | Miles/Qwen loader | Local training processor input |
| `Sample.rollout_video_sources` | `list[str] | None` | Miles dataset | Original video paths, URLs, or data URIs sent to SGLang |
| `Sample.rollout_prompt_ids` | `list[int] | None` | Miles tokenizer | Initial prompt IDs before multimodal expansion |
| `Sample.tokens` | `list[int]` | Miles processor and generation code | Processor-expanded training sequence plus generated/tool tokens |
| `Sample.multimodal_train_inputs` | processor tensor dictionary or `None` | Miles processor | Pixel values and grid tensors consumed during training |

`rollout_video_sources` is deliberately video-specific. If audio is added later, it should receive an explicit contract in that PR instead of being hidden in a generic one-key modality dictionary.

## Data flow after this PR

```text
Dataset receives a structured conversation
    |
    |-- qwen_vl_utils.process_vision_info(...)
    |      -> multimodal_inputs["images"]
    |      -> multimodal_inputs["videos"]
    |
    `-- _extract_rollout_video_sources(...)
           -> rollout_video_sources: list[str] | None

Generation renders the initial prompt once for video samples
    |
    |-- local processor
    |      -> processor_prompt_ids
    |      -> multimodal_train_inputs
    |
    `-- tokenizer
           -> rollout_prompt_ids

Miles canonical context
    [processor_prompt_ids] + [generated/tool tokens]

HTTP request context
    [rollout_prompt_ids] + [same generated/tool tokens]

HTTP media
    image_data = encoded resolved images
    video_data = rollout_video_sources

SGLang
    -> decodes the tokenizer-only prompt
    -> associates media placeholders with image_data/video_data
    -> loads and preprocesses the video
    -> expands its inference-side prompt
    -> generates output IDs

Miles
    -> appends output IDs to Sample.tokens
    -> keeps local processor IDs and tensors for training
```

## Why two prompt ID sequences exist

The local processor and SGLang both need to expand video placeholders, but for different consumers:

- Miles needs locally expanded IDs and processor tensors for training.
- SGLang needs the unexpanded logical placeholder plus the raw video source so it can perform inference-side preprocessing.

For a multi-turn context:

```text
Sample.tokens:
    [expanded initial prompt] [assistant output] [tool response]

Outgoing input_ids:
    [tokenizer initial prompt] [assistant output] [tool response]
```

`build_rollout_input_ids` replaces only the initial prompt prefix. Generated and tool tokens are not decoded and retokenized.

## HTTP boundary

`build_rollout_engine_multimodal_payload` is the only conversion point for rollout media:

```python
image_data: list[str]  # PNG data URIs, existing behavior
video_data: list[str]  # original path, URL, or data URI
```

Before constructing `video_data`, Miles checks:

- `multimodal_inputs` contains only `images` and `videos`.
- Every source is a string.
- The source count equals the number of locally processed logical videos.
- Video item dictionaries do not contain per-item processing options that the request cannot replay.

A local path must be visible from the SGLang worker. URLs and data URIs do not have that shared-filesystem requirement.

## File-by-file production walkthrough

### `miles/utils/types.py`

- Adds `Sample.rollout_video_sources`.
- Adds `Sample.rollout_prompt_ids`.
- Clears `rollout_prompt_ids` on retry while preserving the original video sources.

`Sample.tokens` remains the canonical training-side token sequence.

### `miles/utils/processing_utils.py`

- `_iter_multimodal_content` traverses one structured conversation or a batch.
- `_extract_rollout_video_sources` extracts video strings in prompt occurrence order.
- `process_vision_info_with_video_sources` returns both locally resolved media and raw video sources.
- `process_vision_info` remains the compatibility wrapper for callers that only need local processor inputs.

Source order must match the order used by `qwen_vl_utils`; the later count check detects missing or duplicated sources.

### `miles/utils/data.py`

- Dataset construction stores `multimodal_inputs` and `rollout_video_sources` on the same `Sample`.
- Long-prompt filtering reuses the already resolved `multimodal_inputs` instead of loading the media a second time.

### `miles/rollout/generate_utils/multimodal.py`

- Preserves the existing image-to-data-URI path.
- Validates and maps `rollout_video_sources` to SGLang's `video_data` request key.
- Detects whether a sample is multimodal for prefill batching.
- Replaces the expanded initial prompt prefix with `rollout_prompt_ids` for inference requests.

### `miles/rollout/generate_utils/generate_endpoint_utils.py`

- Renders a structured video prompt once.
- Sends that same rendered text to the local processor and tokenizer.
- Stores processor outputs in the training-side fields.
- Adds optional rollout IDs and video sources to request construction.
- Refuses to initialize canonical `Sample.tokens` from tokenizer-only request IDs.

The last guard prevents an inference representation from silently becoming training state.

### `miles/rollout/generate_hub/single_turn.py`

- Builds request IDs from the rollout representation.
- Sends video sources with the request.
- Initializes `Sample.tokens` with processor IDs before the response updater runs.

Partial rollouts keep their generated suffix while only the initial prompt prefix is replaced for the next request.

### `miles/rollout/generate_hub/multi_turn.py`

- Sends media and rollout IDs on every stateless inference request.
- Preserves canonical context, processor tensors, and rollout prompt IDs when creating per-turn samples.

With `--generate-multi-samples`, accumulated responses remain prompt context in `tokens`, while each returned sample's `response_length`, loss mask, and logprobs cover only that turn. This is existing per-turn training behavior, not lost history.

### `miles/rollout/sglang_rollout.py`

- Reuses the shared prompt-processing code instead of maintaining a second implementation.
- Uses the shared image/video payload builder.
- Sends rollout request IDs while retaining processor IDs in `Sample.tokens`.

### `miles/rollout/generate_utils/prefill_logprobs.py`

- Reconstructs tokenizer-only request IDs for video samples.
- Sends `video_data` during prefill scoring.
- Keeps `logprob_start_len` based on the processor-expanded training sequence.
- Disables batch prefill for image or video samples.

SGLang expands the multimodal prompt before producing token logprobs, so the returned offset still corresponds to the expanded sequence.

### `miles/rollout/generate_utils/sample_utils.py`

- Requires video sources and rollout prompt IDs to agree when multi-turn samples are merged.

## Test walkthrough

### `tests/fast/rollout/generate_utils/test_multimodal.py`

- Verifies image and video HTTP payloads, prompt-prefix replacement, and context-length calculation.
- Rejects unsupported `multimodal_inputs` keys such as `audios`.
- Verifies processor IDs and rollout IDs remain separate.
- Locks existing image-only behavior.
- Exercises a single-turn video request through response integration.

### `tests/fast/utils/test_processing_utils.py`

- Verifies raw video sources retain prompt order.
- Rejects non-string sources and per-item preprocessing options.

### `tests/fast/rollout/generate_utils/test_prefill_logprobs.py`

- Verifies video prefill uses rollout IDs and `video_data` while retaining the expanded logprob offset.

## Review conclusions

The multi-turn `generate_multi_samples` reset intentionally creates per-turn training samples. Copying accumulated response fields into every later sample would change that behavior to cumulative training samples.

Malformed structured prompts are not silently skipped. The accepted shape matches `qwen_vl_utils`, and failing is safer than dropping a video source from the rollout request.

## Non-goals and remaining assumptions

- Audio is not part of this PR.
- Per-item video options such as `fps` are not propagated.
- Local and SGLang processor defaults and versions must remain compatible.
- The PR does not compare Miles-expanded IDs with SGLang-expanded IDs at runtime.
- Local video paths require a filesystem shared with SGLang workers.
- Model-specific SGLang video validation can remain a separate engine-side change.

## Validation

Focused checks for this PR:

```bash
pytest -q \
  tests/fast/rollout/generate_utils/test_multimodal.py \
  tests/fast/utils/test_processing_utils.py \
  tests/fast/rollout/generate_utils/test_prefill_logprobs.py \
  tests/fast/rollout/generate_utils/test_sample_utils.py \
  tests/fast/utils/test_types.py

black --check <changed Python files>
isort --check-only <changed Python files>
python -m compileall -q <changed Python files>
git diff --check
```

Latest focused result: `30 passed`; formatting, import ordering, compilation, and `git diff --check` passed.

The broader single-turn fixture currently has an upstream setup failure involving a missing `sglang_dp_size` argument before it reaches this PR's code.
