# PR 1689 walkthrough

PR: <https://github.com/radixark/miles/pull/1689>

This document tracks the video-rollout change from the dataset boundary through SGLang and back into the training sample. It is intended to be updated whenever the PR's data contract changes.

## Scope

The PR fixes video propagation from Miles to an SGLang rollout engine. It does not add audio support or replace the existing image transport.

Runtime diff against PR base `bf536263` at the time of this revision:

- Production: `+206/-54`
- Tests: `+219`
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
    `-- extract_rollout_video_sources(...)
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
- `extract_rollout_video_sources` extracts video strings in prompt occurrence order.
- `process_vision_info` returns locally resolved image and video inputs.

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

## Exhaustive line-by-line review snapshot

This section is the cumulative source diff from PR base `bf536263` to code snapshot `2d08c6f`. It excludes this walkthrough file itself. Every production and test line changed by the PR appears below.

Read each hunk header as `@@ -old_start,old_count +new_start,new_count @@`: `-` lines are the pre-PR code and `+` lines are the current PR code. The function shown in the hunk header identifies the scope; the earlier walkthrough sections explain how each scope participates in the dataflow.

### Recommended review order

1. `miles/utils/types.py`: persistent `Sample` contract.
2. `miles/utils/processing_utils.py` and `miles/utils/data.py`: raw-source extraction and dataset ownership.
3. `miles/rollout/generate_utils/multimodal.py`: HTTP media validation and prompt-prefix conversion.
4. `miles/rollout/generate_utils/generate_endpoint_utils.py`: creation of training IDs versus rollout IDs.
5. Single-turn, multi-turn, legacy rollout, and prefill consumers.
6. Sample merging and the three focused test files.

The production diff is shown in repository path order rather than review order so it can be compared directly with `git diff`.

### Current-head line map

These are the current code lines containing PR changes at snapshot `2d08c6f`. Deleted pre-PR lines have no current line number and appear in the exact diff below.

| File | Current changed lines |
| --- | --- |
| `miles/utils/types.py` | 18-20, 222-224, 227 |
| `miles/utils/processing_utils.py` | 154-190, 192, 195-197, 202 |
| `miles/utils/data.py` | 99, 217, 222, 226, 234 |
| `miles/rollout/generate_utils/multimodal.py` | 1-57 |
| `miles/rollout/generate_utils/generate_endpoint_utils.py` | 12, 15-26, 33-39, 46-48, 51-52, 60-61, 71, 79, 84-91, 97-98 |
| `miles/rollout/generate_hub/single_turn.py` | 9, 37, 39-44, 50-52 |
| `miles/rollout/generate_hub/multi_turn.py` | 12, 50-58, 66-68, 70-72 |
| `miles/rollout/sglang_rollout.py` | 26, 29-35, 145, 175, 179, 181, 183-184 |
| `miles/rollout/generate_utils/prefill_logprobs.py` | 11-16, 30-35, 37, 54, 62 |
| `miles/rollout/generate_utils/sample_utils.py` | 128-129 |
| `tests/fast/rollout/generate_utils/test_multimodal.py` | 1-148 |
| `tests/fast/rollout/generate_utils/test_prefill_logprobs.py` | 170-188 |
| `tests/fast/utils/test_processing_utils.py` | 1-52 |

### Exact production diff

```diff
diff --git a/miles/rollout/generate_hub/multi_turn.py b/miles/rollout/generate_hub/multi_turn.py
index 97814ec..532b10f 100644
--- a/miles/rollout/generate_hub/multi_turn.py
+++ b/miles/rollout/generate_hub/multi_turn.py
@@ -11,0 +12 @@ from miles.rollout.generate_utils.generate_endpoint_utils import (
+    compute_rollout_input_ids,
@@ -49 +50,9 @@ async def generate(input: GenerateFnInput) -> GenerateFnOutput:
-        payload, halt_status = compute_request_payload(args, sample.tokens, input.sampling_params)
+        rollout_input_ids = compute_rollout_input_ids(sample, sample.tokens, prompt_tokens_ids)
+        payload, halt_status = compute_request_payload(
+            args,
+            sample.tokens,
+            input.sampling_params,
+            multimodal_inputs=sample.multimodal_inputs,
+            rollout_video_sources=sample.rollout_video_sources,
+            rollout_input_ids=rollout_input_ids,
+        )
@@ -56,0 +66,3 @@ async def generate(input: GenerateFnInput) -> GenerateFnOutput:
+            context_tokens = sample.tokens
+            multimodal_train_inputs = sample.multimodal_train_inputs
+            rollout_prompt_ids = sample.rollout_prompt_ids
@@ -57,0 +70,3 @@ async def generate(input: GenerateFnInput) -> GenerateFnOutput:
+            sample.tokens = context_tokens.copy()
+            sample.multimodal_train_inputs = multimodal_train_inputs
+            sample.rollout_prompt_ids = rollout_prompt_ids
diff --git a/miles/rollout/generate_hub/single_turn.py b/miles/rollout/generate_hub/single_turn.py
index 5c0a15b..d1ab600 100644
--- a/miles/rollout/generate_hub/single_turn.py
+++ b/miles/rollout/generate_hub/single_turn.py
@@ -8,0 +9 @@ from miles.rollout.generate_utils.generate_endpoint_utils import (
+    compute_rollout_input_ids,
@@ -35,0 +37 @@ async def generate(input: GenerateFnInput) -> GenerateFnOutput:
+    rollout_input_ids = compute_rollout_input_ids(sample, input_ids, prompt_ids)
@@ -37 +39,6 @@ async def generate(input: GenerateFnInput) -> GenerateFnOutput:
-        args, input_ids=input_ids, sampling_params=sampling_params, multimodal_inputs=sample.multimodal_inputs
+        args,
+        input_ids=input_ids,
+        rollout_input_ids=rollout_input_ids,
+        sampling_params=sampling_params,
+        multimodal_inputs=sample.multimodal_inputs,
+        rollout_video_sources=sample.rollout_video_sources,
@@ -42,0 +50,3 @@ async def generate(input: GenerateFnInput) -> GenerateFnOutput:
+    if not sample.tokens:
+        sample.tokens = list(prompt_ids)
+
diff --git a/miles/rollout/generate_utils/generate_endpoint_utils.py b/miles/rollout/generate_utils/generate_endpoint_utils.py
index d50098e..db4fbe0 100644
--- a/miles/rollout/generate_utils/generate_endpoint_utils.py
+++ b/miles/rollout/generate_utils/generate_endpoint_utils.py
@@ -12 +12 @@ from miles.utils.lora import LORA_ADAPTER_NAME, is_lora_enabled
-from miles.utils.processing_utils import encode_image_for_rollout_engine
+from miles.utils.processing_utils import call_processor
@@ -14,0 +15,12 @@ from miles.utils.types import Sample
+from .multimodal import build_rollout_engine_multimodal_payload, build_rollout_input_ids
+
+
+def _render_prompt(tokenizer, prompt, tools=None) -> str:
+    if not isinstance(prompt, str):
+        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, tools=tools)
+    return prompt
+
+
+def _tokenize_prompt(tokenizer, prompt, tools=None) -> list[int]:
+    return tokenizer.encode(_render_prompt(tokenizer, prompt, tools=tools), add_special_tokens=False)
+
@@ -21 +33,4 @@ def compute_prompt_ids_from_sample(state, sample, tools=None):
-        processor_output = state.processor(text=prompt, **sample.multimodal_inputs)
+        rollout_prompt = _render_prompt(state.tokenizer, prompt, tools=tools) if sample.rollout_video_sources else None
+        processor_output = call_processor(
+            state.processor, rollout_prompt if rollout_prompt is not None else prompt, sample.multimodal_inputs
+        )
@@ -22,0 +38,2 @@ def compute_prompt_ids_from_sample(state, sample, tools=None):
+        prompt_ids = prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else list(prompt_ids)
+        prompt_ids = [int(token_id) for token_id in prompt_ids]
@@ -28,0 +46,3 @@ def compute_prompt_ids_from_sample(state, sample, tools=None):
+        sample.rollout_prompt_ids = (
+            state.tokenizer.encode(rollout_prompt, add_special_tokens=False) if rollout_prompt is not None else None
+        )
@@ -30,5 +49,0 @@ def compute_prompt_ids_from_sample(state, sample, tools=None):
-    else:
-        if not isinstance(prompt, str):
-            prompt = state.tokenizer.apply_chat_template(
-                prompt, tokenize=False, add_generation_prompt=True, tools=tools
-            )
@@ -36 +51,2 @@ def compute_prompt_ids_from_sample(state, sample, tools=None):
-        return state.tokenizer.encode(prompt, add_special_tokens=False)
+    sample.rollout_prompt_ids = None
+    return _tokenize_prompt(state.tokenizer, prompt, tools=tools)
@@ -43,0 +60,2 @@ def compute_request_payload(
+    rollout_video_sources: list[str] | None = None,
+    rollout_input_ids: list[int] | None = None,
@@ -53 +71 @@ def compute_request_payload(
-        "input_ids": input_ids,
+        "input_ids": rollout_input_ids if rollout_input_ids is not None else input_ids,
@@ -61,2 +79 @@ def compute_request_payload(
-    if image_data := (multimodal_inputs or {}).get("images"):
-        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]
+    payload.update(build_rollout_engine_multimodal_payload(multimodal_inputs, rollout_video_sources))
@@ -66,0 +84,8 @@ def compute_request_payload(
+def compute_rollout_input_ids(sample: Sample, input_ids: list[int], processor_prompt_ids: list[int]) -> list[int]:
+    return build_rollout_input_ids(
+        input_ids,
+        processor_prompt_ids=processor_prompt_ids,
+        rollout_prompt_ids=sample.rollout_prompt_ids,
+    )
+
+
@@ -71,0 +97,2 @@ async def update_sample_from_response(
+        if sample.rollout_prompt_ids is not None:
+            raise ValueError("Canonical processor-expanded prompt tokens must be initialized before video rollout")
diff --git a/miles/rollout/generate_utils/multimodal.py b/miles/rollout/generate_utils/multimodal.py
new file mode 100644
index 0000000..dc75db6
--- /dev/null
+++ b/miles/rollout/generate_utils/multimodal.py
@@ -0,0 +1,57 @@
+from typing import Any
+
+from miles.utils.processing_utils import encode_image_for_rollout_engine
+
+
+def build_rollout_engine_multimodal_payload(
+    multimodal_inputs: dict[str, Any] | None,
+    rollout_video_sources: list[str] | None,
+) -> dict[str, list[str]]:
+    multimodal_inputs = multimodal_inputs or {}
+    unsupported_keys = multimodal_inputs.keys() - {"images", "videos"}
+    if unsupported_keys:
+        raise ValueError(f"Unsupported multimodal input keys: {sorted(unsupported_keys)}")
+
+    payload = {}
+    if image_data := multimodal_inputs.get("images"):
+        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]
+
+    processed_videos = multimodal_inputs.get("videos")
+    if rollout_video_sources is not None and any(not isinstance(source, str) for source in rollout_video_sources):
+        raise TypeError("Rollout video sources must be paths, URLs, or data URIs")
+    processed_video_count = len(processed_videos) if processed_videos is not None else 0
+    rollout_video_count = len(rollout_video_sources) if rollout_video_sources is not None else 0
+    if processed_video_count != rollout_video_count:
+        raise ValueError(
+            "Video processor inputs and rollout sources must have the same length: "
+            f"processed={processed_video_count}, rollout={rollout_video_count}"
+        )
+    if rollout_video_sources:
+        payload["video_data"] = list(rollout_video_sources)
+
+    return payload
+
+
+def has_multimodal_inputs(
+    multimodal_inputs: dict[str, Any] | None,
+    rollout_video_sources: list[str] | None,
+) -> bool:
+    processor_media = ((multimodal_inputs or {}).get(key) for key in ("images", "videos"))
+    return any(value is not None and len(value) > 0 for value in processor_media) or bool(rollout_video_sources)
+
+
+def build_rollout_input_ids(
+    input_ids: list[int],
+    *,
+    processor_prompt_ids: list[int],
+    rollout_prompt_ids: list[int] | None,
+) -> list[int]:
+    input_ids = list(input_ids)
+    processor_prompt_ids = list(processor_prompt_ids)
+    if rollout_prompt_ids is None:
+        return input_ids
+
+    if input_ids[: len(processor_prompt_ids)] != processor_prompt_ids:
+        raise ValueError("Cannot build rollout_input_ids: input IDs do not start with the processed prompt IDs")
+
+    return list(rollout_prompt_ids) + input_ids[len(processor_prompt_ids) :]
diff --git a/miles/rollout/generate_utils/prefill_logprobs.py b/miles/rollout/generate_utils/prefill_logprobs.py
index d11752f..651b130 100644
--- a/miles/rollout/generate_utils/prefill_logprobs.py
+++ b/miles/rollout/generate_utils/prefill_logprobs.py
@@ -9 +8,0 @@ from miles.utils.lora import LORA_ADAPTER_NAME, is_lora_enabled
-from miles.utils.processing_utils import encode_image_for_rollout_engine
@@ -11,0 +11,6 @@ from miles.utils.types import Sample
+from .multimodal import (
+    build_rollout_engine_multimodal_payload,
+    build_rollout_input_ids,
+    has_multimodal_inputs,
+)
+
@@ -24,0 +30,6 @@ def _build_prefill_scoring_payload(
+    processor_prompt_ids = sample.tokens[:prompt_len]
+    rollout_input_ids = build_rollout_input_ids(
+        sample.tokens,
+        processor_prompt_ids=processor_prompt_ids,
+        rollout_prompt_ids=sample.rollout_prompt_ids,
+    )
@@ -26 +37 @@ def _build_prefill_scoring_payload(
-        "input_ids": sample.tokens,
+        "input_ids": rollout_input_ids,
@@ -43,3 +54 @@ def _build_prefill_scoring_payload(
-    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
-        image_data = sample.multimodal_inputs["images"]
-        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]
+    payload.update(build_rollout_engine_multimodal_payload(sample.multimodal_inputs, sample.rollout_video_sources))
@@ -53 +62 @@ def _can_batch_prefill_score(args: Any, samples: list[Sample]) -> bool:
-    return not any(sample.multimodal_inputs and sample.multimodal_inputs.get("images") for sample in samples)
+    return not any(has_multimodal_inputs(sample.multimodal_inputs, sample.rollout_video_sources) for sample in samples)
diff --git a/miles/rollout/generate_utils/sample_utils.py b/miles/rollout/generate_utils/sample_utils.py
index 1594dac..fa21b13 100644
--- a/miles/rollout/generate_utils/sample_utils.py
+++ b/miles/rollout/generate_utils/sample_utils.py
@@ -127,0 +128,2 @@ def _merge_sample_pair(a: Sample, b: Sample, tokenizer) -> Sample:
+            rollout_video_sources=_merge_equal_value("rollout_video_sources"),
+            rollout_prompt_ids=_merge_equal_value("rollout_prompt_ids"),
diff --git a/miles/rollout/sglang_rollout.py b/miles/rollout/sglang_rollout.py
index 64662ed..7280845 100644
--- a/miles/rollout/sglang_rollout.py
+++ b/miles/rollout/sglang_rollout.py
@@ -26,6 +26 @@ from miles.utils.misc import SingletonMeta, load_function
-from miles.utils.processing_utils import (
-    call_processor,
-    encode_image_for_rollout_engine,
-    load_processor,
-    load_tokenizer,
-)
+from miles.utils.processing_utils import load_processor, load_tokenizer
@@ -34 +29,6 @@ from miles.utils.types import Sample
-from .generate_utils.generate_endpoint_utils import get_indexer_topk_from_response
+from .generate_utils.generate_endpoint_utils import (
+    compute_prompt_ids_from_sample,
+    compute_rollout_input_ids,
+    get_indexer_topk_from_response,
+)
+from .generate_utils.multimodal import build_rollout_engine_multimodal_payload
@@ -145,9 +145 @@ async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, A
-    if state.processor and sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
-        processor_output = call_processor(state.processor, sample.prompt, sample.multimodal_inputs)
-        prompt_ids = processor_output["input_ids"][0]
-        prompt_ids = prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else list(prompt_ids)
-        sample.multimodal_train_inputs = {
-            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
-        } or None
-    else:
-        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)
+    prompt_ids = compute_prompt_ids_from_sample(state, sample)
@@ -183,3 +175 @@ async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, A
-    if sample.multimodal_inputs and sample.multimodal_inputs["images"]:
-        image_data = sample.multimodal_inputs["images"]
-        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]
+    payload.update(build_rollout_engine_multimodal_payload(sample.multimodal_inputs, sample.rollout_video_sources))
@@ -189 +179 @@ async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, A
-        payload["input_ids"] = sample.tokens
+        input_ids = sample.tokens
@@ -191 +181 @@ async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, A
-        payload["input_ids"] = prompt_ids
+        input_ids = prompt_ids
@@ -193 +183,2 @@ async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, A
-            sample.tokens = prompt_ids
+            sample.tokens = prompt_ids.copy()
+    payload["input_ids"] = compute_rollout_input_ids(sample, input_ids, prompt_ids)
diff --git a/miles/utils/data.py b/miles/utils/data.py
index fd79cc5..1ecf47b 100644
--- a/miles/utils/data.py
+++ b/miles/utils/data.py
@@ -99,4 +99 @@ def filter_long_prompt(origin_samples: list[Sample], tokenizer, processor, max_l
-            from miles.utils.processing_utils import process_vision_info
-
-            multimodal_inputs = process_vision_info(sample.prompt, processor)
-            processor_output = call_processor(processor, sample.prompt, multimodal_inputs)
+            processor_output = call_processor(processor, sample.prompt, sample.multimodal_inputs)
@@ -220 +217 @@ class Dataset:
-                from miles.utils.processing_utils import process_vision_info
+                from miles.utils.processing_utils import extract_rollout_video_sources, process_vision_info
@@ -224,0 +222 @@ class Dataset:
+                rollout_video_sources = extract_rollout_video_sources(prompt)
@@ -227,0 +226 @@ class Dataset:
+                rollout_video_sources = None
@@ -234,0 +234 @@ class Dataset:
+                    rollout_video_sources=rollout_video_sources,
diff --git a/miles/utils/processing_utils.py b/miles/utils/processing_utils.py
index 855a06a..40033ea 100644
--- a/miles/utils/processing_utils.py
+++ b/miles/utils/processing_utils.py
@@ -153,0 +154,37 @@ def load_processor(name_or_path: str, **kwargs):
+def _iter_multimodal_content(prompt):
+    """Yield structured content items from one conversation or a batch."""
+    if not isinstance(prompt, list) or not prompt:
+        return
+
+    conversations = [prompt] if isinstance(prompt[0], dict) else prompt
+    for conversation in conversations:
+        for message in conversation:
+            content = message.get("content")
+            if not isinstance(content, list):
+                continue
+            for item in content:
+                if isinstance(item, dict):
+                    yield item
+
+
+def extract_rollout_video_sources(prompt) -> list[str] | None:
+    video_sources = []
+    for item in _iter_multimodal_content(prompt):
+        if item.get("type") != "video":
+            continue
+
+        unsupported_options = set(item) - {"type", "video"}
+        if unsupported_options:
+            raise ValueError(
+                "Video rollout cannot replay per-item processing options; configure matching local and rollout "
+                f"processor defaults instead: {sorted(unsupported_options)}"
+            )
+
+        source = item.get("video")
+        if not isinstance(source, str):
+            raise TypeError("Video rollout input must be a path, URL, or data URI")
+        video_sources.append(source)
+
+    return video_sources or None
+
+
@@ -155 +192 @@ def process_vision_info(prompt, processor):
-    # TODO: temporary solution, will write image utils for miles later
+    # TODO: temporary solution, will write model-independent media utils later
@@ -158,2 +195,3 @@ def process_vision_info(prompt, processor):
-    if hasattr(processor.image_processor, "patch_size"):
-        image_patch_size = processor.image_processor.patch_size
+    image_processor = getattr(processor, "image_processor", None)
+    if image_processor is not None and hasattr(image_processor, "patch_size"):
+        image_patch_size = image_processor.patch_size
@@ -164,2 +202 @@ def process_vision_info(prompt, processor):
-    multimodal_inputs = {"images": images, "videos": videos}
-    return multimodal_inputs
+    return {"images": images, "videos": videos}
diff --git a/miles/utils/types.py b/miles/utils/types.py
index cd7637d..ead1bff 100644
--- a/miles/utils/types.py
+++ b/miles/utils/types.py
@@ -18 +18,3 @@ class Sample:
-    multimodal_inputs: dict[str, Any] = None  # raw multimodal data, e.g. images, videos, etc.
+    multimodal_inputs: dict[str, Any] = None
+    rollout_video_sources: list[str] | None = None
+    rollout_prompt_ids: list[int] | None = None
@@ -220,2 +222,3 @@ class Sample:
-        multimodal_inputs, metadata, generate_function_path, session_id) and
-        restores everything else to dataclass defaults.
+        multimodal_inputs, rollout_video_sources, metadata,
+        generate_function_path, session_id) and restores everything else to
+        dataclass defaults.
@@ -223,0 +227 @@ class Sample:
+        self.rollout_prompt_ids = None
```

### Exact test diff

The first new test file covers the shared request contract and single-turn integration. The prefill addition locks the expanded training offset versus unexpanded request IDs. The processing tests lock raw-source order and fail-fast validation.

```diff
diff --git a/tests/fast/rollout/generate_utils/test_multimodal.py b/tests/fast/rollout/generate_utils/test_multimodal.py
new file mode 100644
index 0000000..d7e192d
--- /dev/null
+++ b/tests/fast/rollout/generate_utils/test_multimodal.py
@@ -0,0 +1,148 @@
+from types import SimpleNamespace
+
+import pytest
+from PIL import Image
+
+from miles.rollout.base_types import GenerateFnInput
+from miles.rollout.generate_hub import single_turn
+from miles.rollout.generate_utils.generate_endpoint_utils import (
+    compute_prompt_ids_from_sample,
+    compute_request_payload,
+)
+from miles.rollout.generate_utils.multimodal import (
+    build_rollout_engine_multimodal_payload,
+    build_rollout_input_ids,
+)
+from miles.utils.types import Sample
+
+PROCESSOR_PROMPT_IDS = [100, 101, 102]
+ROLLOUT_PROMPT_IDS = [1, 2]
+
+
+class _Processor:
+    def __init__(self):
+        self.text = None
+
+    def __call__(self, text, **kwargs):
+        self.text = text
+        return {"input_ids": [PROCESSOR_PROMPT_IDS], "pixel_values_videos": "train-only"}
+
+
+class _Tokenizer:
+    def apply_chat_template(self, prompt, **kwargs):
+        return "<video>rendered prompt"
+
+    def encode(self, text, add_special_tokens):
+        assert add_special_tokens is False
+        return ROLLOUT_PROMPT_IDS
+
+
+def _args(**overrides):
+    defaults = dict(
+        sglang_router_ip="127.0.0.1",
+        sglang_router_port=30000,
+        rollout_max_response_len=16,
+        rollout_max_context_len=None,
+        use_rollout_routing_replay=False,
+        use_rollout_indexer_replay=False,
+        lora_rank=0,
+        lora_adapter_path=None,
+        sglang_speculative_algorithm=None,
+    )
+    return SimpleNamespace(**(defaults | overrides))
+
+
+def _video_sample():
+    return Sample(
+        prompt="<video>describe it",
+        multimodal_inputs={"videos": [object()]},
+        rollout_video_sources=["video.mp4"],
+    )
+
+
+def test_multimodal_request_contract():
+    image = Image.new("RGB", (2, 2), color="red")
+    media_payload = build_rollout_engine_multimodal_payload(
+        {"images": [image], "videos": [object()]},
+        ["https://example.test/video.mp4"],
+    )
+    rollout_ids = build_rollout_input_ids(
+        PROCESSOR_PROMPT_IDS + [20],
+        processor_prompt_ids=PROCESSOR_PROMPT_IDS,
+        rollout_prompt_ids=ROLLOUT_PROMPT_IDS,
+    )
+    request, status = compute_request_payload(
+        _args(rollout_max_context_len=5),
+        input_ids=PROCESSOR_PROMPT_IDS,
+        rollout_input_ids=ROLLOUT_PROMPT_IDS,
+        sampling_params={"max_new_tokens": 10},
+    )
+
+    assert media_payload["image_data"][0].startswith("data:image/png;base64,")
+    assert media_payload["video_data"] == ["https://example.test/video.mp4"]
+    assert rollout_ids == ROLLOUT_PROMPT_IDS + [20]
+    assert status is None
+    assert request["input_ids"] == ROLLOUT_PROMPT_IDS
+    assert request["sampling_params"]["max_new_tokens"] == 2
+
+    with pytest.raises(ValueError, match="same length"):
+        build_rollout_engine_multimodal_payload({"videos": [object()]}, None)
+    with pytest.raises(ValueError, match="audios"):
+        build_rollout_engine_multimodal_payload({"audios": [object()]}, None)
+
+
+def test_prompt_processing_keeps_training_and_rollout_ids_separate():
+    sample = _video_sample()
+    sample.prompt = [{"role": "user", "content": [{"type": "video", "video": "video.mp4"}]}]
+    processor = _Processor()
+    state = SimpleNamespace(processor=processor, tokenizer=_Tokenizer())
+
+    prompt_ids = compute_prompt_ids_from_sample(state, sample)
+
+    assert prompt_ids == PROCESSOR_PROMPT_IDS
+    assert sample.rollout_prompt_ids == ROLLOUT_PROMPT_IDS
+    assert sample.multimodal_train_inputs == {"pixel_values_videos": "train-only"}
+    assert processor.text == "<video>rendered prompt"
+
+
+def test_image_only_keeps_the_existing_request_contract():
+    image = Image.new("RGB", (2, 2), color="red")
+    sample = Sample(prompt="<image>describe it", multimodal_inputs={"images": [image]})
+    state = SimpleNamespace(processor=_Processor(), tokenizer=_Tokenizer())
+    prompt_ids = compute_prompt_ids_from_sample(state, sample)
+
+    payload, _ = compute_request_payload(
+        _args(), prompt_ids, {"max_new_tokens": 4}, multimodal_inputs=sample.multimodal_inputs
+    )
+
+    assert sample.rollout_prompt_ids is None
+    assert payload["input_ids"] == PROCESSOR_PROMPT_IDS
+    assert payload["image_data"][0].startswith("data:image/png;base64,")
+
+
+@pytest.mark.asyncio
+async def test_single_turn_sends_rollout_ids_but_keeps_processor_ids(monkeypatch):
+    requests = []
+
+    async def fake_post(url, payload):
+        requests.append(payload)
+        return {
+            "text": "answer",
+            "meta_info": {
+                "output_token_logprobs": [(-0.1, 20)],
+                "finish_reason": {"type": "stop"},
+            },
+        }
+
+    monkeypatch.setattr(single_turn, "post", fake_post)
+    args = _args()
+    sample = _video_sample()
+    state = SimpleNamespace(args=args, processor=_Processor(), tokenizer=_Tokenizer())
+
+    output = await single_turn.generate(
+        GenerateFnInput(state=state, sample=sample, sampling_params={"max_new_tokens": 4}, evaluation=False)
+    )
+
+    assert requests[0]["input_ids"] == ROLLOUT_PROMPT_IDS
+    assert requests[0]["video_data"] == ["video.mp4"]
+    assert output.samples.tokens == PROCESSOR_PROMPT_IDS + [20]
diff --git a/tests/fast/rollout/generate_utils/test_prefill_logprobs.py b/tests/fast/rollout/generate_utils/test_prefill_logprobs.py
index aa736a8..307c92c 100644
--- a/tests/fast/rollout/generate_utils/test_prefill_logprobs.py
+++ b/tests/fast/rollout/generate_utils/test_prefill_logprobs.py
@@ -169,0 +170,19 @@ async def test_recompute_samples_batches_by_logprob_start_len(monkeypatch):
+
+
+def test_video_prefill_uses_rollout_prompt_ids_and_canonical_logprob_offset():
+    sample = Sample(
+        tokens=[100, 101, 102, 20, 21],
+        response_length=2,
+        status=Sample.Status.COMPLETED,
+        multimodal_inputs={"videos": [object()]},
+        rollout_video_sources=["video.mp4"],
+        rollout_prompt_ids=[1, 2],
+    )
+    args = SimpleNamespace(sglang_enable_lora=False, sglang_router_policy="round_robin")
+
+    payload = prefill_logprobs._build_prefill_scoring_payload(args, sample, {})
+
+    assert payload["input_ids"] == [1, 2, 20, 21]
+    assert payload["video_data"] == ["video.mp4"]
+    assert payload["logprob_start_len"] == 2
+    assert prefill_logprobs._can_batch_prefill_score(args, [sample]) is False
diff --git a/tests/fast/utils/test_processing_utils.py b/tests/fast/utils/test_processing_utils.py
new file mode 100644
index 0000000..240e6b4
--- /dev/null
+++ b/tests/fast/utils/test_processing_utils.py
@@ -0,0 +1,52 @@
+import sys
+from types import SimpleNamespace
+
+import pytest
+
+from miles.utils.processing_utils import extract_rollout_video_sources, process_vision_info
+
+
+def test_vision_inputs_and_rollout_sources_follow_prompt_order(monkeypatch):
+    calls = {}
+
+    def fake_process_vision_info(prompt, image_patch_size):
+        calls["prompt"] = prompt
+        calls["image_patch_size"] = image_patch_size
+        return ["resolved-image"], ["processed-video-1", "processed-video-2"]
+
+    monkeypatch.setitem(
+        sys.modules,
+        "qwen_vl_utils",
+        SimpleNamespace(process_vision_info=fake_process_vision_info),
+    )
+    prompt = [
+        {
+            "role": "user",
+            "content": [
+                {"type": "video", "video": "first.mp4"},
+                {"type": "image", "image": "image.png"},
+                {"type": "video", "video": "https://example.test/second.mp4"},
+            ],
+        }
+    ]
+    processor = SimpleNamespace(image_processor=SimpleNamespace(patch_size=16))
+
+    rollout_video_sources = extract_rollout_video_sources(prompt)
+    processor_inputs = process_vision_info(prompt, processor)
+
+    assert processor_inputs == {
+        "images": ["resolved-image"],
+        "videos": ["processed-video-1", "processed-video-2"],
+    }
+    assert rollout_video_sources == ["first.mp4", "https://example.test/second.mp4"]
+    assert calls == {"prompt": prompt, "image_patch_size": 16}
+
+
+def test_extract_rollout_video_sources_rejects_inputs_the_engine_cannot_replay():
+    invalid_items = [
+        ({"type": "video", "video": ["frame-1.png"]}, TypeError),
+        ({"type": "video", "video": "video.mp4", "fps": 4}, ValueError),
+    ]
+    for item, error_type in invalid_items:
+        with pytest.raises(error_type):
+            extract_rollout_video_sources([{"role": "user", "content": [item]}])
```

### Completeness check

At snapshot `2d08c6f`, the exhaustive sections above contain the same 13 files reported by:

```bash
git diff --stat bf536263c4a7cc8978781c420a3424fa2eb12fb9...2d08c6f -- . ':!1689_walkthrough.md'
```

Expected source/test total:

```text
13 files changed, 425 insertions(+), 54 deletions(-)
```
