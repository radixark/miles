---
title: Customization
description: The plug-points where you can drop in your own Python without forking Miles.
---
Most of Miles's behavior can be replaced with user-supplied Python by passing a
`--*-path` flag. This page lists every such hook, the function signature it expects,
and the default it replaces.

## At a glance

| Stage | Flag | Replaces |
|---|---|---|
| **Rollout** | `--rollout-function-path` | The whole rollout loop |
| | `--custom-agent-function-path` | The agent-environment loop inside a TITO session |
| | `--custom-generate-function-path` | A single sample's generation |
| | `--data-source-path` | How prompts are loaded |
| | `--eval-function-path` | The eval rollout |
| **Session** | `--session-message-matcher` | Prefix-replay equivalence |
| **Reward** | `--custom-rm-path` | Reward computation |
| | `--custom-reward-post-process-path` | Reward normalization |
| **Filtering** | `--dynamic-sampling-filter-path` | Per-group filter (DAPO) |
| | `--buffer-filter-path` | Buffer dequeue filter |
| | `--rollout-sample-filter-path` | Per-sample loss filter |
| | `--rollout-all-samples-process-path` | Inspect all samples post-rollout |
| | `--rollout-data-postprocess-path` | Mutate samples post-logprob |
| **Training** | `--custom-loss-function-path` | The loss formula |
| | `--custom-tis-function-path` | Importance sampling correction |
| | `--custom-pg-loss-reducer-function-path` | Loss reduction (Dr.GRPO) |
| | `--custom-convert-samples-to-train-data-path` | Sample to tensor batch |
| **Megatron hooks** | `--custom-megatron-init-path` | After Megatron init |
| | `--custom-megatron-before-log-prob-hook-path` | Before logprob compute |
| | `--custom-megatron-before-train-step-hook-path` | Before each train step |
| | `--custom-megatron-post-save-hook-path` | After each checkpoint save |
| **Logging** | `--custom-rollout-log-function-path` | Train-rollout logging |
| | `--custom-eval-rollout-log-function-path` | Eval-rollout logging |
| **Model** | `--custom-model-provider-path` | Megatron model factory |

---

## Rollout

### `--rollout-function-path`

Replace the entire rollout function. Use this only for fundamentally different flows
such as multi-agent co-evolution.

```python
def generate_rollout(args, rollout_id, data_source, evaluation=False) \
        -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    ...
```

**Default:** `miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn`; use `miles.rollout.sglang_rollout.generate_rollout` under `MILES_USE_LEGACY_ROLLOUT_V1=1`.

Plain functions with the signature above are wrapped in a legacy adapter. A class-based rollout
function needs to subclass `miles.rollout.base_types.BaseRolloutFn`.

**Reference:** [`examples/experimental/multi_agent/rollout_with_multi_agents.py`](https://github.com/radixark/miles/blob/main/examples/experimental/multi_agent/rollout_with_multi_agents.py).

### `--custom-generate-function-path`

Replace just the generation step inside the default rollout. Most tool-use, RAG, and
multi-turn workflows live here.

```python
async def custom_generate(args, sample: Sample, sampling_params: dict) -> Sample:
    ...
```

The hook also accepts the `GenerateFnInput -> GenerateFnOutput` form; both
signatures load through the same adapter. See
[Generate Endpoint](/user-guide/generate-endpoint) for the full contract.

**Reference:** [`examples/experimental/search-r1/generate_with_search.py`](https://github.com/radixark/miles/blob/main/examples/experimental/search-r1/generate_with_search.py).


### `--custom-agent-function-path`

Enabled when you set `--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate`.
Use `--custom-agent-function-path` to specify the async agent or environment loop
that sends OpenAI-compatible chat requests through Miles' TITO session server.


```python
async def run_agent(
    base_url: str,
    prompt,
    request_kwargs: dict,
    metadata: dict,
    **kwargs,
) -> dict | None:
    ...
```

See [Agentic Rollout (TITO)](/user-guide/agentic-rollout) for the full wiring and
message/token ownership contract.


### `--data-source-path`

```python
class CustomDataSource(DataSource):
    def get_samples(self, num_samples) -> list[list[Sample]]: ...
    def save(self, directory: Path) -> None: ...
    def load(self, directory: Path) -> None: ...
```

**Default:** `miles.rollout.data_source.RolloutDataSource`.

### Training dataset mixing

Use `--rollout-data-config mix.yaml --num-rollout 100` to train from several
datasets without merging their files. This selects
`miles.rollout.mixed_data_source.MixedRolloutDataSource` automatically:

```yaml
datasets:
  math:
    path: /data/math.jsonl
    samples_per_step: 3
    max_prompt_len: 4096
  science:
    path: /data/science.parquet
    samples_per_step: 5
    input_key: question
    label_key: answer
```

`datasets` accepts a mapping keyed by name or a list whose entries include `name`.
Names must be unique. Each entry needs `path` and a positive integer
`samples_per_step`. The latter is a **weight**, not an exact count: a request for
eight prompts with the weights above draws three math and five science prompts.
Smaller or non-divisible requests carry their allocation remainder into later
requests, without floating-point rounding drift. The selected prompts are shuffled
together before each is copied `--n-samples-per-prompt` times, so each GRPO group
still contains responses to one prompt.

Ratios apply to **prompts drawn**, including additional oversampling requests.
Dynamic filtering, retries and asynchronous completion can change the proportions
in the final accepted training batch. The mixer does not refill a filtered group
from its original dataset to enforce accepted-batch quotas.

Each dataset has an independent cursor and epoch. `--rollout-shuffle` controls
per-dataset epoch shuffling; the mixed prompt order is always shuffled with the
rollout seed. Requests can wrap a small dataset multiple times. A dataset with no
prompts after filtering fails at startup. Every sample carries its source name in
`metadata.dataset_name`; the configured name overrides a conflicting row value.

Dataset loading uses the existing `--input-key`, `--label-key`, `--tool-key` and
`--metadata-key` defaults. Override those fields with `input_key`, `label_key`,
`tool_key` and `metadata_key` in an entry, or in an optional top-level `defaults`
mapping. `max_prompt_len` also supports a default; it can only shorten the global
`--rollout-max-prompt-len` bound. An omitted or null limit uses the global bound.
Chat templating and multimodal options remain global. Structured conversation
prompts require `--apply-chat-template` when a prompt limit is set; the mixer
rejects a limit that the loader cannot enforce. Relative paths are resolved from
the working directory, like `--prompt-data`, and existing row-slice paths are supported.

An optional `sha256` field pins the bytes of each dataset file. A mismatch stops
startup and reports both hashes. Without a pin, startup logs a warning containing
the computed digest to paste into the config. For a sliced path, the digest covers
the entire underlying file. Directory datasets are not supported.

Rollout checkpoints save every dataset's cursor and epoch, allocation remainder,
mixing RNG state, data hashes and global sample/group counters. Resume restores the
exact next prompt draws and rejects changed data or sampling/preprocessing
configuration, even when the original config had no checksum pins. Adding a pin
that matches the same file does not invalidate a checkpoint. This guarantees
source continuation, not deterministic inference or async completion order.

This mode cannot be combined with `--prompt-data`,
`--disable-rollout-global-dataset`, `--partial-rollout`, or a different custom
`--data-source-path`. Set `--num-rollout` explicitly: independently advancing
datasets do not define a single epoch from which to infer the run length.

The initial schema covers dataset loading and mixing. Per-dataset response limits,
generation functions and agent settings are not yet accepted; unknown settings
fail instead of being silently ignored. Rows continue to select existing graders
through `metadata.rm_type`.

### `--eval-function-path`

Same signature as `--rollout-function-path`. Defaults to whatever rollout function is
configured.

---

## Session

### `--session-message-matcher`

Some harnesses do not replay history verbatim — they reserialize tool-call arguments or drop `reasoning_content` — and the default `strict` matcher counts that as divergence (v1 rollback, v2 branching). This flag loosens what "the same message" means during replay: choose a looser built-in selector (see [Agentic Rollout (TITO)](/user-guide/agentic-rollout#choose-replay-matching)) or supply your own matcher via a trusted dotted import path:

```python
def matcher(stored_message: dict[str, Any], replayed_message: dict[str, Any]) -> bool:
    ...
```

`stored_message` is the authoritative stored message; return `True` to accept the replayed message as the same history at that position. Keep the matcher a fast, synchronous, side-effect-free equivalence check — exceptions or non-`bool` results return HTTP 500, and startup fails if the path does not resolve.

---

## Reward

### `--custom-rm-path`

```python
async def custom_rm(args, sample: Sample) -> float:
    ...

# Batched mode (set --group-rm)
async def batched_custom_rm(args, samples: list[Sample]) -> list[float]:
    ...
```

**Built-in `--rm-type` options:** `math`, `dapo`, `deepscaler`, `gemma_math`, `f1`,
`gpqa`, `ifbench`, `remote_rm` (with `--rm-url`), `random`, `deterministic_random`.
Prefixing any of them with `boxed_` (for example `boxed_math`) extracts `\boxed{}`
from the response before grading.

### `--custom-reward-post-process-path`

Hook to normalize rewards differently from the default GRPO normalization.

---

## Filtering

### `--dynamic-sampling-filter-path`

Per-group filter; runs after scoring, before queueing for training.

```python
def filter_function(args, samples: list[Sample], **kwargs) -> FilterOutput:
    return FilterOutput(keep=True, reason=None)
```

**Stock implementation:** `miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`.

### `--buffer-filter-path`

Pops samples from the rollout buffer at dequeue time. The default is
`pop_first` in `miles/rollout/data_source.py`.

```python
def buffer_filter(
    args,
    rollout_id: int | None,
    buffer: list[list[Sample]],
    num_samples: int,
) -> list[list[Sample]]:
    ...
```

### `--rollout-sample-filter-path`

Per-sample, in-place. Set `s.remove_sample = True` to exclude a sample from the loss
(advantage normalization still uses it).

The framework passes `data: list[list[Sample]]` — a list of
`n_samples_per_prompt`-size groups — so iterate the outer list once to reach `Sample`
objects:

```python
def filter_function(args, data: list[list[Sample]]) -> None:
    for group in data:
        for s in group:
            if not_good(s):
                s.remove_sample = True
```

### `--rollout-all-samples-process-path`

Runs after rollout completes and can see all samples, including filtered ones.
Useful for logging or analysis.

### `--rollout-data-postprocess-path`

Runs after log probabilities have been computed but before training. Useful for
updating loss masks based on per-token logprobs.

---

## Training

### `--custom-loss-function-path`

Replace the GRPO/PPO loss. Requires `--loss-type custom_loss`. Useful for novel
objectives or multi-objective work.

### `--custom-tis-function-path`

Importance sampling correction for off-policy training when train and inference
diverge.

**Reference:** [`examples/infra_features/train_infer_mismatch_helper/mis.py`](https://github.com/radixark/miles/blob/main/examples/infra_features/train_infer_mismatch_helper/mis.py).

### `--custom-pg-loss-reducer-function-path`

```python
def get_pg_loss_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    ...
```

Use case: Dr.GRPO divides by a constant instead of effective token count.
**Reference:** [`examples/experimental/DrGRPO/custom_reducer.py`](https://github.com/radixark/miles/blob/main/examples/experimental/DrGRPO/custom_reducer.py).

### `--custom-convert-samples-to-train-data-path`

```python
def convert_samples_to_train_data(args, samples) -> dict:
    return {
        "tokens":           [...],
        "response_lengths": [...],
        "rewards":          [...],
        "raw_reward":       [...],
        "truncated":        [...],
        "sample_indices":   [...],
        "loss_masks":       [...],
        # optional
        "round_number":            [...],
        "rollout_log_probs":       [...],
        "rollout_routed_experts":  [...],
        "metadata":                [...],
        "multimodal_train_inputs": [...],
        "teacher_log_probs":       [...],
    }
```

---

## Megatron hooks

| Flag | Signature |
|---|---|
| `--custom-megatron-init-path` | `def custom_init(args) -> None` |
| `--custom-megatron-before-log-prob-hook-path` | `def custom_hook(args, model, store_prefix) -> None` |
| `--custom-megatron-before-train-step-hook-path` | `def custom_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None` |
| `--custom-megatron-post-save-hook-path` | `def hook(args, rollout_id: int, checkpoint_dir: str, hf_checkpoint_dir: str | None) -> None` |

The Megatron init, log-prob, and train-step hooks give access to the live model
and optimizer, useful for custom probes, weight clipping, or surgical interventions.
The post-save hook runs on rank 0 after checkpoint save completion and receives
the saved checkpoint paths instead of live model objects.

---

## Logging

```python
# Training rollouts
def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    ...

# Eval rollouts
def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    ...
```

Return `True` to suppress Miles's default logging, `False` to layer on top.

---

## Model

### `--custom-model-provider-path`

Replace Megatron's default model factory.

```python
def custom_model_provider(
    pre_process: bool,
    post_process: bool,
    vp_stage: int | None = None,
) -> GPTModel:
    ...
```

---

## Worked example

A custom rollout plus a custom reward in one launch script:

```bash
ROLLOUT_ARGS+=(
   --custom-generate-function-path my_pkg.search_rollout.generate
   --custom-rm-path                my_pkg.rewards.f1_with_grounding
   --metadata-key metadata
   --rollout-max-response-len 4096
)
```

That is the entire delta from the stock GRPO recipe, with no source changes to Miles.

→ Next: [Server arguments reference](/user-guide/cli-reference)
