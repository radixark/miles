---
title: Customization
description: The 20+ plug-points where you can drop in your own Python without forking Miles.
---

# Customization

Miles is small at the core and big at the edges. The trainer is one Python file
(`train.py`) and an algorithm; everything else can be replaced by passing
`--<thing>-path my_module.my_function`.

This page is a complete catalogue of those plug-points.

## At a glance

| Stage | Flag | Replaces |
|---|---|---|
| **Rollout** | `--rollout-function-path` | The whole rollout loop |
| | `--custom-generate-function-path` | A single sample's generation |
| | `--data-source-path` | How prompts are loaded |
| | `--eval-function-path` | The eval rollout |
| **Reward** | `--custom-rm-path` | Reward computation |
| | `--custom-reward-post-process-path` | Reward normalisation |
| **Filtering** | `--dynamic-sampling-filter-path` | Per-group filter (DAPO) |
| | `--buffer-filter-path` | Buffer dequeue filter |
| | `--rollout-sample-filter-path` | Per-sample loss filter |
| | `--rollout-all-samples-process-path` | Inspect all samples post-rollout |
| | `--rollout-data-postprocess-path` | Mutate samples post-logprob |
| **Training** | `--custom-loss-function-path` | The loss formula |
| | `--custom-tis-function-path` | Importance sampling correction |
| | `--custom-pg-loss-reducer-function-path` | Loss reduction (Dr.GRPO) |
| | `--custom-convert-samples-to-train-data-path` | Sample → tensor batch |
| **Megatron hooks** | `--custom-megatron-init-path` | After Megatron init |
| | `--custom-megatron-before-log-prob-hook-path` | Before logprob compute |
| | `--custom-megatron-before-train-step-hook-path` | Before each train step |
| **Logging** | `--custom-rollout-log-function-path` | Train-rollout logging |
| | `--custom-eval-rollout-log-function-path` | Eval-rollout logging |
| **Routing** | `--miles-router-middleware-paths` | Router middleware |
| **Model** | `--custom-model-provider-path` | Megatron model factory |

The rest of this page documents each one.

---

## Rollout

### `--rollout-function-path` — the whole loop

Replace the entire rollout function. Use this only for **fundamentally different** flows
(e.g. multi-agent co-evolution).

```python
async def generate_rollout(args, rollout_id, *, evaluation=False) \
        -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    ...
```

**Default:** `miles.rollout.sglang_rollout.generate_rollout` (or the experimental
`InferenceRolloutFn` if `enable_experimental_rollout_refactor()`).

**Reference:** [`examples/multi_agent/rollout_with_multi_agents.py`](https://github.com/radixark/miles/blob/main/examples/multi_agent/rollout_with_multi_agents.py).

### `--custom-generate-function-path` — per-sample generation

Replace just the generation step inside the default rollout. Most tool-use / RAG /
multi-turn workflows live here.

```python
async def custom_generate(args, sample: Sample, sampling_params: dict) -> Sample:
    ...
```

**Reference:** [`examples/search-r1/generate_with_search.py`](https://github.com/radixark/miles/blob/main/examples/search-r1/generate_with_search.py).

### `--data-source-path` — where prompts come from

```python
class CustomDataSource(DataSource):
    def get_samples(self, num_samples) -> list[list[Sample]]: ...
    def add_samples(self, samples) -> None: ...
    def save(self, rollout_id) -> None: ...
    def load(self, rollout_id=None) -> None: ...
```

**Default:** `miles.rollout.data_source.RolloutDataSourceWithBuffer`.

### `--eval-function-path` — eval rollout

Same signature as `--rollout-function-path`. Defaults to whatever rollout function you've
configured.

---

## Reward

### `--custom-rm-path`

```python
# Single-sample mode
async def custom_rm(args, sample: Sample) -> float:
    ...

# Batched mode (set --group-rm)
async def batched_custom_rm(args, samples: list[Sample]) -> list[float]:
    ...
```

**Built-in `--rm-type` options:** `math`, `dapo`, `deepscaler`, `f1`, `gpqa`, `ifbench`,
`remote_rm` (with `--rm-url`).

### `--custom-reward-post-process-path`

Hook to normalise rewards differently from the default GRPO normalisation.

---

## Filtering

### `--dynamic-sampling-filter-path` — per-group

```python
def filter_function(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    return DynamicFilterOutput(keep=True, reason=None)
```

**Stock implementation:** `miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`.

### `--buffer-filter-path` — at dequeue time

```python
def buffer_filter(samples: list[list[Sample]]) -> list[list[Sample]]:
    ...
```

### `--rollout-sample-filter-path` — per-sample, in-place

```python
def filter_function(args, samples: list[Sample]) -> None:
    for s in samples:
        if not_good(s):
            s.remove_sample = True
```

### `--rollout-all-samples-process-path`

Run after rollout completes — for logging or analysis of *all* samples (kept and dropped).

### `--rollout-data-postprocess-path`

Run after log probabilities have been computed but before training. Useful for updating
loss masks based on per-token logprobs.

---

## Training

### `--custom-loss-function-path` — replace GRPO/PPO loss

Requires `--loss-type custom_loss`. Useful for novel objectives or multi-objective work.

### `--custom-tis-function-path` — importance sampling

For off-policy correction when train ≠ inference. **Reference:**
[`examples/train_infer_mismatch_helper/mis.py`](https://github.com/radixark/miles/blob/main/examples/train_infer_mismatch_helper/mis.py).

### `--custom-pg-loss-reducer-function-path` — loss reduction

```python
def get_pg_loss_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    ...
```

**Use case:** Dr.GRPO divides by a constant instead of effective token count. **Reference:**
[`examples/DrGRPO/custom_reducer.py`](https://github.com/radixark/miles/blob/main/examples/DrGRPO/custom_reducer.py).

### `--custom-convert-samples-to-train-data-path` — sample → tensor batch

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

These give you per-step access to the live Megatron model and optimiser — handy for
custom probes, weight clipping, or surgical interventions.

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

## Router

### `--miles-router-middleware-paths`

Inject middleware into the [Miles Router](../advanced/miles-router.md) for request /
response transformation, caching, or custom routing.

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

Plugging in a custom rollout + custom reward in one launch script:

```bash
ROLLOUT_ARGS+=(
   --custom-generate-function-path my_pkg.search_rollout.generate
   --custom-rm-path                my_pkg.rewards.f1_with_grounding
   --metadata-key metadata
   --rollout-max-response-len 4096
)
```

That's the entire delta from the stock GRPO recipe. No code changes to Miles itself.

→ Next: [Server arguments reference](cli-reference.md)
