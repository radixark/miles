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
| | `--custom-generate-function-path` | A single sample's generation |
| | `--data-source-path` | How prompts are loaded |
| | `--eval-function-path` | The eval rollout |
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

**Default:** `miles.rollout.sglang_rollout.generate_rollout`, or
`miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn` when
`enable_experimental_rollout_refactor()` is on.

**Reference:** [`examples/multi_agent/rollout_with_multi_agents.py`](https://github.com/radixark/miles/blob/main/examples/multi_agent/rollout_with_multi_agents.py).

### `--custom-generate-function-path`

Replace just the generation step inside the default rollout. Most tool-use, RAG, and
multi-turn workflows live here.

```python
async def custom_generate(args, sample: Sample, sampling_params: dict) -> Sample:
    ...
```

**Reference:** [`examples/search-r1/generate_with_search.py`](https://github.com/radixark/miles/blob/main/examples/search-r1/generate_with_search.py).

### `--data-source-path`

```python
class CustomDataSource(DataSource):
    def get_samples(self, num_samples) -> list[list[Sample]]: ...
    def add_samples(self, samples) -> None: ...
    def save(self, rollout_id) -> None: ...
    def load(self, rollout_id=None) -> None: ...
```

**Default:** `miles.rollout.data_source.RolloutDataSourceWithBuffer`.

### `--eval-function-path`

Same signature as `--rollout-function-path`. Defaults to whatever rollout function is
configured.

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

**Built-in `--rm-type` options:** `math`, `dapo`, `deepscaler`, `f1`, `gpqa`,
`ifbench`, `remote_rm` (with `--rm-url`), `random`.

### `--custom-reward-post-process-path`

Hook to normalize rewards differently from the default GRPO normalization.

---

## Filtering

### `--dynamic-sampling-filter-path`

Per-group filter; runs after scoring, before queueing for training.

```python
def filter_function(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    return DynamicFilterOutput(keep=True, reason=None)
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

**Reference:** [`examples/train_infer_mismatch_helper/mis.py`](https://github.com/radixark/miles/blob/main/examples/train_infer_mismatch_helper/mis.py).

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
**Reference:** [`examples/DrGRPO/custom_reducer.py`](https://github.com/radixark/miles/blob/main/examples/DrGRPO/custom_reducer.py).

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
