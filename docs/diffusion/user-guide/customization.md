---
title: Customization
description: Plug-points where you can drop in your own Python without forking miles-diffusion.
---
Most of miles-diffusion's behavior can be replaced with user-supplied Python by
passing a `--*-path` flag (loaded via `miles.utils.misc.load_function` as a
dotted import path). This page lists the diffusion-relevant hooks, the
signatures they expect, and the defaults they replace — in the same spirit as
[Miles customization](https://miles.radixark.com/docs/user-guide/customization).

## At a glance

| Stage | Flag | Replaces |
|---|---|---|
| **Rollout** | `--rollout-function-path` | The whole rollout loop |
| | `--eval-function-path` | The eval rollout |
| | `--data-source-path` | How prompts / buffer are loaded |
| | `--custom-generate-function-path` | One microgroup's generation |
| | `--diffusion-step-strategy-path` | Which denoising steps enter training |
| **Reward** | `--custom-rm-path` | Batched reward computation |
| | `--custom-reward-post-process-path` | Advantage normalization |
| **Filtering** | `--dynamic-sampling-filter-path` | Per-group keep / drop |
| | `--buffer-filter-path` | Buffer dequeue selection |
| | `--rollout-sample-filter-path` | Per-sample loss mask |
| **Training** | `--custom-expand-samples-to-train-pairs-path` | Trajectory → train pairs |
| | `--custom-convert-samples-to-train-data-path` | Full sample → train_data |
| | `--custom-prepare-train-batch-path` | DiT batch preparation |
| | `--custom-loss-function-path` | Loss formula only |
| | `--sde-step-backend-path` | Train-side SDE dynamics |
| **Logging** | `--custom-rollout-log-function-path` | Train-rollout logging |
| | `--custom-eval-rollout-log-function-path` | Eval-rollout logging |
| **Model family** | `--train-pipeline-config-path` | Family `TrainPipelineConfig` |
| | `--model-backend-path` | Model load / FSDP backend |

`--loss-type` (`policy_loss` / `nft` / `sft_loss` / `custom_loss`) is the
shortcut that auto-fills several of the train-side paths; see § Training.

***

## Rollout

### `--rollout-function-path`

Replace the entire train rollout. Diffusion recipes use
`miles.rollout.sglang_diffusion_rollout.generate_rollout` (not the LLM default).

```python
def generate_rollout(
    args, rollout_id, data_source, evaluation=False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    ...
```

`miles/ray/rollout.py` passes the data source positionally through `call_rollout_fn`.
Legacy list/dict returns are still wrapped into the corresponding output dataclass by
`miles.rollout.base_types.call_rollout_fn`.

### `--eval-function-path`

Same shape as the rollout function. Defaults to `--rollout-function-path` when
unset.

### `--data-source-path`

**Class** (not a function). Default:
`miles.rollout.data_source.RolloutDataSourceWithBuffer`.

```python
class CustomDataSource:
    def __init__(self, args): ...
    def get_samples(self, num_samples) -> list[list[Sample]]: ...
    def add_samples(self, samples) -> None: ...
    def save(self, rollout_id) -> None: ...
    def load(self, rollout_id=None) -> None: ...
```

### `--custom-generate-function-path`

Replace the inner microgroup generator inside
`generate_and_rm_microgroup`. Default: `generate_microgroup`.

```python
async def custom_generate(
    args, microgroup: list[Sample], sampling_params: dict, *, evaluation: bool = False
) -> list[Sample]:
    ...
```

`evaluation=` is optional — if your signature omits it, the caller skips that kwarg.

### `--diffusion-step-strategy-path`

Select which denoising steps contribute SDE log-probs / train pairs. Stock
implementations live in `miles/rollout/step_strategy_hub.py`.

```python
def strategy(args, sample, num_steps, seed) -> tuple[list[int] | None, list[int] | None]:
    # returns (sde_step_indices, return_step_indices); return must be None today
    ...
```

| Stock | Behavior |
|---|---|
| `sde_window` | Random contiguous window (`--diffusion-num-sde-steps`, `--diffusion-sde-window-range`) |
| `epoch_global_random_choice` | Per-epoch subset of `--diffusion-sde-candidate-steps` |

Details: [SDE step backend](/diffusion/advanced/sde-backend).

***

## Reward

Built-in scorers (`--rm-type pickscore` / `ocr`) are documented in
[Rewards](/diffusion/user-guide/rewards). The hooks below replace that dispatch entirely.

### `--custom-rm-path`

```python
async def custom_rm(args, samples: list[Sample], **kwargs) -> list[float]:
    ...
```

Wired only through `batched_async_rm` — implement per-sample routing inside your
batched function if needed.

```bash
--custom-rm-path my_project.rewards.aesthetic_rm
```

HTTP / remote scoring: implement a batched custom RM and read `args.rm_url` (or
your own flags). Encode images from `sample.generated_output` (see
`_sample_to_rgb_hwc_uint8_frames` in `miles/rollout/rm_hub/pickscore.py`):

```python
import aiohttp
from miles.utils.types import Sample

async def api_rm(args, samples: list[Sample], **kwargs) -> list[float]:
    async with aiohttp.ClientSession() as session:
        rewards = []
        for sample in samples:
            payload = {"prompt": sample.prompt, "image_b64": "<your encoding>"}
            async with session.post(args.rm_url, json=payload) as resp:
                rewards.append((await resp.json())["score"])
        return rewards
```

```bash
--custom-rm-path my_project.rewards.api_rm \
--rm-url http://localhost:8000/score
```

### `--custom-reward-post-process-path`

Replace GRPO advantage normalization in
`RolloutManager._post_process_rewards`.

```python
def post_process(args, samples) -> tuple[list[float], list[float]]:
    # Returns (raw_rewards, normalized_rewards)
    ...
```

Default behavior: reshape to `(-1, n_samples_per_prompt)`, subtract mean
(`--globalize-reward-mean` for batch-level), optionally divide by std
(on by default — `--disable-grpo-std-normalization` turns it off; `--globalize-reward-std`
switches per-group std to batch-wide).

***

## Filtering

### `--dynamic-sampling-filter-path`

Per-group filter after scoring (DAPO-style). Stock:
`miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`.

```python
def filter_function(args, samples: list[Sample], **kwargs):
    # return DynamicFilterOutput(keep=..., reason=...) or a bool
    ...
```

### `--buffer-filter-path`

Select samples when dequeuing from the rollout buffer. Default is `pop_first`
in `miles/rollout/data_source.py`.

```python
def buffer_filter(
    args, rollout_id: int | None, buffer: list[list[Sample]], num_samples: int
) -> list[list[Sample]]:
    ...
```

### `--rollout-sample-filter-path`

Per-sample, in-place. Set `sample.remove_sample = True` to exclude a sample from
the loss (it still participates in advantage normalization).

```python
def filter_function(args, data: list[list[Sample]]) -> None:
    for group in data:
        for s in group:
            if not_good(s):
                s.remove_sample = True
```

***

## Training

`--loss-type` picks the default prepare / formula / expand paths:

| `--loss-type` | Expand pairs | Prepare | Loss formula |
|---|---|---|---|
| `policy_loss` (default) | Flow-GRPO built-in | `prepare_flow_grpo_batch` | `flow_grpo_loss_formula` |
| `nft` | `data_conversion_hub.nft.expand_samples_to_train_pairs` | `loss_hub.nft.prepare_nft_batch` | `nft_loss_formula` |
| `sft_loss` | (via SFT convert) | required custom / SFT defaults | required custom / SFT defaults |
| `custom_loss` | your expand (or convert) | your prepare | your formula |

DiT forward always stays in the FSDP actor; the loss hook only computes the
objective.

### `--custom-expand-samples-to-train-pairs-path`

```python
def expand_samples_to_train_pairs(args, samples, rewards, raw_rewards) -> dict:
    ...
```

Default for Flow-GRPO lives under `miles/ray/data_conversion_hub/flow_grpo.py`.

### `--custom-convert-samples-to-train-data-path`

Replace the entire `RolloutManager._convert_samples_to_train_data` (including
reward post-process + expand). Prefer the narrower expand hook unless you need
full control.

```python
def convert_samples_to_train_data(args, samples) -> dict:
    ...
```

### `--custom-prepare-train-batch-path`

```python
def prepare(ctx, batch, *, pad_to_len=None) -> PreparedBatch:
    ...
```

Builds DiT inputs from train pairs. Defaults:
`miles.backends.fsdp_utils.loss_hub.flow_grpo.prepare_flow_grpo_batch` or the NFT
equivalent under `loss_hub.nft`.

### `--custom-loss-function-path`

```python
def loss_formula(ctx, batch, prepared, *, new_pred, ref_pred, metrics, **kwargs):
    ...
```

### `--sde-step-backend-path`

**Class** implementing `SdeStepBackend`. Auto-selected from
`--diffusion-sde-type` (`sde` / `ode` → `DiffusersSdeStepBackend`, `cps` →
`CpsSdeStepBackend`) unless overridden. See
[SDE step backend](/diffusion/advanced/sde-backend).

***

## Logging

```python
def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    ...

def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    ...
```

Return a truthy value to skip the default logging; falsy layers on top.

***

## Model family

### `--hf-checkpoint` / `--diffusion-model-family` / `--train-pipeline-config-path`

`--hf-checkpoint` names the diffusers pipeline for train + rollout. Family is
resolved from the checkpoint name unless you pass `--diffusion-model-family`
(e.g. `sd3`). For an unregistered family, pass `--train-pipeline-config-path`
to a `TrainPipelineConfig` subclass instead.

### `--model-backend-path`

**Class** for loading components / FSDP / scheduler. Default comes from the
family config (usually a `DiffusersModelBackend`).

***

## Worked example

Custom reward + post-process on top of the stock SD3 Flow-GRPO recipe:

```bash
python3 scripts/run_diffusion_grpo_sd3_ocr_sglang.py \
  --cuda-visible-devices 6,7 \
  --extra-args "
    --custom-rm-path my_project.rewards.api_rm
    --rm-url http://localhost:8000/score
    --custom-reward-post-process-path my_project.rewards.normalize
  "
```

(`--extra-args` forwarding depends on the launch script; you can also splice the
flags into a forked recipe.)

***

## Pairs well with

- [Rewards](/diffusion/user-guide/rewards) — built-in PickScore / OCR and prompt JSONL.
- [SDE step backend](/diffusion/advanced/sde-backend) — step strategies and SDE kernels.
- [LoRA weight sync](/diffusion/advanced/lora) — IPC merge path used by most recipes.
- [SD3 model guide](/diffusion/models/sd3/sd3) — end-to-end recipe flags.
