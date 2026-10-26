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

Replace the entire train rollout. Diffusion recipes pass
`miles.rollout.sglang_diffusion_rollout.generate_rollout`; the flag's default
(`miles.rollout.sglang_rollout.generate_rollout`, the LLM path) is not shipped in miles-diffusion.

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

`RolloutManager` also reads `len(data_source.dataset)` when `--num-rollout` is unset.

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

Select which denoising steps contribute SDE log-probs / train pairs, and which
latents the engine ships back for the trainer. Stock implementations live in
`miles/rollout/step_strategy_hub.py`.

```python
def strategy(args, sample, num_steps, seed) -> tuple[list[int] | None, list[int] | None]:
    # returns (sde_step_indices, return_step_indices)
    # sde None = no per-step filter: every step uses --diffusion-sde-type
    # return None = the engine returns every step's latent
    ...
```

`--rollout-return-full-trajectory` overrides `return_step_indices` back to all
steps for debugging / A-B runs.

| Stock | Behavior |
|---|---|
| `sde_window` | Random contiguous window (`--diffusion-num-sde-steps`, `--diffusion-sde-window-range`); returns all latents its steps touch |
| `epoch_global_random_choice` | Subset of `--diffusion-sde-candidate-steps`, shared by each block of `--rollout-batch-size` consecutive prompt groups (one rollout, unless a dynamic-sampling filter drops groups); returns all latents its steps touch |
| `ode_and_return_last` | No step filter (every step uses `--diffusion-sde-type`, which the DiffusionNFT recipes set to `ode`); return only the final clean `x0` |

DiffusionNFT picks the timesteps it trains on with a separate hook,
`--diffusion-nft-timestep-strategy-path`: `(args, num_steps, generator) -> list[int]`, built in
`drop_random_steps` (default) and `drop_final_steps` in `miles/ray/data_conversion_hub/nft.py`.

***

## Reward

Built-in scorers (`--rm-type pickscore` / `hps` / `ocr` / `custom_api` / `openai_api`) are documented in
[Rewards](/diffusion/user-guide/rewards). `--custom-rm-path` replaces that dispatch entirely; `--custom-reward-post-process-path`
replaces the normalization that runs after scoring.

### `--custom-rm-path`

```python
async def custom_rm(args, samples: list[Sample], **kwargs) -> list[float]:
    ...
```

Wired only through `batched_async_rm` — implement per-sample routing inside your
batched function if needed. Once set, `--rm-type` and `metadata.rm_type` are ignored: the
custom function is the whole dispatch.

```bash
--custom-rm-path my_project.rewards.aesthetic_rm
```

`--custom-rm-args` is an opaque string passed through as `args.custom_rm_args` for the
custom RM to parse (requires `--custom-rm-path`).

Shipped custom RMs:

| Path | What |
|---|---|
| `miles.rollout.rm_hub.weighted_mixture_rm.weighted_mixture_rm` | Weighted sum of built-in rewards (`hps`, `pickscore`, `ocr`, `custom_api`, `openai_api`), weights from `--custom-rm-args "hps=0.7,pickscore=0.3"`; returns a dict per sample, train on it with `--reward-key weighted`. See [Rewards](/diffusion/user-guide/rewards) § Combining rewards. |

HTTP / remote scoring: `--rm-type openai_api` / `custom_api` (configured by
`--openai-api-rm-config` / `--custom-api-rm-config`) call an external service; otherwise
implement a batched custom RM and read `args.rm_url` (or your own flags). Encode images
from `sample.generated_output` (see `generated_output_to_rgb_hwc_uint8_frames` in
`miles/utils/processing_utils.py`):

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
in `miles/rollout/data_source.py`. The stock diffusion rollout never calls
`add_samples`, so this only fires when a custom rollout refills the buffer.

```python
def buffer_filter(
    args, rollout_id: int | None, buffer: list[list[Sample]], num_samples: int
) -> list[list[Sample]]:
    ...
```

***

## Training

`--loss-type` picks the default prepare / formula / expand paths:

| `--loss-type` | Expand pairs | Prepare | Loss formula |
|---|---|---|---|
| `policy_loss` (default) | Flow-GRPO built-in | `prepare_flow_grpo_batch` | `flow_grpo_loss_formula` |
| `nft` | `data_conversion_hub.nft.expand_samples_to_train_pairs` | `loss_hub.nft.prepare_nft_batch` | `nft_loss_formula` |
| `sft_loss` | none; convert `sft_rollout.convert_samples_to_train_data` | `loss_hub.sft.prepare_sft_batch` | `sft_loss_formula` |
| `custom_loss` | your expand (or convert) | your prepare | your formula |

`sft_loss` auto-fills nothing: it requires the convert, rollout-log, prepare and loss
paths, plus `--train-only` and a rollout function that runs no engines (e.g.
`miles.rollout.sft_rollout.generate_rollout`; see `scripts/run_diffusion_sft_wan22.py`). `custom_loss` behaves like `policy_loss`:
any hook left unset falls back to the Flow-GRPO built-in.

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
equivalent under `loss_hub.nft`. `PreparedBatch.latents`, `timesteps` and
`timesteps_for_model` are keyed by latent stream name: `"visual"`, plus e.g.
`"audio"` for joint audio-video models.

### `--custom-loss-function-path`

```python
def loss_formula(ctx, batch, prepared, *, new_pred, old_pred, ref_pred, metrics, **kwargs):
    ...
```

`new_pred`, `old_pred` and `ref_pred` are keyed by the same stream names as `prepared.latents`.
`old_pred` is `None` unless `--loss-type nft`; `ref_pred` is `None` with `--ref-mode none`.

### `--sde-step-backend-path`

**Class** implementing `SdeStepBackend`. Auto-selected from
`--diffusion-sde-type` (`sde` / `ode` → `FlowSdeStepBackend`, `cps` →
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

### `--hf-checkpoint` / `--diffusion-model-family` / `--diffusion-task` / `--train-pipeline-config-path`

`--hf-checkpoint` names the diffusers pipeline for train + rollout. Family is
resolved from the checkpoint name unless you pass `--diffusion-model-family`
(e.g. `sd3`). The longest matching name pattern wins, so `Qwen/Qwen-Image-2.1`
resolves to `qwen_image21`. `--diffusion-task` (e.g. `t2i`) must be one of the
`tasks` that family's `TrainPipelineConfig` declares. For an unregistered family, pass
`--train-pipeline-config-path` to a `TrainPipelineConfig` subclass instead.

### `--model-backend-path`

**Class** for loading components / FSDP. Default comes from the
family config (usually a `DiffusersModelBackend`).

***

## Worked example

Custom reward + post-process on top of the stock SD3 Flow-GRPO recipe:

```bash
python3 scripts/run_diffusion_grpo_sd3_ocr_sglang.py \
  --cuda-visible-devices 6,7 \
  --extra-args "--custom-rm-path my_project.rewards.api_rm --rm-url http://localhost:8000/score --custom-reward-post-process-path my_project.rewards.normalize"
```

(`--extra-args` is spliced verbatim into a `bash -c` command line, so keep it on one line; you can also splice the
flags into a forked recipe.)

***

## Pairs well with

- [Rewards](/diffusion/user-guide/rewards) — built-in PickScore / HPS / OCR and prompt JSONL.
- [SDE step backend](/diffusion/advanced/sde-backend) — step strategies and SDE kernels.
- [LoRA weight sync](/diffusion/advanced/lora) — IPC merge path used by most recipes.
- [SD3 model guide](/diffusion/models/sd3/sd3) — end-to-end recipe flags.
