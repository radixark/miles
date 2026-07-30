---
title: Tinker-compatible API
description: Use the public Tinker SDK with a Miles Megatron trainer and SGLang samplers.
---

Miles can run as a self-hosted, Tinker SDK-compatible training and sampling
data plane. A regular `tinker.ServiceClient` talks to Miles over HTTP; Miles
maps training requests to Megatron multi-LoRA slots and sampling requests to
SGLang.

This integration targets the public SDK's core `ServiceClient`,
`TrainingClient`, and `SamplingClient` workflows. It is not an implementation
of Tinker's hosted control plane.

## Architecture

Each `create_lora_training_client` call reserves one Miles multi-LoRA slot.
Model operations are executed in SDK sequence order on every Megatron rank:

```text
Tinker SDK
    |
    +-- forward / forward_backward / optim_step --> Miles controller
    |                                                |
    |                                                +--> Megatron ranks
    |
    +-- save_weights_for_sampler ------------------> HF PEFT snapshot
                                                     |
                                                     +--> SGLang LoRA loader
                                                            |
                                                            +--> sample
```

Training requests are serialized because they share one Megatron trainer.
Sampling requests run concurrently against immutable snapshots, so a sampler
never observes a partially applied optimizer step.

## Start a service

Install Miles with Megatron-Bridge's `bridge` branch and SGLang's
`sglang-miles` branch. Start a Ray cluster with separate trainer and rollout
GPUs, then launch `train_tinker.py`. A runnable two-GPU Qwen example is in
[`examples/tinker`](https://github.com/radixark/miles/tree/main/examples/tinker).

The important service arguments are:

| Argument | Meaning |
| --- | --- |
| `--tinker-model-name` | Public base-model name accepted from SDK clients. Defaults to `--hf-checkpoint`. |
| `--tinker-tokenizer-id` | Tokenizer metadata returned by `get_info`. Defaults to the public model name. |
| `--tinker-checkpoint-dir` | Local root for training and sampler snapshots. Defaults to `--save/tinker`, or `/tmp/miles/tinker` when `--save` is unset. |
| `--tinker-api-key` | Optional API key checked against the SDK's `X-API-Key` header and redacted from parsed argument logs. Prefer the `TINKER_API_KEY` environment variable so it does not enter process arguments. When neither is set, authentication is disabled. |
| `--tinker-max-concurrent-samples` | Sampling concurrency advertised to SDK clients. |
| `--multi-lora-n-adapters` | Number of training-client adapter slots available concurrently. |
| `--lora-rank` | Maximum rank a client may request. |
| `--target-modules` | Superset of modules clients may enable. Include `lm_head` to support `train_unembed=True`. |

`pipeline-model-parallel-size` and `context-parallel-size` must be 1, and
`qkv-format` must be `thd`. Tensor and data parallelism are supported. Trainer
and rollout GPUs must be disaggregated; colocated mode is rejected.

## Use the official SDK

```python
import tinker
from tinker import types

service = tinker.ServiceClient(
    base_url="http://127.0.0.1:8068",
    api_key="local",  # any non-empty value when server auth is disabled
)
training = service.create_lora_training_client(
    base_model="Qwen/Qwen3-0.6B",
    rank=16,
    seed=7,
    train_attn=True,
    train_mlp=True,
    train_unembed=True,
)

tokens = [9707, 11, 1917]
datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": tokens[1:],
        "weights": [1.0, 1.0],
    },
)

training.forward_backward([datum], "cross_entropy").result()
training.optim_step(types.AdamParams(learning_rate=1e-4)).result()

snapshot = training.save_weights_for_sampler("step-1").result()
sampler = service.create_sampling_client(model_path=snapshot.path)
result = sampler.sample(
    prompt=types.ModelInput.from_ints(tokens),
    num_samples=1,
    sampling_params=types.SamplingParams(max_tokens=32, temperature=0.8),
).result()
```

Both synchronous and asynchronous public SDK methods use the same server
surface.

## Loss contracts

Only `encoded_text` model-input chunks are executable. For every datum,
`target_tokens.shape[0]` must equal the encoded model-input length.
Losses use Tinker's sum reduction; Miles does not divide gradients by datum or
token count.

| Loss | Required `loss_fn_inputs` | Optional config |
| --- | --- | --- |
| `cross_entropy` | `target_tokens`, `weights` | none |
| `importance_sampling` | `target_tokens`, `logprobs`, `advantages` | none |
| `ppo` | `target_tokens`, `logprobs`, `advantages` | `clip_low_threshold`, `clip_high_threshold` |
| `cispo` | `target_tokens`, `logprobs`, `advantages` | `clip_low_threshold`, `clip_high_threshold` |
| `dro` | `target_tokens`, `logprobs`, `advantages` | `beta` |

`target_tokens` is `int64`; all other loss tensors are `float32`.
Cross-entropy also accepts two-dimensional top-k targets and weights with
matching shapes. RL losses require one target per position.

`forward` and `forward_backward` return per-datum target logprobs plus a
`loss:sum` metric. `forward_backward` accumulates gradients until
`optim_step`; the optimizer call applies the SDK's Adam parameters without an
implicit batch-size normalization. The SDK's `forward_backward_custom` is also
supported: its client-side gradient is sent back through weighted
cross-entropy, using the same standard endpoints.

## Checkpoints and samplers

`save_state` stores:

- native per-rank Megatron LoRA shards;
- an HF PEFT adapter snapshot;
- per-slot Adam state; and
- any gradients accumulated before the save.

`load_state_with_optimizer` restores all four, including retained gradients.
`load_state` restores weights only and resets optimizer state and gradients.
`create_training_client_from_state` and
`create_training_client_from_state_with_optimizer` are supported through the
SDK weights-info flow.

`save_weights_for_sampler(name)` creates a named immutable sampler snapshot.
`save_weights_and_get_sampling_client()` creates an ephemeral one. Sampling
supports token prompts, multiple samples, seeds, stop strings or token IDs,
temperature, top-k, top-p, generation logprobs, prompt logprobs, and prompt
top-k logprobs.

## Compatibility boundary

The current service intentionally rejects unsupported requests instead of
silently changing their meaning:

- LoRA training only; no full-parameter training.
- Text token chunks only; no image, audio, or multimodal chunks.
- One configured base model per service process.
- No colocated GPUs, pipeline parallelism, context parallelism, or
  fault-tolerant or independent-DP trainer replicas.
- Packed THD training layout only; BSHD is rejected at startup.
- No hosted-account, project, billing, checkpoint-listing, publishing, or
  access-control APIs.
- Protocol metadata, sessions, futures, and Tinker URI lookup live in the
  controller's memory. Checkpoint files persist, but a service restart does
  not currently rebuild URI metadata from disk.

Use `GET /api/v1/healthz` for readiness. When `TINKER_API_KEY` or
`--tinker-api-key` is set, pass the same value as `api_key` to
`tinker.ServiceClient`. The environment variable takes precedence when both
are present.
