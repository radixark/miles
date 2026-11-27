---
title: Prompt Enhancement with a Frozen Diffusion Model
description: Train an LLM to rewrite image prompts, rendering each rewrite with a frozen SGLang Diffusion engine in the run and scoring the image on a reward pool.
---
A prompt enhancer is an LLM policy that rewrites a user's short image prompt.
Miles trains one with GRPO: a frozen SGLang Diffusion engine, launched as one
more model in the run, renders each rewrite; a scorer rates the image against
the user's original prompt; and the rating is the reward. The generator's
weights never change.

This page covers the four parts that make it work, how fast they are, and their
limits. The runnable recipe is
[`examples/experimental/prompt_enhancer`](https://github.com/radixark/miles/tree/main/examples/experimental/prompt_enhancer),
whose README holds its settings and commands.

## A frozen diffusion model in `--sglang-config`

A model whose server groups set `engine_kind: diffusion` is served by
`sglang.multimodal_gen` instead of a text engine. The policy comes first, since
the first model's router is the run's default:

```yaml
sglang:
  - name: policy
    server_groups:
      - {worker_type: regular, num_gpus: 1}
  - name: generator
    model_path: /root/models/Z-Image-Turbo
    update_weights: false
    server_groups:
      - {worker_type: regular, engine_kind: diffusion, num_gpus: 5}
```

Each GPU of the group runs one engine. A group's `overrides` reach
`sglang.multimodal_gen` as its own server options (for example
`batching_max_size`); Miles sets the engine's GPU and ports itself.

Argument parsing rejects a diffusion model that breaks any of these rules:

- `update_weights: false`; weight updates skip the generator.
- It is not the first model, and not named `eval`.
- Every server group sets `engine_kind: diffusion` and `worker_type: regular`.
- One GPU per engine, and no `--colocate`.
- `--router miles`, the router with an image route.
- `overrides` do not set `num_gpus`, `master_port` or `scheduler_port`.

The engine has no pause, abort or cache-flush routes, so the rollout server
leaves a diffusion model's engines out of pause, continue, flush and abort, and
images in flight during a policy weight update finish. The policy's request cap
counts text engines only.

## Asking for an image

A reward function asks the generator for one PNG through the model's router:

```python
from miles.rollout.generate_utils.image_generation import ImageRequest, image_generation_client

client = image_generation_client(args, "generator", timeout_secs=600)
png = await client.generate_image(ImageRequest(prompt=rewrite, seed=seed, width=512, height=512))
```

The client posts to SGLang's `/v1/images/generations` and checks that the answer
is a PNG of the requested size. A 4xx answer, an unknown model name or a wrong
size raises `NonRetryableError`, which stops the run. A 5xx answer, a broken
connection or a timeout is retried (`max_retries`, default 3) and then raises
`ImageGenerationUnavailable`, which a reward function turns into a `None`
reward, so the missing-reward filter drops the sample's group.
`max_concurrency` caps the client's requests in flight; unset, nothing caps
them.

With engine batching off, the default, the same request gives the same bytes:
the seed fixes the image (batching changes images; see Throughput).
Z-Image-Turbo rounds sizes to a multiple of 16, so keep width and height
multiples of 16.

## Reward pools

A scorer that needs a GPU runs on a reward pool: Ray workers on GPUs of the
run's placement group, after the rollout and eval GPUs. Pools are a YAML list
in `--reward-pool-config`:

```yaml
- name: hpsv3
  scorer_path: miles.rollout.rm_hub.image.hpsv3.HPSv3Scorer
  scorer_kwargs: {checkpoint_path: /root/models/HPSv3/HPSv3.safetensors, base_model_path: /root/models/Qwen2-VL-7B-Instruct}
  num_workers: 1
  uses_gpu: true
  batch_size: 16
  max_wait_ms: 50
```

A worker builds `scorer_path` as `Scorer(device=..., **scorer_kwargs)` and calls
its `score_batch(images, prompts)`, which returns one `float | None` per image.
The reward function receives each pool as a `<name>_pool` keyword argument, in
training and in eval, and calls `await pool.score(images, prompts)`. The pool
sends at most `batch_size` images per call, round robin across workers, and
restarts a worker it finds dead and resends its batch. With `max_wait_ms` above
0, single-image calls arriving together share a batch, sent when full or
`max_wait_ms` after its first image.

Pools need a run in one Ray deployment with GPUs of its own (no `--colocate`)
and class-based rollout and eval functions, which pass the pools on.

`miles.rollout.rm_hub.image.hpsv3.HPSv3Scorer` is HPSv3 (a Qwen2-VL-7B image
preference model). In bf16 it gives bit-identical scores to MizzenAI's
reference implementation on 32 fixed images.

## The recipe

[`run_qwen25_3b_prompt_enhancer.py`](https://github.com/radixark/miles/blob/main/examples/experimental/prompt_enhancer/run_qwen25_3b_prompt_enhancer.py)
uses RePrompt's RL-stage settings: Qwen2.5-3B-Instruct rewrites 1,108
prompts, 32 prompts × 4 rewrites per step, for 3 epochs; Z-Image-Turbo renders
at 512×512 in 9 steps without CFG. The reward is RePrompt's: the image score times
a weight, plus a format term and a length term. Two scorers:

- **HPSv3** (default), on a reward pool in the job, weight 0.08.
- **ImageReward**, RePrompt's own reward, in `scorer_server.py` processes
  outside the job (it needs transformers 4.x), weight 0.25. Passing
  `--scorer-urls` selects it.

With ImageReward, per-sample image seeds (`--seed-per-sample`) and the engine
rendering, the recipe reproduces RePrompt's run in Miles: over two training
seeds its reward curves overlay those of the same run rendered by diffusers,
and its pooled GenEval gain over the untrained policy is +0.015 (McNemar
p = 0.001), against +0.013 for the diffusers-rendered Miles run and +0.029 for
RePrompt's own training code run the same way with Z-Image-Turbo.

The reward function works per sample, since Miles' eval rejects `--group-rm`,
and needs `--rollout-skip-special-tokens`. Its log hook
(`--custom-rollout-log-function-path`) logs per step:

| Metric | Meaning |
|---|---|
| `rollout/pe/render_secs_{p50,p95,max}` | A sample's wait for its image, retries included |
| `rollout/pe/score_secs_{p50,p95,max}` | A sample's wait for its score, coalescing included |
| `rollout/pe/policy_secs` | Rollout start to the latest render start, a proxy for the last scored completion |
| `rollout/pe/render_tail_secs` | Latest render start to the last image |
| `rollout/pe/score_tail_secs` | Last image to the last score |
| `rollout/pe/after_score_secs` | Last score to the rollout's end |
| `rollout/pe/render_window_secs` | First render start to the last image |
| `rollout/pe/images_per_sec` | Images rendered over the render window |

All of them count the step's scored samples only: an image whose score failed,
or one in a dropped group, is not among them. The four phases add up to the
rollout's wall time, `perf/rollout_time`; `after_score_secs` holds what the
rollout does after the last score, such as filtering. When a dropped group is
replaced, its replacement's completion comes late, so `policy_secs` then also
holds the earlier samples' renders and scores. The rollout's start is the log
hook's own time minus `perf/rollout_time`, and a render start is stamped once
the reward has a completion and has built its image request. The phases, render
window and images per second are logged only when every scored sample's render
start is at or after the rollout's start, so samples rendered before it, such
as replayed or injected ones, log no split; the check is a lower bound only
(see Known limits). The trainer logs `perf/train_time` and `perf/step_time`.

## Throughput

Measured on H200 with the recipe's requests (Z-Image-Turbo, 512×512, 9 steps,
no CFG, a trained policy's rewrites as prompts), one engine per GPU started as
Miles starts it, driven by the image client.

Images per second on one engine, by the client's requests in flight and the
engine's `batching_max_size`:

| Batch size | 1 in flight | 32 in flight | 128 in flight |
|---|---|---|---|
| 1 (default) | 3.75 | 4.35 | 4.34 |
| 4 | 3.75 | 4.36 | 4.35 |
| 8 | 3.77 | 4.44 | 4.48 |
| 16 | 3.74 | 4.40 | 4.45 |

Through the Miles router, 1, 2 and 4 engines render 3.87, 7.75 and 15.41
images per second: engines scale linearly, at 3.86 per GPU. The router costs
11% against a direct connection, since it re-sends each answer's JSON (about
1 MB of base64 PNG).

HPSv3 scores 17.6 images per second on one H200 one at a time, and 22.6 in
batches of 4 or more.

A recipe step on H200 (32 prompts × 4 rewrites, 1 trainer GPU, 1 policy GPU,
HPSv3 on one GPU or, in the second row, ImageReward outside the job), mean of
steps 1–7, in seconds:

| Generator engines | Policy | Render tail | Score tail | Train | Step | Images per second |
|---|---|---|---|---|---|---|
| 1 | 1.3 | 32.6 | 0.10 | 3.0 | 37.8 | 3.85 |
| 2 (ImageReward scoring) | 1.2 | 16.1 | 0.03 | 2.7 | 20.9 | 7.68 |
| 5 (the launcher's default, estimated) | 1.3 | ≈6.6 | ≈0.1–0.7 | 3.0 | ≈12 | ≈19 |

Rendering is the step: the policy's 128 completions take about a second, and
the images queue behind them. The five-engine row is not measured; it scales
the render tail by the engines' linear throughput, and its score tail allows
for HPSv3 scoring one image at a time slightly slower than five engines render.

The levers, in the order to pull them:

1. **More generator engines.** They scale linearly, so the generator's share
   of a step shrinks with each one.
2. **Engine batching stays off.** It gains at most 3% here, since one 512²
   image already fills an H200, and it changes the images: of 128 requests
   rendered alone and in batches of 8, one came out identical, and the worst
   kept its subject but changed its scene. A lone request on a batching
   engine, or one queued behind others, gives the same bytes as alone. A run
   that wants batching anyway sets `batching_max_size` in the generator
   group's `overrides`, and its images then depend on what they shared a batch
   with.
3. **No client concurrency cap.** Concurrency 32 already reaches the engine's
   rate and 128 loses nothing, while one request at a time is 14% slower. A
   step's 128 requests on one engine wait at most 30 s, far inside the
   recipe's 600 s client timeout, so `max_concurrency` stays unset.

The recipe's pool coalesces for 50 ms (`max_wait_ms: 50`; the pool's own
default is 0). At one engine it makes no difference (a step's score tail is one
call, 0.1 s). Five engines deliver about 19 images per second, a little more
than HPSv3's 17.6 one at a time; evenly spaced, those images arrive 52 ms apart,
so 50 ms would batch little and the scorer would trail the last image by about
0.7 s (128 images at 17.6 per second against a 6.6 s render window). Renders
that finish in bursts batch more and shorten that; how much is not measured.
A scorer that falls behind shows as a growing `rollout/pe/score_tail_secs`.

## Known limits

- One failed image drops its whole group: the missing-reward filter drops any
  group with a `None` reward.
- In-run eval shares the training generator's engines.
- A diffusion cell's restart under fault tolerance is untested.
- Out of scope: multi-GPU diffusion engines, training the diffusion model,
  video and multi-turn image environments, and batching that keeps rollout
  trajectories.
- A 200 answer that breaks SGLang's format in one of two ways is not stopped as
  it should be: non-ASCII text in `b64_json` raises a bare `ValueError` (the
  group is dropped instead of the run stopping), and a 24-byte answer ending
  right after the PNG header's width and height passes the size check.
- `geneval.py` binds an output directory to its policy, prompts, seeds and
  image count, but not to the engine and weights that rendered it, so resuming
  against another generator mixes images.
- The step split checks only that every scored sample's render start is at or
  after the rollout's start: a sample rendered outside the rollout after that
  point is counted, and a score stamped after the rollout's end would give a
  negative `after_score_secs`. Nothing in the recipe produces either.
- The launcher accepts a negative `--scorer-gpus` with `--scorer imagereward`.
- The reward-pool deployment checks live in the pool's config module rather
  than with the other argument checks in `miles/utils/arguments.py`.
