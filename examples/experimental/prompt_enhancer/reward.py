"""Reward for prompt enhancement: render the policy's rewrite with the run's frozen diffusion model and score the image
against the user's original prompt, as `--custom-rm-path examples.experimental.prompt_enhancer.reward.prompt_enhancer_rm`.

The terms are RePrompt's (https://github.com/microsoft/DKI_LLM/tree/main/RePrompt, `train_reprompt.py`), on the
completion with the assistant prefix the prompt ends in put back in front of it, summed with weight 1 each:

- image: the score of the image rendered from the first single-line `<answer>...</answer>` (or "a photo" when there
  is none) against the original prompt, times `--pe-image-weight`: RePrompt's `0.5 * IR / 2.0` for ImageReward.
- format: 1 when the completion is `<think>` reasoning on one line, a newline and one `<answer>`, else -1.
- length: 1 when the answer has 16 to 76 words, -1 outside that, 0 with no answer.

The image comes from the model named by `--pe-generator-model` in `--sglang-config`, through its router. The score comes
from the reward pool named by `--pe-scorer-pool` (`--reward-pool-config`), or else from ImageReward in `scorer_server.py`
processes at `--pe-scorer-urls`; both receive the PNG bytes. The reward is per sample, since Miles' eval rejects
`--group-rm`; it is `None` (the group is dropped) when the engine fails, the pool cannot score the image, or every
scorer server attempt fails.
"""

import asyncio
import base64
import hashlib
import itertools
import json
import logging
import os
import re
import time

import httpx
import numpy as np

from miles.rollout.generate_utils.image_generation import (
    ImageGenerationClient,
    ImageGenerationUnavailable,
    ImageRequest,
    image_generation_client,
)
from miles.utils.args.schema import A, Arg, BaseConfig
from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.retry_utils import NonRetryableError
from miles.utils.tracking_utils import tracking
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

ASSISTANT_PREFIX = "Let me solve this step by step.\n<think>"
ORI_PROMPT_KEY = "ori_prompt"
REWARD_KEY = "prompt_enhancer_reward"
_FORMAT = re.compile(r"Let me solve this step by step.\n<think>.*?</think>\n<answer>.*?</answer>$")
_ANSWER = re.compile(r"<answer>(.*?)</answer>")
_IMAGE_TIMEOUT_SECS = 600.0
_SCORER_TIMEOUT_SECS = 300.0
_SCORER_ATTEMPTS = 3
_RETRY_DELAY_SECS = 1.0

_clients: dict[tuple[str, tuple[str, int]], ImageGenerationClient] = {}
_next_scorer = itertools.count()


class PromptEnhancerRewardConfig(BaseConfig):
    pe_generator_model: A[str, Arg(help="The frozen diffusion model's name in --sglang-config.")] = "generator"
    pe_width: A[int, Arg(help="Image width; the engine must produce exactly this size.")] = 512
    pe_height: A[int, Arg(help="Image height; the engine must produce exactly this size.")] = 512
    pe_num_inference_steps: A[int | None, Arg(help="Denoising steps; unset uses the model's default.")] = None
    pe_guidance_scale: A[float | None, Arg(help="CFG scale; unset uses the model's default.")] = None
    pe_seed_per_sample: A[
        bool, Arg(help="Draw the image seed per sample instead of one per group (RePrompt's own run, for parity).")
    ] = False
    pe_scorer_pool: A[
        str | None, Arg(help="The reward pool in --reward-pool-config that scores the image, e.g. hpsv3.")
    ] = None
    pe_scorer_urls: A[
        str | None,
        Arg(help="Comma-separated base URLs of scorer_server.py (ImageReward) processes, when no pool scores."),
    ] = None
    pe_image_weight: A[
        float, Arg(help="Weight of the image score in the reward; RePrompt's 0.25 for ImageReward.")
    ] = 0.25


class PromptEnhancerLogConfig(BaseConfig):
    pe_sample_log: A[str | None, Arg(help="Directory for one JSONL of scored samples per rollout.")] = None


def format_reward(completion: str) -> float:
    return 1.0 if _FORMAT.match(completion) else -1.0


def length_reward(completion: str) -> float:
    match = _ANSWER.search(completion)
    if not match:
        return 0.0
    return 1.0 if 15 < len(match.group(1).split()) < 77 else -1.0


def t2i_prompt(completion: str) -> str:
    match = _ANSWER.search(completion)
    return match.group(1) if match else "a photo"


def image_seed(ori_prompt: str, draw: int | None) -> int:
    """A 31-bit seed fixed by the original prompt and one draw of it (a group, or a sample)."""
    digest = hashlib.blake2b(f"{draw}\x00{ori_prompt}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


async def prompt_enhancer_rm(args, sample: Sample, **kwargs) -> float | None:
    if (args.pe_scorer_pool is None) == (args.pe_scorer_urls is None):
        raise NonRetryableError("prompt_enhancer_rm scores with one of --pe-scorer-pool and --pe-scorer-urls")
    if not args.rollout_skip_special_tokens:
        raise NonRetryableError(
            "prompt_enhancer_rm scores the completion text: run with --rollout-skip-special-tokens"
        )
    completion = ASSISTANT_PREFIX + sample.response
    ori_prompt = sample.metadata[ORI_PROMPT_KEY]
    request = ImageRequest(
        prompt=t2i_prompt(completion),
        seed=image_seed(ori_prompt, sample.index if args.pe_seed_per_sample else sample.group_index),
        width=args.pe_width,
        height=args.pe_height,
        num_inference_steps=args.pe_num_inference_steps,
        guidance_scale=args.pe_guidance_scale,
    )
    started = time.monotonic()
    try:
        png = await _client(args).generate_image(request)
    except ImageGenerationUnavailable as e:
        logger.warning(f"no image for sample {sample.index}: {e}")
        return None
    rendered = time.monotonic()
    image_score = await _score_image(args, png=png, prompt=ori_prompt, kwargs=kwargs)
    if image_score is None:
        return None
    parts = {
        "t2i": args.pe_image_weight * image_score,
        "format": format_reward(completion),
        "length": length_reward(completion),
        "image_score": image_score,
        "t2i_prompt": request.prompt,
        "seed": request.seed,
        "render_secs": rendered - started,
        "score_secs": time.monotonic() - rendered,
    }
    sample.metadata[REWARD_KEY] = parts
    return parts["t2i"] + parts["format"] + parts["length"]


prompt_enhancer_rm.config_class = PromptEnhancerRewardConfig


def _client(args) -> ImageGenerationClient:
    # One client per router, so every call shares the client's connection pool and concurrency cap.
    key = (args.pe_generator_model, tuple(args.sglang_model_routers.get(args.pe_generator_model) or ()))
    if key not in _clients:
        _clients[key] = image_generation_client(
            args,
            args.pe_generator_model,
            timeout_secs=_IMAGE_TIMEOUT_SECS,
            retry_initial_delay_secs=_RETRY_DELAY_SECS,
        )
    return _clients[key]


async def _score_image(args, *, png: bytes, prompt: str, kwargs: dict) -> float | None:
    if args.pe_scorer_pool is None:
        return await _score_on_servers(args.pe_scorer_urls.split(","), png=png, prompt=prompt)
    if (pool := kwargs.get(f"{args.pe_scorer_pool}_pool")) is None:
        raise NonRetryableError(f"--pe-scorer-pool {args.pe_scorer_pool} names no pool in --reward-pool-config")
    (score,) = await pool.score([png], [prompt])
    return score


async def _score_on_servers(urls: list[str], *, png: bytes, prompt: str) -> float | None:
    payload = {"image_b64": base64.b64encode(png).decode(), "prompt": prompt}
    client = GeneralHttpClientProvider.client()
    for attempt in range(_SCORER_ATTEMPTS):
        url = urls[next(_next_scorer) % len(urls)]
        try:
            response = await client.post(f"{url}/score", json=payload, timeout=_SCORER_TIMEOUT_SECS)
            response.raise_for_status()
            return float(response.json()["image_reward"])
        except (httpx.HTTPError, ValueError, KeyError) as e:
            logger.warning(f"scorer {url} failed (attempt {attempt + 1}/{_SCORER_ATTEMPTS}): {e!r}")
            await asyncio.sleep(_RETRY_DELAY_SECS * 2**attempt)
    return None


def log_rollout_data(rollout_id, args, samples: list[Sample], rollout_extra_metrics, rollout_time) -> bool:
    """`--custom-rollout-log-function-path`: reward terms, render and score latencies, and the scored samples."""
    scored = [s for s in samples if REWARD_KEY in s.metadata]
    if not scored:
        return False
    parts = [s.metadata[REWARD_KEY] for s in scored]
    metrics = {f"rollout/pe/{key}": float(np.mean([p[key] for p in parts])) for key in ("t2i", "format", "length")}
    # Samples saved before the pool, as debug rollout data, hold ImageReward's score under its old key.
    metrics["rollout/pe/image_score"] = float(np.mean([p.get("image_score", p.get("image_reward")) for p in parts]))
    metrics["rollout/pe/reward"] = float(np.mean([s.get_reward_value(args) for s in scored]))
    metrics["rollout/pe/completion_tokens"] = float(np.mean([s.response_length for s in scored]))
    for key in ("render_secs", "score_secs"):
        values = [p[key] for p in parts]
        metrics[f"rollout/pe/{key}_p50"] = float(np.percentile(values, 50))
        metrics[f"rollout/pe/{key}_p95"] = float(np.percentile(values, 95))
    metrics["rollout/pe/unscored"] = float(len(samples) - len(scored))
    metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
    logger.info(f"prompt_enhancer {rollout_id}: {json.dumps(metrics)}")
    tracking.log(args, metrics, step_key="rollout/step")
    if args.pe_sample_log:
        os.makedirs(args.pe_sample_log, exist_ok=True)
        with open(os.path.join(args.pe_sample_log, f"rollout_{rollout_id:05d}.jsonl"), "w") as f:
            for s, p in zip(scored, parts, strict=True):
                row = {"prompt": s.metadata[ORI_PROMPT_KEY], "completion": ASSISTANT_PREFIX + s.response}
                f.write(json.dumps(row | {"group": s.group_index} | p) + "\n")
    return False


log_rollout_data.config_class = PromptEnhancerLogConfig
