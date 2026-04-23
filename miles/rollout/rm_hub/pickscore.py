from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import numpy as np
import ray
import torch
from PIL import Image

from miles.utils.misc import SingletonMeta
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _feature_tensor(features):
    if isinstance(features, torch.Tensor):
        return features
    return features.pooler_output


def _sample_to_rgb_hwc_uint8(sample: Sample) -> np.ndarray:
    frame_chw = sample.generated_output.detach().cpu()[:, 0, :, :]
    hwc = frame_chw.float().numpy().transpose(1, 2, 0)
    if float(hwc.max()) <= 1.0 + 1e-3:
        hwc = hwc * 255.0
    return np.ascontiguousarray(hwc.clip(0, 255).astype(np.uint8))


class PickScoreScorer(torch.nn.Module):
    """Small local copy of Flow-GRPO's PickScore scorer.

    The scorer consumes final PIL images and prompt strings, then returns one
    scalar reward per prompt/image pair.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        processor_path: str,
        model_path: str,
    ) -> None:
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device)

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> list[float]:
        image_inputs = self.processor(
            images=list(images),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = self.processor(
            text=list(prompts),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {key: value.to(device=self.device) for key, value in image_inputs.items()}
        text_inputs = {key: value.to(device=self.device) for key, value in text_inputs.items()}

        image_embs = _feature_tensor(self.model.get_image_features(**image_inputs))
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        text_embs = _feature_tensor(self.model.get_text_features(**text_inputs))
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)
        scores = scores.diag() / 26.0
        return [float(score) for score in scores.detach().cpu()]


@ray.remote
class PickScoreRewardActor:
    def __init__(
        self,
        *,
        processor_path: str,
        model_path: str,
    ) -> None:
        gpu_ids = ray.get_gpu_ids()
        use_cuda = bool(gpu_ids) and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
        device = "cuda" if use_cuda else "cpu"
        self.scorer = PickScoreScorer(
            device=device,
            processor_path=processor_path,
            model_path=model_path,
        )

    def score_batch(self, images: list[np.ndarray], prompts: list[str]) -> list[float]:
        pil_images = [Image.fromarray(image) for image in images]
        return self.scorer(prompts, pil_images)


class AsyncPickScorePool(metaclass=SingletonMeta):
    """Ray actor pool for GPU PickScore reward inference."""

    def __init__(self, args) -> None:
        num_workers = args.pickscore_num_workers
        num_gpus_per_worker = args.pickscore_num_gpus_per_worker
        self._batch_size = args.pickscore_batch_size
        self._actors = [
            PickScoreRewardActor.options(
                num_cpus=1,
                num_gpus=num_gpus_per_worker,
                scheduling_strategy="DEFAULT",
            ).remote(
                processor_path=args.pickscore_processor_path,
                model_path=args.pickscore_model_path,
            )
            for _ in range(num_workers)
        ]
        self._round_robin_index = 0
        logger.info(
            "Initialized PickScore actor pool with %d workers, %.3f GPUs/worker, batch_size=%d.",
            num_workers,
            num_gpus_per_worker,
            self._batch_size,
        )

    def _next_actor(self):
        i = self._round_robin_index % len(self._actors)
        self._round_robin_index += 1
        return self._actors[i]

    async def score(self, images: list[np.ndarray], prompts: list[str]) -> list[float]:
        refs = []
        for start in range(0, len(images), self._batch_size):
            end = start + self._batch_size
            refs.append(self._next_actor().score_batch.remote(images[start:end], prompts[start:end]))

        loop = asyncio.get_running_loop()
        chunked_scores = await loop.run_in_executor(None, ray.get, refs)
        return [float(score) for chunk in chunked_scores for score in chunk]


async def pickscore_rm(args, samples: Sequence[Sample]) -> list[float]:
    pool = AsyncPickScorePool(args)
    images = [_sample_to_rgb_hwc_uint8(sample) for sample in samples]
    prompts = [sample.prompt for sample in samples]
    return await pool.score(images, prompts)
