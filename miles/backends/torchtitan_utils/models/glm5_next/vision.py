import functools

import torch

from miles.backends.torchtitan_utils.image_embeddings import ImageInputs, select_pixel_values
from miles.utils.processing_utils import load_processor
from miles_plugins.models.glm5_next.vision import build_glm5_next_visual


class Glm5NextImageEncoder:
    """The frozen vision tower and the image processor that rebuilds pixels from PNG bytes."""

    def __init__(self, hf_checkpoint: str, *, device: torch.device, dtype: torch.dtype):
        self.visual = build_glm5_next_visual(hf_checkpoint, device=device, dtype=dtype)
        self.merge_size = self.visual.spatial_merge_size
        self._hf_checkpoint = hf_checkpoint
        self._device = device
        self._dtype = dtype

    @functools.cached_property
    def image_processor(self):
        """Loaded on the first PNG microbatch; runs that ship pixel tensors never need it."""
        processor = load_processor(self._hf_checkpoint, trust_remote_code=True)
        if processor is None or getattr(processor, "image_processor", None) is None:
            processor = _sglang_glm5_next_processor(self._hf_checkpoint)
        if processor is None or getattr(processor, "image_processor", None) is None:
            raise RuntimeError(f"{self._hf_checkpoint} has no image processor to rebuild PNG images with")
        return processor.image_processor

    @torch.no_grad()
    def encode(self, inputs: ImageInputs, images: list[int]) -> torch.Tensor:
        pixel_values = select_pixel_values(inputs, images, lambda: self.image_processor)
        grid_thw = inputs.grid_thw[images]
        return self.visual(
            pixel_values.to(device=self._device, dtype=self._dtype), grid_thw=grid_thw.to(self._device)
        ).pooler_output


def _sglang_glm5_next_processor(hf_checkpoint: str):
    """SGLang's GLM-5.3 processor, the one the rollout engine runs.

    Transformers releases without a native GLM-5.3 processor (5.12.1, the
    pin of the torchtitan image) make AutoProcessor return a bare tokenizer
    for this checkpoint, which load_processor discards. Rebuilding with the
    engine's own processor keeps training pixels equal to rollout pixels.
    None when this SGLang does not ship it.
    """
    try:
        from sglang.srt.configs.glm5_next_processing import Glm5NextProcessor
    except ImportError:
        return None
    return Glm5NextProcessor.from_pretrained(hf_checkpoint, trust_remote_code=True)
