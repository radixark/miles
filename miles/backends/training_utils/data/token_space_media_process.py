"""Multimodal training inputs computed on the trainer from shared media references.

With ``--multimodal-token-space-processing`` the rollout ships ordered media references instead of
processor tensors. Each trainer rank rebuilds the tensors with SGLang's ``TokenSpaceProcessor``,
the same core the engines serve with, built from the engines' ``--sglang-mm-process-config`` and
run on the device the engine uses, so training sees the features the rollout engine saw.
"""

import asyncio
import functools
import json
from collections.abc import Sequence

import torch

from miles.utils.processing_utils import extract_multimodal_train_inputs


def compute_multimodal_train_inputs(
    hf_checkpoint: str,
    mm_process_config: dict | None,
    multimodal_inputs: Sequence[dict[str, list[str]] | None],
    tokens: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> list[dict[str, torch.Tensor] | None]:
    """Rebuild each sample's processor tensors on the trainer from its media references and expanded tokens.

    Args:
        hf_checkpoint: The checkpoint whose config and processor SGLang serves.
        mm_process_config: The engines' ``--sglang-mm-process-config``.
        multimodal_inputs: Each sample's media references, or None for a text-only sample.
        tokens: Each sample's tokens, with the media placeholders SGLang expanded.
        device: Where the image and video processors run, as on the engine, and where the tensors land.

    Returns:
        Each sample's ``multimodal_train_inputs``, or None for a text-only sample.
    """
    # Canonical JSON so the config can key the processor cache.
    processor = _token_space_processor(hf_checkpoint, json.dumps(mm_process_config or {}, sort_keys=True))
    return [
        (
            _sample_train_inputs(processor, sample_multimodal_inputs, sample_tokens, device=device)
            if sample_multimodal_inputs
            else None
        )
        for sample_multimodal_inputs, sample_tokens in zip(multimodal_inputs, tokens, strict=True)
    ]


@functools.cache
def _token_space_processor(hf_checkpoint: str, mm_process_config_json: str):
    # Deferred: only SGLang builds with the token-space processor provide these modules.
    from sglang.srt.multimodal.processors.processor_config import MultimodalProcessorConfig
    from sglang.srt.multimodal.token_space_mm_process.processor import TokenSpaceProcessor
    from sglang.srt.multimodal.token_space_mm_process.registry import strategy_for
    from sglang.srt.utils.hf_transformers import get_config, get_processor

    hf_config = get_config(hf_checkpoint, trust_remote_code=True)
    strategy_class = strategy_for(hf_config)
    if strategy_class is None:
        raise ValueError(f"--multimodal-token-space-processing does not support {hf_config.architectures}")
    # The engine's default image processor backend; the engine resolves "auto" the same way.
    hf_processor = get_processor(hf_checkpoint, trust_remote_code=True, image_processor_backend="auto")
    return TokenSpaceProcessor.for_strategy(strategy_class)(
        hf_config,
        hf_processor,
        processor_config=MultimodalProcessorConfig(
            image_processor_backend="auto", mm_process_config=json.loads(mm_process_config_json)
        ),
    )


def _sample_train_inputs(
    processor, multimodal_inputs: dict[str, list[str]], tokens: torch.Tensor, *, device: torch.device
):
    input_ids = tokens.tolist()
    batch_feature = asyncio.run(
        processor.process(
            image_data=multimodal_inputs.get("images"),
            video_data=multimodal_inputs.get("videos"),
            audio_data=multimodal_inputs.get("audios"),
            input_ids=input_ids,
            mm_token_expansion_start_len=len(input_ids),
            image_device=str(device),
            video_device=str(device),
        )
    )
    # mm_modalities and mm_spans tell SGLang where each media's tower output goes; the model does not take them.
    train_inputs = extract_multimodal_train_inputs(batch_feature)
    return {key: value.to(device) for key, value in train_inputs.items() if key not in ("mm_modalities", "mm_spans")}
