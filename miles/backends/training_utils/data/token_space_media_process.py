"""Multimodal training inputs computed on the trainer from shared media references.

With ``--multimodal-token-space-processing`` the rollout ships ordered media references instead of
processor tensors. Each trainer rank rebuilds the tensors with SGLang's ``TokenSpaceProcessor``,
the same core the rollout engines serve with, built from their server args the way they build it,
so training sees the features the rollout engines saw.
"""

import asyncio
import functools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from miles.utils.processing_utils import extract_multimodal_train_inputs


@dataclass(frozen=True)
class RolloutEngineProcessorArgs:
    """The rollout engines' server args that decide what SGLang's token-space processor outputs.

    The trainer builds its processor from the same values, so it computes the features the engines computed.
    """

    model_path: str
    tokenizer_path: str
    tokenizer_mode: str
    tokenizer_backend: str
    revision: str | None
    model_config_parser: str
    model_override_args_json: str
    image_processor_backend: str
    disable_fast_image_processor: bool
    mm_process_config_json: str

    @classmethod
    def from_sglang_config(cls, sglang) -> "RolloutEngineProcessorArgs":
        """Read the values every rollout engine shares; ``common_value`` rejects one the engine groups disagree on."""
        model_path = sglang.common_value("model_path")
        # Canonical JSON so the args can key the processor cache.
        mm_process_config_json = json.dumps(sglang.common_value("mm_process_config") or {}, sort_keys=True)
        return cls(
            model_path=model_path,
            tokenizer_path=sglang.common_value("tokenizer_path") or model_path,
            tokenizer_mode=sglang.common_value("tokenizer_mode"),
            tokenizer_backend=sglang.common_value("tokenizer_backend"),
            revision=sglang.common_value("revision"),
            model_config_parser=sglang.common_value("model_config_parser"),
            model_override_args_json=sglang.common_value("json_model_override_args") or "{}",
            image_processor_backend=sglang.common_value("image_processor_backend"),
            disable_fast_image_processor=sglang.common_value("disable_fast_image_processor"),
            mm_process_config_json=mm_process_config_json,
        )


def compute_multimodal_train_inputs(
    rollout_engine_args: RolloutEngineProcessorArgs,
    multimodal_inputs: Sequence[dict[str, list[str]] | None],
    tokens: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> list[dict[str, torch.Tensor] | None]:
    """Rebuild each sample's processor tensors on the trainer from its media references and expanded tokens.

    Args:
        rollout_engine_args: The rollout engines' server args that decide the processor output.
        multimodal_inputs: Each sample's media references, or None for a text-only sample.
        tokens: Each sample's tokens, with the media placeholders SGLang expanded.
        device: This rank's device; the tensors land here, and it stands in for the engine's GPU.

    Returns:
        Each sample's ``multimodal_train_inputs``, or None for a text-only sample.
    """
    trainer_processor = _trainer_processor(rollout_engine_args, device)
    return [
        (
            _sample_train_inputs(trainer_processor, sample_multimodal_inputs, sample_tokens, device=device)
            if sample_multimodal_inputs
            else None
        )
        for sample_multimodal_inputs, sample_tokens in zip(multimodal_inputs, tokens, strict=True)
    ]


class _TrainerProcessor(NamedTuple):
    core: Any
    # The image device the engine's processor wrapper passes, mapped to this rank's device.
    image_device: str | None


@functools.cache
def _trainer_processor(rollout_engine_args: RolloutEngineProcessorArgs, device: torch.device) -> _TrainerProcessor:
    # Deferred: only SGLang builds with the token-space processor provide these modules.
    from sglang.srt.multimodal.processors.processor_config import MultimodalProcessorConfig
    from sglang.srt.multimodal.token_space_mm_process.processor import TokenSpaceProcessor
    from sglang.srt.multimodal.token_space_mm_process.registry import register_token_space_models, strategy_for
    from sglang.srt.utils.hf_transformers import get_config, get_processor

    # Miles launches every rollout engine with trust_remote_code.
    trust_remote_code = True
    register_token_space_models()
    hf_config = get_config(
        rollout_engine_args.model_path,
        trust_remote_code=trust_remote_code,
        revision=rollout_engine_args.revision,
        model_override_args=json.loads(rollout_engine_args.model_override_args_json),
        model_config_parser=rollout_engine_args.model_config_parser,
    )
    strategy_class = strategy_for(hf_config)
    if strategy_class is None:
        raise ValueError(f"--multimodal-token-space-processing does not support {hf_config.architectures}")
    image_processor_backend = (
        "pil" if rollout_engine_args.disable_fast_image_processor else rollout_engine_args.image_processor_backend
    )
    hf_processor = get_processor(
        rollout_engine_args.tokenizer_path,
        tokenizer_mode=rollout_engine_args.tokenizer_mode,
        trust_remote_code=trust_remote_code,
        revision=rollout_engine_args.revision,
        image_processor_backend=image_processor_backend,
        tokenizer_backend=rollout_engine_args.tokenizer_backend,
        model_name=rollout_engine_args.model_path,
    )
    # SGLang's worker threads decode on bare "cuda", which a new thread resolves to GPU 0.
    worker_thread_device = device if device.type == "cuda" else None
    core = TokenSpaceProcessor.for_strategy(strategy_class)(
        hf_config,
        hf_processor,
        processor_config=MultimodalProcessorConfig(
            image_processor_backend=rollout_engine_args.image_processor_backend,
            disable_fast_image_processor=rollout_engine_args.disable_fast_image_processor,
            mm_process_config=json.loads(rollout_engine_args.mm_process_config_json),
        ),
        worker_thread_device=worker_thread_device,
    )
    image_device = _image_device(hf_processor, image_processor_backend=image_processor_backend, device=device)
    return _TrainerProcessor(core=core, image_device=image_device)


def _image_device(hf_processor, *, image_processor_backend: str, device: torch.device) -> str | None:
    """The image device SGLang's processor wrapper derives, with this rank's device in place of the engine's."""
    from sglang.srt.multimodal.serving_preprocessing import (
        apply_npu_image_preprocess_patches,
        fast_image_processor_device,
    )
    from transformers import BaseImageProcessor

    if image_processor_backend == "pil" or not isinstance(hf_processor.image_processor, BaseImageProcessor):
        return None
    image_device = fast_image_processor_device(
        hf_processor, base_gpu_id=device.index, rl_on_policy_target=None, true_on_policy_enabled=False
    )
    if image_device == "npu":
        apply_npu_image_preprocess_patches(hf_processor)
    return image_device


def _sample_train_inputs(
    trainer_processor: _TrainerProcessor,
    multimodal_inputs: dict[str, list[str]],
    tokens: torch.Tensor,
    *,
    device: torch.device,
):
    input_ids = tokens.tolist()
    # No video device, as the engine's wrapper: the core takes it from the image device or the mm_process_config.
    batch_feature = asyncio.run(
        trainer_processor.core.process(
            image_data=multimodal_inputs.get("images"),
            video_data=multimodal_inputs.get("videos"),
            audio_data=multimodal_inputs.get("audios"),
            input_ids=input_ids,
            mm_token_expansion_start_len=len(input_ids),
            image_device=trainer_processor.image_device,
        )
    )
    # mm_modalities and mm_spans tell SGLang where each media's tower output goes; the model does not take them.
    train_inputs = extract_multimodal_train_inputs(batch_feature)
    return {key: value.to(device) for key, value in train_inputs.items() if key not in ("mm_modalities", "mm_spans")}
