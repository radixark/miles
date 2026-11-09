"""Multimodal training inputs computed on the trainer from shared media references.

With ``--multimodal-token-space-processing`` the rollout ships ordered media references instead of
processor tensors. Each trainer rank rebuilds the tensors with SGLang's ``TokenSpaceProcessor``,
the same core the rollout engines serve with, built from their server args the way they build it,
so training sees the features the rollout engines saw.

For models that take them instead of computing their own, each sample also gets what SGLang computes from the
expanded tokens:

- ``mm_image_positions`` / ``mm_video_positions``: [num_rows] sample-local token index of every vision tower
  output row, from the processor's ``mm_spans``;
- its mRoPE positions, [num_tokens, 3], text-only samples included. They are not multimodal inputs but the
  model's ``position_ids``, so they travel apart; ``get_batch`` packs them like the tokens.

The rebuild runs on a background thread from the moment the rollout data arrives. Until a sample's rebuild
finishes, its entries are ``Future`` objects; ``get_batch`` resolves the entries of the micro-batch it assembles
and waits only for those.
"""

import asyncio
import functools
import json
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from miles.utils.processing_utils import extract_multimodal_train_inputs


class _SampleBuild(NamedTuple):
    """One sample's rebuilt processor tensors and mRoPE positions."""

    multimodal_train_inputs: dict[str, torch.Tensor] | None
    mrope_position_ids: torch.Tensor | None


class TokenSpaceTrainInputs(NamedTuple):
    # Each sample's build future, or its finished processor tensors with the tower row placement; None for a
    # text-only sample.
    multimodal_train_inputs: list[Future | dict[str, torch.Tensor] | None]
    # The same build futures, read for each sample's [num_tokens, 3] positions; None for a model that computes
    # its own.
    mrope_position_ids: list[Future | torch.Tensor] | None


def resolved_multimodal_train_inputs(
    entries: Sequence[Future | dict[str, torch.Tensor] | None],
) -> list[dict[str, torch.Tensor] | None]:
    """Each sample's processor tensors, waiting for the builds still running in the background."""
    return [entry.result().multimodal_train_inputs if isinstance(entry, Future) else entry for entry in entries]


def resolved_mrope_position_ids(entries: Sequence[Future | torch.Tensor]) -> list[torch.Tensor] | None:
    """Each sample's positions, or None when the model has no mRoPE and the processor computed none."""
    positions = [entry.result().mrope_position_ids if isinstance(entry, Future) else entry for entry in entries]
    if all(sample_positions is None for sample_positions in positions):
        return None
    assert all(sample_positions is not None for sample_positions in positions), "mRoPE positions for only some samples"
    return positions


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
    def from_sglang_config(cls, sglang, *, hf_checkpoint: str) -> "RolloutEngineProcessorArgs":
        """Read the values the actor engines share; ``actor_value`` rejects one their groups disagree on.

        A frozen reference, reward or eval model may load its own checkpoint and processor settings. The trainer
        rebuilds inputs for samples the actor engines expanded, so it takes theirs; with no actor engine
        (``--debug-train-only``), the trainer's checkpoint, which is their default ``model_path``.
        """
        actor_model_paths = {
            group.model_path for model in sglang.models if model.update_weights for group in model.server_groups
        }
        assert len(actor_model_paths) <= 1, f"the actor engines load different models: {sorted(actor_model_paths)}"
        model_path = actor_model_paths.pop() if actor_model_paths else hf_checkpoint
        # Canonical JSON so the args can key the processor cache.
        mm_process_config_json = json.dumps(sglang.actor_value("mm_process_config") or {}, sort_keys=True)
        return cls(
            model_path=model_path,
            tokenizer_path=sglang.actor_value("tokenizer_path") or model_path,
            tokenizer_mode=sglang.actor_value("tokenizer_mode"),
            tokenizer_backend=sglang.actor_value("tokenizer_backend"),
            revision=sglang.actor_value("revision"),
            model_config_parser=sglang.actor_value("model_config_parser"),
            model_override_args_json=sglang.actor_value("json_model_override_args") or "{}",
            image_processor_backend=sglang.actor_value("image_processor_backend"),
            disable_fast_image_processor=sglang.actor_value("disable_fast_image_processor"),
            mm_process_config_json=mm_process_config_json,
        )


def compute_multimodal_train_inputs(
    trainer_processor: "TrainerProcessor",
    multimodal_inputs: Sequence[dict[str, list[str]] | None],
    tokens: Sequence[torch.Tensor],
    *,
    mm_processor_worker_num: int,
    with_sglang_placement: bool,
    device: torch.device,
) -> TokenSpaceTrainInputs:
    """Start rebuilding each sample's processor tensors and mRoPE positions on the trainer from its media references.

    Args:
        trainer_processor: The processor built from the rollout engines' server args (``build_trainer_processor``).
        multimodal_inputs: Each sample's media references, or None for a text-only sample.
        tokens: Each sample's tokens, with the media placeholders SGLang expanded.
        mm_processor_worker_num: Samples built at once (--trainer-mm-processor-worker-num).
        with_sglang_placement: Also return the tower row placement and the mRoPE positions SGLang computes.
        device: This rank's device; the tensors land here, and it stands in for the engine's GPU.

    Returns:
        Each sample's build future, which resolves to its processor tensors and, with SGLang's placement, its
        positions once the background rebuild reaches the sample.
    """
    builds = [
        _submit_sample_build(
            trainer_processor,
            sample_multimodal_inputs,
            sample_tokens,
            mm_processor_worker_num=mm_processor_worker_num,
            with_sglang_placement=with_sglang_placement,
            device=device,
        )
        for sample_multimodal_inputs, sample_tokens in zip(multimodal_inputs, tokens, strict=True)
    ]
    return TokenSpaceTrainInputs(builds, list(builds) if with_sglang_placement else None)


@functools.cache
def _build_executor(mm_processor_worker_num: int) -> ThreadPoolExecutor:
    # One sample per thread; the processor runs as many media workers (--trainer-mm-processor-worker-num).
    return ThreadPoolExecutor(max_workers=mm_processor_worker_num, thread_name_prefix="miles-token-space-build")


def _submit_sample_build(
    trainer_processor: "TrainerProcessor",
    multimodal_inputs: dict[str, list[str]] | None,
    tokens: torch.Tensor,
    *,
    mm_processor_worker_num: int,
    with_sglang_placement: bool,
    device: torch.device,
) -> Future:
    """Queue one sample's rebuild; the executor's future carries its ``_SampleBuild``, or the error that stopped it."""

    def rebuild() -> _SampleBuild:
        if multimodal_inputs:
            sample_train_inputs, positions = _sample_train_inputs(
                trainer_processor,
                multimodal_inputs,
                tokens,
                with_sglang_placement=with_sglang_placement,
                device=device,
            )
        else:
            sample_train_inputs = None
            positions = _text_mrope_positions(trainer_processor, tokens) if with_sglang_placement else None
        return _SampleBuild(sample_train_inputs, None if positions is None else positions.T.contiguous().to(device))

    return _build_executor(mm_processor_worker_num).submit(rebuild)


class TrainerProcessor(NamedTuple):
    """The trainer's copy of the rollout engines' token-space processor; the training actor builds it once."""

    core: Any
    # The image device the engine's processor wrapper passes, mapped to this rank's device.
    image_device: str | None


def build_trainer_processor(
    rollout_engine_args: RolloutEngineProcessorArgs,
    *,
    mm_processor_worker_num: int,
    mm_io_worker_num: int,
    device: torch.device,
) -> TrainerProcessor:
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
            # SGLang runs the HF processor on that many isolated clones; 0 I/O threads lets it pick its own count.
            mm_processor_worker_num=mm_processor_worker_num,
            mm_io_worker_num=mm_io_worker_num,
        ),
        worker_thread_device=worker_thread_device,
    )
    image_device = _image_device(hf_processor, image_processor_backend=image_processor_backend, device=device)
    return TrainerProcessor(core=core, image_device=image_device)


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
    trainer_processor: TrainerProcessor,
    multimodal_inputs: dict[str, list[str]],
    tokens: torch.Tensor,
    *,
    with_sglang_placement: bool,
    device: torch.device,
):
    from sglang.srt.multimodal.modality import Modality

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
            compute_mrope_positions=with_sglang_placement,
        )
    )
    train_inputs = extract_multimodal_train_inputs(batch_feature)
    modalities, spans = train_inputs.pop("mm_modalities"), train_inputs.pop("mm_spans")
    mrope_positions = train_inputs.pop("mrope_positions", None)
    train_inputs.pop("mrope_position_delta", None)
    train_inputs = {key: value.to(device) for key, value in train_inputs.items()}
    if not with_sglang_placement:
        return train_inputs, None
    # The token index of every tower output row, from where SGLang placed each media (half-open spans).
    for modality, key in ((Modality.IMAGE, "mm_image_positions"), (Modality.VIDEO, "mm_video_positions")):
        rows = [
            torch.arange(start, end)
            for media, media_spans in zip(modalities, spans, strict=True)
            if media == modality
            for start, end in media_spans
        ]
        if rows:
            train_inputs[key] = torch.cat(rows).to(device)
    return train_inputs, mrope_positions


def _text_mrope_positions(trainer_processor: TrainerProcessor, tokens: torch.Tensor) -> torch.Tensor | None:
    mrope_positions, _ = trainer_processor.core.strategy.compute_mrope_positions({"input_ids": tokens.tolist()})
    return mrope_positions
