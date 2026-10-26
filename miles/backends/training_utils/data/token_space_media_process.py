"""Multimodal training inputs computed on the trainer from shared media references.

With ``--multimodal-token-space-processing`` the rollout ships ordered media references instead of
processor tensors. Each trainer rank rebuilds the tensors with SGLang's ``TokenSpaceProcessor``,
the same core the engines serve with, built from the engines' ``--sglang-mm-process-config`` and
run on the device the engine uses, so training sees the features the rollout engine saw.

Besides the processor tensors, each sample carries what SGLang computes from the expanded tokens:

- ``mm_image_positions`` / ``mm_video_positions``: [num_rows] sample-local token index of every vision tower
  output row, from the processor's ``mm_spans``;
- ``mrope_position_ids``: [num_tokens, 3] mRoPE positions, token-major so the samples of a micro-batch
  concatenate along dim 0. Text-only samples carry them too.

The rebuild runs in the background as soon as the rollout data arrives; ``get_batch`` waits only
for the samples of the micro-batch it assembles.
"""

import asyncio
import functools
import json
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future

import torch

from miles.utils.processing_utils import extract_multimodal_train_inputs
from miles.utils.timer import Timer

# Processor worker threads.
_NUM_WORKERS = 16


class PendingTrainInputs:
    """One sample's processor tensors, still being rebuilt in the background.

    Samples that share their media and the tokens up to the last media placeholder share one rebuild of that
    prefix; each sample extends the prefix's mRoPE positions over its own tokens after it.
    """

    def __init__(self, future: Future, num_tokens: int):
        self._future = future
        self._num_tokens = num_tokens

    def result(self) -> dict[str, torch.Tensor] | None:
        shared = self._future.result()
        if shared is None or "mrope_prefix" not in shared:
            return shared
        train_inputs = {key: value for key, value in shared.items() if key not in ("mrope_prefix", "mrope_delta")}
        prefix, delta = shared["mrope_prefix"], shared["mrope_delta"]
        # Past the last media every position is the token index plus SGLang's mrope_position_delta.
        tail = torch.arange(prefix.shape[0], self._num_tokens, device=prefix.device) + delta
        train_inputs["mrope_position_ids"] = torch.cat([prefix, tail[:, None].expand(-1, 3)])
        return train_inputs


def resolve_train_inputs(train_inputs):
    """The sample's processor tensors, waiting for them if they are still being rebuilt.

    The wait is the media pipeline's bubble, logged as ``perf/token_space_media_wait_time``.
    """
    if not isinstance(train_inputs, PendingTrainInputs):
        return train_inputs
    start = time.perf_counter()
    result = train_inputs.result()
    Timer().add("token_space_media_wait", time.perf_counter() - start)
    return result


def prewarm_token_space_processor(hf_checkpoint: str, mm_process_config: dict | None, *, device: torch.device) -> None:
    """Build the processor in the background, overlapping actor initialization."""
    _processor_future(hf_checkpoint, _config_key(mm_process_config), device)


def start_multimodal_train_inputs(
    hf_checkpoint: str,
    mm_process_config: dict | None,
    multimodal_inputs: Sequence[dict[str, list[str]] | None],
    tokens: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> list[PendingTrainInputs]:
    """Start rebuilding each sample's processor tensors from its media references and expanded tokens.

    Args:
        hf_checkpoint: The checkpoint whose config and processor SGLang serves.
        mm_process_config: The engines' ``--sglang-mm-process-config``.
        multimodal_inputs: Each sample's media references, or None for a text-only sample.
        tokens: Each sample's tokens, with the media placeholders SGLang expanded.
        device: Where the image and video processors run, as on the engine, and where the tensors stay.

    Returns:
        Each sample's pending ``multimodal_train_inputs``.
    """
    processor_future = _processor_future(hf_checkpoint, _config_key(mm_process_config), device)
    loop = _background_loop(device)
    placeholder_ids = _placeholder_token_ids(processor_future.result())
    by_prefix: dict[tuple, Future] = {}
    pending = []
    for sample_multimodal_inputs, sample_tokens in zip(multimodal_inputs, tokens, strict=True):
        input_ids = sample_tokens.tolist()
        if not sample_multimodal_inputs:
            future = asyncio.run_coroutine_threadsafe(
                _text_train_inputs(processor_future, input_ids, device=device), loop
            )
            pending.append(PendingTrainInputs(future, len(input_ids)))
            continue
        prefix = input_ids[: _media_prefix_len(input_ids, placeholder_ids)]
        key = (json.dumps(sample_multimodal_inputs, sort_keys=True), tuple(prefix))
        if key not in by_prefix:
            by_prefix[key] = asyncio.run_coroutine_threadsafe(
                _media_train_inputs(processor_future, sample_multimodal_inputs, prefix, device=device), loop
            )
        pending.append(PendingTrainInputs(by_prefix[key], len(input_ids)))
    return pending


def _config_key(mm_process_config: dict | None) -> str:
    # Canonical JSON so the config can key the processor cache.
    return json.dumps(mm_process_config or {}, sort_keys=True)


def _placeholder_token_ids(processor) -> set[int]:
    params = processor.strategy.mm_token_expansion_params
    return {
        params[key] for key in ("image_token_id", "video_token_id", "audio_token_id") if params.get(key) is not None
    }


def _media_prefix_len(input_ids: list[int], placeholder_ids: set[int]) -> int:
    """Length of the tokens up to and including the last media placeholder."""
    for index in range(len(input_ids) - 1, -1, -1):
        if input_ids[index] in placeholder_ids:
            return index + 1
    raise ValueError("a sample with media references has no media placeholder token")


@functools.cache
def _background_loop(device: torch.device) -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()

    def run():
        # A new thread's current CUDA device is GPU 0, whichever GPU this rank trains on. The processor is
        # built on this thread, and SGLang starts its media IO threads on the current device.
        if device.type == "cuda":
            torch.cuda.set_device(device)
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=run, name="miles-token-space-media", daemon=True).start()
    return loop


@functools.cache
def _processor_future(hf_checkpoint: str, mm_process_config_json: str, device: torch.device) -> Future:
    async def build():
        return _build_processor(hf_checkpoint, mm_process_config_json, device)

    return asyncio.run_coroutine_threadsafe(build(), _background_loop(device))


def _build_processor(hf_checkpoint: str, mm_process_config_json: str, device: torch.device):
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
            image_processor_backend="auto",
            mm_process_config=json.loads(mm_process_config_json),
            mm_processor_worker_num=_NUM_WORKERS,
            # The features feed this rank's model, so skip the CPU copy the engines make for transport.
            keep_mm_features_on_device=device.type == "cuda",
        ),
        compute_mrope_positions=True,
    )


async def _media_train_inputs(
    processor: Future, multimodal_inputs: dict[str, list[str]], prefix: list[int], *, device: torch.device
):
    from sglang.srt.multimodal.modality import Modality

    processor = await asyncio.wrap_future(processor)
    batch_feature = await processor.process(
        image_data=multimodal_inputs.get("images"),
        video_data=multimodal_inputs.get("videos"),
        audio_data=multimodal_inputs.get("audios"),
        input_ids=prefix,
        # The tokens already carry the engine's expansion, so nothing past their end is expanded again.
        mm_token_expansion_start_len=len(prefix),
        image_device=str(device),
        video_device=str(device),
    )
    train_inputs = extract_multimodal_train_inputs(batch_feature)
    modalities, spans = train_inputs.pop("mm_modalities"), train_inputs.pop("mm_spans")
    mrope_positions = train_inputs.pop("mrope_positions", None)
    mrope_delta = train_inputs.pop("mrope_position_delta", None)
    train_inputs = {key: value.to(device) for key, value in train_inputs.items()}
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
    if mrope_positions is not None:
        train_inputs["mrope_prefix"] = mrope_positions.T.contiguous().to(device)
        train_inputs["mrope_delta"] = int(mrope_delta)
    return train_inputs


async def _text_train_inputs(processor: Future, input_ids: list[int], *, device: torch.device):
    processor = await asyncio.wrap_future(processor)
    mrope_positions, _ = processor.strategy.compute_mrope_positions(input_ids, {})
    if mrope_positions is None:
        return None
    return {"mrope_position_ids": mrope_positions.T.contiguous().to(device)}
