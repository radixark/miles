import base64
import inspect
import io
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer as RawTokenizer
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

from miles.utils.hf_config import register_hf_config_aliases

logger = logging.getLogger(__name__)


def _fix_v5_tokenizer_components(tokenizer: PreTrainedTokenizerBase, model_name_or_path: str) -> None:
    # transformers v5's LlamaTokenizerFast rebuilds pre_tokenizer/decoder in
    # __init__, discarding the originals from tokenizer.json.  DeepSeek-V3.2
    # declares LlamaTokenizerFast but actually uses ByteLevel, so without this
    # fix the loaded tokenizer decodes Metaspace ▁ instead of ByteLevel Ġ/Ċ
    # and diverges from the sglang-served tokenizer.  Mirrors sglang's
    # _fix_v5_tokenizer_components (hf_transformers_utils.py).
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None:
        return

    try:
        local_path = Path(model_name_or_path) / "tokenizer.json"
        if local_path.is_file():
            tok_file = str(local_path)
        else:
            tok_file = hf_hub_download(model_name_or_path, "tokenizer.json", local_files_only=True)
        raw = RawTokenizer.from_file(tok_file)
    except Exception as e:
        logger.warning("Could not load tokenizer.json for %s: %s", model_name_or_path, e)
        return

    raw_pre = type(raw.pre_tokenizer).__name__ if raw.pre_tokenizer else None
    loaded_pre = type(backend.pre_tokenizer).__name__ if backend.pre_tokenizer else None

    if raw_pre and loaded_pre and raw_pre != loaded_pre:
        logger.info(
            "Fixing v5 tokenizer component mismatch for %s: pre_tokenizer %s -> %s, decoder %s -> %s",
            model_name_or_path,
            loaded_pre,
            raw_pre,
            type(backend.decoder).__name__ if backend.decoder else None,
            type(raw.decoder).__name__ if raw.decoder else None,
        )
        backend.pre_tokenizer = raw.pre_tokenizer
        backend.decoder = raw.decoder


# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


_TOKENIZER_CACHE: dict[tuple, PreTrainedTokenizerBase] = {}


def _make_cache_key(name_or_path: str, chat_template_path: str | None, kwargs: dict) -> tuple | None:
    try:
        kwargs_items = tuple(sorted(kwargs.items()))
        hash(kwargs_items)
    except TypeError:
        return None
    return (name_or_path, chat_template_path, kwargs_items)


def load_tokenizer(name_or_path: str, chat_template_path: str | None = None, **kwargs) -> PreTrainedTokenizerBase:
    # Cache keyed by (name, chat_template_path, kwargs) — the fast suite creates
    # hundreds of SessionServer / MockSGLangServer fixtures and each previously
    # triggered a fresh AutoTokenizer.from_pretrained, tripping HF Hub rate limits.
    cache_key = _make_cache_key(name_or_path, chat_template_path, kwargs)
    if cache_key is not None and cache_key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[cache_key]

    register_hf_config_aliases()
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    _fix_v5_tokenizer_components(tokenizer, name_or_path)
    if chat_template_path:
        assert os.path.isfile(chat_template_path), (
            f"chat_template_path not found: {chat_template_path}. "
            f"Ensure the path is accessible on this node (e.g. inside the miles repo or on a shared filesystem)."
        )
        with open(chat_template_path) as f:
            tokenizer.chat_template = f.read()
        logger.info("Loaded custom chat template from %s", chat_template_path)

    if cache_key is not None:
        _TOKENIZER_CACHE[cache_key] = tokenizer
    return tokenizer


def build_processor_kwargs(multimodal_inputs: dict | None = None) -> dict:

    modality_forced = {"return_tensors": "pt"}

    result = dict(multimodal_inputs) if multimodal_inputs else {}

    # return_tensors=None for text (input_ids), "pt" for modality-specific outputs.
    # Use per-modality dicts to avoid transformers >=5.0 duplicate kwarg error.
    result["text_kwargs"] = {**result.get("text_kwargs", {}), "return_tensors": None}
    for key in ("audio_kwargs", "images_kwargs", "videos_kwargs"):
        if key in result:
            result[key] = {**result[key], **modality_forced}
        else:
            result[key] = modality_forced.copy()

    return result


def processor_requires_medias(processor) -> bool:
    try:
        params = inspect.signature(processor).parameters
        return "medias" in params and "text" in params
    except (TypeError, ValueError):
        return hasattr(processor, "media_processor")


def call_processor(processor, text, multimodal_inputs: dict | None = None):
    multimodal_inputs = multimodal_inputs or {}

    # for kimi-vl & kimi-2.5
    if processor_requires_medias(processor):
        medias = []
        if images := multimodal_inputs.get("images"):
            medias.extend({"type": "image", "image": image} for image in images)
        if videos := multimodal_inputs.get("videos"):
            medias.extend({"type": "video", "video": video} for video in videos)
        return processor(text=text, medias=medias)

    kwargs = build_processor_kwargs(multimodal_inputs)
    return processor(text=text, **kwargs)


def extract_multimodal_train_inputs(processor_output: Mapping[str, Any]) -> dict[str, Any] | None:
    excluded_keys = {"input_ids", "attention_mask", "mm_token_type_ids"}
    return {key: value for key, value in processor_output.items() if key not in excluded_keys} or None


def load_processor(name_or_path: str, **kwargs):
    from miles.utils.chat_template_utils.inkling import is_inkling_checkpoint

    if is_inkling_checkpoint(name_or_path):
        from miles_plugins.models.inkling.mm_processor import InklingTrainProcessor

        return InklingTrainProcessor(name_or_path)

    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    return proc


def process_vision_info(prompt, processor):
    if hasattr(processor, "extract_media"):
        return processor.extract_media(prompt)

    # TODO: temporary solution, will write image utils for miles later
    from qwen_vl_utils import process_vision_info as qwen_process_vision_info

    if hasattr(processor.image_processor, "patch_size"):
        image_patch_size = processor.image_processor.patch_size
    else:
        logger.info(f"Using default patch size: {DEFAULT_PATCH_SIZE}")
        image_patch_size = DEFAULT_PATCH_SIZE
    images, videos = qwen_process_vision_info(prompt, image_patch_size=image_patch_size)
    multimodal_inputs = {"images": images, "videos": videos}
    return multimodal_inputs


_DEFERRED_MEDIA_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_DEFERRED_MEDIA_CACHE_MAX = int(os.getenv("MILES_MEDIA_CACHE_ENTRIES", "16"))
_DEFERRED_MEDIA_LOCK = threading.Lock()


def extract_deferrable_media_refs(prompt, processor) -> dict | None:
    """Return {media_type: [ref, ...]} when every media item is cheap to reopen.

    "Cheap" means a local file: `file://...` or a bare path, which can be
    resolved at use time for the price of an Image.open. Decoding when the
    Dataset is built instead costs memory proportional to the corpus -- a decoded
    RGB page is ~5.75 MB, so a 51k-page dataset pins ~295 GB for the whole run --
    even though the result is only read inside sglang_rollout.generate(), for one
    sample, and is dead afterwards.

    Inline base64 and remote URLs are NOT cheap to re-resolve, so this returns
    None for them and the caller decodes eagerly, exactly as before. The same
    bail-out covers a processor with its own ``extract_media`` (it may consume
    modalities or context this function does not model) and any content part
    type other than text/image/video, so anything this function cannot
    faithfully re-resolve takes the eager path unchanged. Returns None rather
    than {} when there is no media, so the caller can distinguish "defer this"
    from "nothing to defer".
    """
    if not isinstance(prompt, list):
        return None
    if hasattr(processor, "extract_media"):
        return None
    refs: dict[str, list] = {}
    found = False
    for message in prompt:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "text":
                continue
            if kind not in ("image", "video"):
                # image_url, audio, anything unknown: not modelled here, so do
                # not defer -- a partial refs dict would silently drop media.
                return None
            value = part.get(kind)
            found = True
            if not isinstance(value, str):
                return None
            # A data: URI carries the bytes inline, so "re-resolving" one means
            # holding the whole payload anyway. It also contains no "://", so it
            # must be rejected before the bare-path test below.
            if value.startswith("data:"):
                return None
            if not (value.startswith("file://") or "://" not in value):
                return None
            refs.setdefault(kind, []).append(value)
    return refs if found else None


def deferred_media_cache_key(refs: dict) -> tuple:
    """Hashable identity for a set of refs -- the same paths mean the same media."""
    return tuple((kind, tuple(items)) for kind, items in sorted(refs.items()))


def resolve_deferred_media(refs: dict, processor) -> dict:
    """Turn refs back into media, reusing the result across a GRPO group.

    Rebuilds the minimal conversation process_vision_info expects, so the output
    is identical to what the eager path would have produced: same fetch_image,
    same to_rgb, same smart_resize.

    The n samples of a group are deepcopies of one prompt and therefore share
    refs, so without the cache this decodes the same pages n times. Entries are
    evicted oldest-first and only groups in flight can hit, so a handful suffices.
    """
    key = deferred_media_cache_key(refs)
    with _DEFERRED_MEDIA_LOCK:
        hit = _DEFERRED_MEDIA_CACHE.get(key)
        if hit is not None:
            _DEFERRED_MEDIA_CACHE.move_to_end(key)
            return hit

    content = [{"type": kind, kind: ref} for kind, items in refs.items() for ref in items]
    resolved = process_vision_info([{"role": "user", "content": content}], processor)

    with _DEFERRED_MEDIA_LOCK:
        _DEFERRED_MEDIA_CACHE[key] = resolved
        _DEFERRED_MEDIA_CACHE.move_to_end(key)
        while len(_DEFERRED_MEDIA_CACHE) > _DEFERRED_MEDIA_CACHE_MAX:
            _DEFERRED_MEDIA_CACHE.popitem(last=False)
    return resolved


_PROCESSOR_OUTPUT_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_PROCESSOR_OUTPUT_LOCK = threading.Lock()


def call_processor_cached(processor, text, multimodal_inputs, cache_key):
    """call_processor, memoised on (prompt text, media identity).

    pixel_values depends only on the prompt and the images, and every sample in a
    GRPO group has both identical -- so this runs once per group instead of once
    per sample. That removes the redundant processor passes and, because the
    result is shared, shrinks what Ray has to serialise to the trainer by the
    same factor (measured: ~23 GB -> ~2.9 GB per step, a 24s stall in
    torch._legacy_save).
    """
    key = (text if isinstance(text, str) else str(text), cache_key)
    with _PROCESSOR_OUTPUT_LOCK:
        hit = _PROCESSOR_OUTPUT_CACHE.get(key)
        if hit is not None:
            _PROCESSOR_OUTPUT_CACHE.move_to_end(key)
            return hit

    out = call_processor(processor, text, multimodal_inputs)

    with _PROCESSOR_OUTPUT_LOCK:
        _PROCESSOR_OUTPUT_CACHE[key] = out
        _PROCESSOR_OUTPUT_CACHE.move_to_end(key)
        while len(_PROCESSOR_OUTPUT_CACHE) > _DEFERRED_MEDIA_CACHE_MAX:
            _PROCESSOR_OUTPUT_CACHE.popitem(last=False)
    return out


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"
