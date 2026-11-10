"""Media references shared between Miles and the rollout engine.

A media reference is an absolute path on a filesystem that Miles and every SGLang
engine mount. In token space ``Sample.multimodal_inputs`` holds one ordered list of
references per modality for the whole conversation, requests carry these paths instead
of the media bytes, and the trainer rebuilds the processor tensors from the same files.
The Miles-side processor path keeps its own media handling and never sees a reference.

Sources that are not already shared paths (data URLs, raw bytes, PIL images) are
written once under ``media_dir``, named by content hash.
"""

import base64
import hashlib
import io
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from miles.utils.file_utils import atomic_write_bytes

# Content part type -> (multimodal_inputs key, part fields that may hold the source).
_IMAGE_PART = ("images", ("image", "image_url"))
_AUDIO_PART = ("audios", ("audio", "audio_url", "input_audio"))
_VIDEO_PART = ("videos", ("video", "video_url"))
_PART_TYPES = {
    "image": _IMAGE_PART,
    "image_url": _IMAGE_PART,
    "input_image": _IMAGE_PART,
    "audio": _AUDIO_PART,
    "audio_url": _AUDIO_PART,
    "input_audio": _AUDIO_PART,
    "video": _VIDEO_PART,
    "video_url": _VIDEO_PART,
}
# multimodal_inputs key -> the SGLang GenerateReqInput field that carries those references.
_TO_SGLANG_MEDIA_FIELDS = {"images": "image_data", "audios": "audio_data", "videos": "video_data"}
# qwen-vl-utils per-part options; a reference carries only the source, so they would be dropped without notice.
_PER_PART_OPTIONS = frozenset(
    {"resized_height", "resized_width", "min_pixels", "max_pixels", "total_pixels", "fps", "nframes", "max_frames"}
)


def has_media(multimodal_inputs: dict[str, list[str]] | None) -> bool:
    return bool(multimodal_inputs) and any(multimodal_inputs.values())


def collect_media_refs(messages: Iterable[dict[str, Any]], media_dir: str | None) -> dict[str, list[str]] | None:
    """Collect every media part of ``messages`` as a shared-path reference, in conversation order.

    Args:
        messages: Chat messages whose list contents may hold image, audio or video parts.
        media_dir: Shared directory for sources that are not shared paths yet, or None.

    Returns:
        The references per modality in conversation order, or None for text-only messages.
    """
    multimodal_inputs: dict[str, list[str]] = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else None
            if part_type not in _PART_TYPES:
                continue
            key, fields = _PART_TYPES[part_type]
            if options := _PER_PART_OPTIONS.intersection(part):
                raise ValueError(
                    f"Media part options {sorted(options)} are not supported: a media reference carries only its "
                    "source, so set them for every media in the processor configuration (--sglang-mm-process-config)."
                )
            source = next((part[field] for field in fields if part.get(field) is not None), None)
            if isinstance(source, dict) and "url" in source:
                source = source["url"]
            multimodal_inputs.setdefault(key, []).append(materialize_media_ref(source, media_dir))
    return multimodal_inputs or None


def materialize_media_ref(source: Any, media_dir: str | None) -> str:
    """Materialize one media source as an absolute shared path; one not yet a file is written under ``media_dir``.

    Args:
        source: A path or file:// URL, a data URL, raw bytes, a PIL image, or an HF {"bytes", "path"} feature.
        media_dir: Shared directory where non-path sources are written, or None to reject them.

    Returns:
        The absolute path of the media on the shared filesystem.
    """
    if isinstance(source, dict):
        # A parquet HF Image/Audio feature arrives as {"bytes", "path"}; the bytes are authoritative.
        source = source.get("bytes") or source.get("path")
    if isinstance(source, Image.Image):
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")
        return _dump_media_by_hash(buffer.getvalue(), media_dir)
    if isinstance(source, (bytes, bytearray)):
        return _dump_media_by_hash(bytes(source), media_dir)
    if not isinstance(source, str):
        raise TypeError(f"Unsupported media source type: {type(source).__name__}")
    if source.startswith("data:"):
        return _dump_media_by_hash(base64.b64decode(source.split(",", 1)[1]), media_dir)
    if source.startswith(("http://", "https://")):
        raise ValueError(
            f"Remote media URLs are not supported: {source}; download the media to a filesystem shared with every "
            "SGLang engine and reference its path."
        )
    # Not resolve(): engines and trainer ranks mount the path as written, not the target of a symlink on this node.
    path = Path(os.path.abspath(source.removeprefix("file://")))
    if not path.is_file():
        raise FileNotFoundError(f"Media file not found: {source}")
    return str(path)


def rollout_media_payload(multimodal_inputs: dict[str, list[str]] | None) -> dict[str, list[str]]:
    """Request fields carrying every media reference of the conversation, in order."""
    return {_TO_SGLANG_MEDIA_FIELDS[key]: refs for key, refs in (multimodal_inputs or {}).items() if refs}


def _dump_media_by_hash(data: bytes, media_dir: str | None) -> str:
    """Dump media bytes under ``media_dir``, named by their sha256 and written once, and return the path."""
    if media_dir is None:
        raise ValueError(
            "This media source is not a shared file path; set --shared-media-dir to a directory "
            "shared with every SGLang engine so Miles can write it there."
        )
    digest = hashlib.sha256(data).hexdigest()
    path = Path(os.path.abspath(media_dir)) / digest[:2] / digest
    if not path.is_file():
        os.makedirs(path.parent, exist_ok=True)
        atomic_write_bytes(path, data)
    return str(path)
