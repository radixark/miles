"""Images on the TITO session path.

The chat template renders one placeholder per image (``<|image_pad|>`` for
Qwen3-VL, ``<|image|>`` between its markers for GLM-V; the family's
``ImageTokens`` entry in ``tito_tokenizer`` names it), which is what the
engine wants: SGLang expands that placeholder into the image's token count
in place and keeps every other id verbatim (``SGLANG_MM_AVOID_RETOKENIZE``).
The trainer wants the expanded form plus the images, so the session server
keeps three things per image and reconciles them when samples are built:

* the PNG bytes, stored once per distinct image under a content hash and
  referenced from the stored messages as ``mm://sha256:<hex>`` (a request
  that replays 40 screenshots per turn would otherwise copy 40 data URLs into
  every record);
* the image processor's grid (``image_grid_thw``) and token count, computed
  once with the same processor the model runs;
* the expansion of each placeholder at sample time, checked against the
  engine's reported ``prompt_tokens``.

The training sample ships the images as PNG bytes
(``multimodal_train_inputs = {"image_png", "image_png_lengths",
"image_grid_thw"}``); the trainer rebuilds pixel tensors per micro-batch
from them.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

MEDIA_URL_PREFIX = "mm://sha256:"
_IMAGE_PART_TYPES = ("image_url", "input_image")


@dataclass(frozen=True)
class MediaEntry:
    """One image as the trainer and the engine see it."""

    png: bytes
    grid_thw: tuple[int, int, int]
    num_tokens: int
    data_url: str  # what the engine receives; the client's exact bytes

    @property
    def key(self) -> str:
        return hashlib.sha256(self.png).hexdigest()


def _decode_data_url(url: str) -> tuple[str, bytes]:
    if not url.startswith("data:"):
        raise ValueError(f"session images must be data URLs, got {url[:32]!r}")
    header, _, payload = url.partition(",")
    if ";base64" not in header:
        raise ValueError("session image data URLs must be base64")
    try:
        return header[len("data:") : header.index(";")], base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"session image data URL is not valid base64: {e}") from e


def _as_png(media_type: str, data: bytes) -> tuple[bytes, Any]:
    """The image as PNG bytes and as a PIL image; a non-PNG is re-encoded.
    Bytes PIL cannot read are a client error (ValueError), not a server one."""
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (OSError, ValueError, SyntaxError) as e:  # UnidentifiedImageError is an OSError
        raise ValueError(f"session image bytes are not a readable image: {e}") from e
    if media_type == "image/png":
        return data, image
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue(), image


class MediaStore:
    """Content-addressed images of one session, with the processor's count and grid."""

    def __init__(
        self, processor: Any, image_pad_id: int | None, *, expands: bool = True, family: str | None = None
    ) -> None:
        self.processor = processor
        self.image_pad_id = image_pad_id
        #: The engine expands one placeholder into the image's token count
        #: (``ImageTokens.expands``); the samples expand the record the same way.
        self.expands = expands
        self.family = family
        self.entries: dict[str, MediaEntry] = {}

    @property
    def enabled(self) -> bool:
        return self.processor is not None and self.image_pad_id is not None

    def __eq__(self, other: object) -> bool:
        # Two stores are the same session state when they hold the same images under
        # the same tokens (a snapshot of the session compares its deep copy with it).
        if not isinstance(other, MediaStore):
            return NotImplemented
        return (
            (self.processor is None) == (other.processor is None)
            and self.image_pad_id == other.image_pad_id
            and self.expands == other.expands
            and self.family == other.family
            and self.entries == other.entries
        )

    def snapshot(self) -> MediaStore:
        """A store with the images registered so far. Entries are frozen and shared;
        only the index is copied, so an image registered later is not in the copy.
        Caller holds the session lock."""
        copy = MediaStore(self.processor, self.image_pad_id, expands=self.expands, family=self.family)
        copy.entries = dict(self.entries)
        return copy

    def register(self, url: str) -> str:
        """Store the image behind a data URL; return its ``mm://`` reference."""
        if url.startswith(MEDIA_URL_PREFIX):
            if url[len(MEDIA_URL_PREFIX) :] not in self.entries:
                raise ValueError(f"unknown media reference {url}")
            return url
        if not self.enabled:
            if self.image_pad_id is not None:
                raise ValueError("this session server cannot carry images: the model has no image processor")
            raise ValueError(
                f"this session server cannot carry images: TITO family {self.family or 'default'} has no "
                "entry in the image-token table (ImageTokens in "
                "miles.utils.chat_template_utils.tito_tokenizer); add the family's begin marker, "
                "placeholder and end marker there"
            )
        media_type, data = _decode_data_url(url)
        if not media_type.startswith("image/"):
            raise ValueError(f"unsupported media type {media_type!r}; only images are carried")
        png, image = _as_png(media_type, data)
        key = hashlib.sha256(png).hexdigest()
        if key not in self.entries:
            grid, count = image_grid_and_count(self.processor, image)
            self.entries[key] = MediaEntry(png=png, grid_thw=grid, num_tokens=count, data_url=url)
        return MEDIA_URL_PREFIX + key

    def get(self, reference: str) -> MediaEntry:
        if not reference.startswith(MEDIA_URL_PREFIX):
            raise ValueError(f"not a media reference: {reference[:32]!r}")
        return self.entries[reference[len(MEDIA_URL_PREFIX) :]]


def image_grid_and_count(processor: Any, image: Any) -> tuple[tuple[int, int, int], int]:
    """The image processor's grid and the number of tokens the image occupies
    after expansion, from the processor itself (``_get_num_multimodal_tokens``
    is the transformers convention SGLang's exact-token path also uses)."""
    output = processor.image_processor(images=[image], return_tensors="np")
    grid = tuple(int(v) for v in np.asarray(output["image_grid_thw"])[0].tolist())
    counts = processor._get_num_multimodal_tokens(image_sizes=[(image.height, image.width)]).num_image_tokens
    return grid, int(counts[0])


def _image_url_of(part: dict[str, Any]) -> str | None:
    if not isinstance(part, dict) or part.get("type") not in _IMAGE_PART_TYPES:
        return None
    image = part.get("image_url")
    if isinstance(image, str):
        return image
    if isinstance(image, dict):
        return image.get("url")
    return None


def _with_image_url(part: dict[str, Any], url: str) -> dict[str, Any]:
    part = deepcopy(part)
    if isinstance(part.get("image_url"), dict):
        part["image_url"] = {**part["image_url"], "url": url}
    else:
        part["image_url"] = url
    return part


def normalize_messages(messages: list[dict[str, Any]], store: MediaStore) -> list[dict[str, Any]]:
    """The messages with every image data URL replaced by its ``mm://``
    reference, registering new images in ``store``. Messages without images
    are returned as they are."""
    out = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list) or not any(_image_url_of(p) for p in content):
            out.append(message)
            continue
        parts = []
        for part in content:
            url = _image_url_of(part)
            parts.append(part if url is None else _with_image_url(part, store.register(url)))
        out.append({**message, "content": parts})
    return out


def restore_messages(messages: list[dict[str, Any]], store: MediaStore) -> list[dict[str, Any]]:
    """The inverse of ``normalize_messages`` for the engine request: every
    ``mm://`` reference becomes the client's data URL again."""
    out = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        parts = []
        for part in content:
            url = _image_url_of(part)
            if url is not None and url.startswith(MEDIA_URL_PREFIX):
                part = _with_image_url(part, store.get(url).data_url)
            parts.append(part)
        out.append({**message, "content": parts})
    return out


def media_references(messages: list[dict[str, Any]]) -> list[str]:
    """Every image reference in prompt order."""
    refs = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            url = _image_url_of(part)
            if url is not None:
                refs.append(url)
    return refs


def expand_image_placeholders(token_ids: list[int], counts: list[int], pad_id: int) -> list[int]:
    """``token_ids`` with the i-th image placeholder repeated ``counts[i]``
    times: the same expansion the engine applies to the request ids. Raises
    when the placeholder count and the image count differ."""
    positions = [i for i, t in enumerate(token_ids) if t == pad_id]
    if len(positions) != len(counts):
        raise ValueError(
            f"{len(positions)} image placeholder(s) in the prompt but {len(counts)} image(s) in the messages"
        )
    out: list[int] = []
    cursor = 0
    for position, count in zip(positions, counts, strict=True):
        out.extend(token_ids[cursor:position])
        out.extend([pad_id] * count)
        cursor = position + 1
    out.extend(token_ids[cursor:])
    return out


def train_inputs(entries: list[MediaEntry]) -> dict[str, np.ndarray]:
    """The images of one sample in the trainer's PNG format."""
    pngs = [entry.png for entry in entries]
    return {
        "image_png": np.frombuffer(b"".join(pngs), dtype=np.uint8).copy(),
        "image_png_lengths": np.asarray([len(png) for png in pngs], dtype=np.int64),
        "image_grid_thw": np.asarray([entry.grid_thw for entry in entries], dtype=np.int64).reshape(-1, 3),
    }
