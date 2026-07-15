from __future__ import annotations

import json
import os
from typing import Any

import torch

from miles_plugins.models.inkling.ops.image_processing import InklingImagePatchifier

IMAGE_TOKEN_ID = -101
AUDIO_TOKEN_ID = -102
ROLE_MESSAGE_TOKENS = {
    "user": "<|message_user|>",
    "assistant": "<|message_model|>",
    "system": "<|message_system|>",
    "tool": "<|message_tool|>",
}
MESSAGE_MODEL = "<|message_model|>"
CONTENT_TEXT = "<|content_text|>"
CONTENT_THINKING = "<|content_thinking|>"
CONTENT_XML = "<|content_xml|>"
CONTENT_IMAGE = "<|content_image|>"
CONTENT_AUDIO_INPUT = "<|content_audio_input|>"
END_MESSAGE = "<|end_message|>"
AUDIO_END = "<|audio_end|>"

_IMAGE_PART_TYPES = frozenset({"image", "input_image", "image_url"})
_AUDIO_PART_TYPES = frozenset({"audio", "input_audio", "audio_url"})


class _TokenizerAdapter:
    """encode_special/encode_text over the HF o200k+overlay tokenizer."""

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self._special_cache: dict[str, int] = {}

    def encode_special(self, literal: str) -> int:
        tid = self._special_cache.get(literal)
        if tid is None:
            ids = self._tok.encode(literal, add_special_tokens=False)
            assert len(ids) == 1, f"Inkling special {literal!r} must encode to one id, got {ids}"
            tid = self._special_cache[literal] = ids[0]
        return tid

    def encode_text(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)


def _iter_render_parts(content: Any):
    """Yield (kind, text) per content part: kind in {text, image, audio}."""
    if content is None:
        return
    if isinstance(content, str):
        if content:
            yield ("text", content)
        return
    if not isinstance(content, (list, tuple)):
        raise TypeError("message content must be a string or a sequence of parts")
    for part in content:
        if isinstance(part, str):
            yield ("text", part)
            continue
        if not isinstance(part, dict):
            raise TypeError(f"content part must be a dict, got {type(part).__name__}")
        ptype = part.get("type")
        if ptype in (None, "text", "input_text"):
            text = part.get("text", "")
            yield ("text", text if isinstance(text, str) else "")
        elif ptype in _IMAGE_PART_TYPES:
            yield ("image", "")
        elif ptype in _AUDIO_PART_TYPES:
            yield ("audio", "")
        else:
            raise ValueError(f"unsupported content part type: {ptype!r}")


def _append_message(
    input_ids: list[int],
    tok: _TokenizerAdapter,
    role: str,
    kind: str,
    text: str,
    *,
    author_name: str | None = None,
) -> None:
    input_ids.append(tok.encode_special(ROLE_MESSAGE_TOKENS[role]))
    if author_name:
        input_ids.extend(tok.encode_text(author_name))

    if kind == "text":
        input_ids.append(tok.encode_special(CONTENT_TEXT))
        input_ids.extend(tok.encode_text(text))
    elif kind == "image":
        input_ids.append(tok.encode_special(CONTENT_IMAGE))
        input_ids.append(IMAGE_TOKEN_ID)
    elif kind == "audio":
        input_ids.append(tok.encode_special(CONTENT_AUDIO_INPUT))
        input_ids.append(AUDIO_TOKEN_ID)
        input_ids.append(tok.encode_special(AUDIO_END))
    elif kind == "thinking":
        input_ids.append(tok.encode_special(CONTENT_THINKING))
        input_ids.extend(tok.encode_text(text))
    elif kind == "xml":
        input_ids.append(tok.encode_special(CONTENT_XML))
        input_ids.extend(tok.encode_text(text))
    else:
        raise ValueError(f"unsupported Inkling render part kind: {kind!r}")

    input_ids.append(tok.encode_special(END_MESSAGE))


# Named levels and the 0.99 ceiling mirror sglang serving_chat (values above
# 0.99 are out-of-distribution for the model and degrade generation).
_EFFORT_LEVELS = {"none": 0.0, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}


def _reasoning_effort_from_env() -> float | None:
    val = os.environ.get("INKLING_REASONING_EFFORT", "").strip().lower()
    if not val:
        return None
    if val in _EFFORT_LEVELS:
        return _EFFORT_LEVELS[val]
    return max(0.0, min(0.99, float(val)))


def render_inkling_messages_to_ids(
    messages: list[dict[str, Any]],
    tok: _TokenizerAdapter,
    *,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    reasoning_effort: float | None = None,
) -> list[int]:
    """Token-level Inkling chat render with one media sentinel per image/audio part."""
    input_ids: list[int] = []

    if tools:
        _append_message(
            input_ids,
            tok,
            "system",
            "xml",
            json.dumps(tools, ensure_ascii=False),
            author_name="tool_declare",
        )

    if reasoning_effort is not None:
        _append_message(
            input_ids,
            tok,
            "system",
            "text",
            f"Thinking effort level: {reasoning_effort}",
        )

    for msg in messages:
        role = msg.get("role")
        if role not in ROLE_MESSAGE_TOKENS:
            raise ValueError(f"unsupported Inkling message role {role!r}")
        if role == "assistant" and msg.get("reasoning_content"):
            rc = msg["reasoning_content"]
            if not isinstance(rc, str):
                raise TypeError("assistant reasoning_content must be a string")
            _append_message(input_ids, tok, role, "thinking", rc)
        for kind, text in _iter_render_parts(msg.get("content", "")):
            _append_message(input_ids, tok, role, kind, text)

    if add_generation_prompt:
        input_ids.append(tok.encode_special(MESSAGE_MODEL))
    return input_ids


def _load_pil_image(spec):
    """Resolve one image content part's payload to a PIL image (no resize)."""
    from PIL import Image

    if isinstance(spec, Image.Image):
        return spec
    if isinstance(spec, str):
        path = spec[len("file://") :] if spec.startswith("file://") else spec
        if path.startswith(("http://", "https://", "data:")):
            raise ValueError(f"Inkling extract_media: resolve URL images upstream, got {spec[:64]!r}")
        img = Image.open(path)
        img.load()
        return img
    raise TypeError(f"unsupported image payload type {type(spec).__name__}")


class InklingTrainProcessor:
    """Duck-typed processor for miles' generic multimodal pipeline (Inkling)."""

    def __init__(self, hf_checkpoint: str):
        from miles.utils.processing_utils import load_tokenizer

        with open(os.path.join(hf_checkpoint, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        vision_cfg = cfg.get("vision_config") or {}
        self.patchifier = InklingImagePatchifier(patch_size=vision_cfg.get("patch_size", 40))
        audio_cfg = cfg.get("audio_config") or {}
        from miles_plugins.models.inkling.ops.audio_processing import InklingAudioDmelExtractor

        # dmel grid must match what the model de-bins with (audio_config)
        self.dmel_extractor = InklingAudioDmelExtractor(
            params={
                "n_mels": audio_cfg.get("n_mel_bins", 80),
                "num_dmel_bins": audio_cfg.get("mel_vocab_size", 16),
                **{k: audio_cfg[k] for k in ("dmel_min_value", "dmel_max_value") if k in audio_cfg},
            }
        )
        self.tok = _TokenizerAdapter(load_tokenizer(hf_checkpoint, trust_remote_code=True))

    # -- dataset side (duck-type hook in miles.utils.processing_utils.process_vision_info)
    def extract_media(self, prompt: list[dict[str, Any]]) -> dict:
        images, audios = [], []
        for msg in prompt:
            content = msg.get("content", "")
            if not isinstance(content, (list, tuple)):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype in _IMAGE_PART_TYPES:
                    spec = part.get("image", part.get("image_url"))
                    if isinstance(spec, dict):
                        spec = spec.get("url")
                    images.append(_load_pil_image(spec))
                elif ptype in _AUDIO_PART_TYPES:
                    spec = part.get("audio", part.get("audio_url"))
                    if isinstance(spec, dict):
                        spec = spec.get("url")
                    # keep the raw file bytes: lossless transport to both the
                    # rollout engine and the train-side dmel extractor
                    if isinstance(spec, str):
                        path = spec[len("file://") :] if spec.startswith("file://") else spec
                        with open(path, "rb") as f:
                            spec = f.read()
                    assert isinstance(
                        spec, (bytes, bytearray)
                    ), f"audio payload must be a path or bytes, got {type(spec).__name__}"
                    audios.append(bytes(spec))
        return {"images": images or None, "audios": audios or None, "videos": None}

    # -- rollout side (call_processor)
    def __call__(self, text=None, images=None, videos=None, audios=None, **kwargs):
        assert videos is None or not videos, "Inkling processor: video inputs unsupported"
        assert isinstance(text, (list, tuple)), (
            "Inkling processor expects the raw message list as `text` (run without "
            f"--apply-chat-template so the prompt stays structured), got {type(text).__name__}"
        )
        input_ids = render_inkling_messages_to_ids(list(text), self.tok, reasoning_effort=_reasoning_effort_from_env())

        out: dict[str, Any] = {"input_ids": [input_ids]}
        n_img_sentinels = sum(1 for t in input_ids if t == IMAGE_TOKEN_ID)
        n_images = len(images) if images else 0
        assert (
            n_img_sentinels == n_images
        ), f"Inkling render produced {n_img_sentinels} image sentinel(s) but {n_images} image(s) given"
        if n_images:
            feat = self.patchifier.preprocess(list(images))
            out["mm_vision_patches"] = feat["vision_patches_bthwc"]
            out["mm_vision_num_patches"] = torch.tensor(feat["num_patches"], dtype=torch.long)

        n_aud_sentinels = sum(1 for t in input_ids if t == AUDIO_TOKEN_ID)
        n_audios = len(audios) if audios else 0
        assert (
            n_aud_sentinels == n_audios
        ), f"Inkling render produced {n_aud_sentinels} audio sentinel(s) but {n_audios} audio(s) given"
        if n_audios:
            feat = self.dmel_extractor.extract(list(audios))
            out["mm_audio_dmel"] = torch.cat(feat["dmel_bins"], dim=0)  # [sum_T, n_mels] int32
            out["mm_audio_num_tokens"] = torch.tensor(feat["num_audio_tokens"], dtype=torch.long)
        return out
