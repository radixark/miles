"""Media references: how a dataset's media reaches SGLang as shared paths.

    chat messages ──resolve_media_refs──▶ {"images": [p0, p1], "audios": [p2], "videos": [p3]}
         │                                           │                        │
         │ path / file:// URL ─▶ as is               ├─rollout_media_payload─▶ {"image_data": [...], ...}  (SGLang)
         │ data URL / bytes / PIL ─▶ written         └─load_media_for_processor─▶ PIL images, audio bytes   (Miles)
         ▼   once under media_dir by content hash                              videos rejected

The tests pin the order and duplicates of the references, that non-path sources are written once by content
hash, the rejections, and that the Miles-side processor sees the original pixels.
"""

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from miles.utils.media_refs import (
    has_media,
    load_media_for_processor,
    resolve_media_ref,
    resolve_media_refs,
    rollout_media_payload,
)


def _png_bytes(color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_refs_keep_conversation_order_across_messages_and_modalities(tmp_path):
    paths = {}
    for name in ("a.png", "b.png", "c.wav"):
        paths[name] = tmp_path / name
        paths[name].write_bytes(name.encode())
    messages = [
        {"role": "system", "content": "plain text is skipped"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(paths["b.png"])},
                {"type": "audio", "audio": str(paths["c.wav"])},
                {"type": "text", "text": "and"},
                {"type": "image_url", "image_url": {"url": f"file://{paths['a.png']}"}},
            ],
        },
        {"role": "user", "content": [{"type": "input_image", "image": str(paths["b.png"])}]},
    ]

    multimodal_inputs = resolve_media_refs(messages, media_dir=None)

    # A repeated image is a second reference: one reference per placeholder, never deduplicated.
    assert multimodal_inputs == {
        "images": [str(paths["b.png"]), str(paths["a.png"]), str(paths["b.png"])],
        "audios": [str(paths["c.wav"])],
    }
    assert rollout_media_payload(multimodal_inputs) == {
        "image_data": multimodal_inputs["images"],
        "audio_data": multimodal_inputs["audios"],
    }


def test_text_only_messages_have_no_media():
    multimodal_inputs = resolve_media_refs(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], media_dir=None
    )

    assert multimodal_inputs is None
    assert not has_media(multimodal_inputs)
    assert rollout_media_payload(multimodal_inputs) == {}


@pytest.mark.parametrize(
    "make_source",
    [
        lambda data: data,
        lambda data: {"bytes": data, "path": "ignored.png"},
        lambda data: "data:image/png;base64," + base64.b64encode(data).decode(),
        lambda data: Image.open(io.BytesIO(data)),
    ],
    ids=["bytes", "hf_image_feature", "data_url", "pil"],
)
def test_non_path_sources_are_written_once_by_content_hash(tmp_path, make_source):
    data = _png_bytes("red")

    first = resolve_media_ref(make_source(data), media_dir=str(tmp_path))
    second = resolve_media_ref(make_source(data), media_dir=str(tmp_path))

    assert first == second
    assert Path(first).is_absolute() and Path(first).is_relative_to(tmp_path)
    assert Image.open(first).getpixel((0, 0)) == (255, 0, 0)


def test_non_path_source_without_media_dir_is_rejected():
    with pytest.raises(ValueError, match="--shared-media-dir"):
        resolve_media_ref(_png_bytes("red"), media_dir=None)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_media_ref(str(tmp_path / "missing.png"), media_dir=None)


def test_video_refs_reach_the_engine_payload_but_not_the_miles_processor(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mp4")

    multimodal_inputs = resolve_media_refs(
        [{"role": "user", "content": [{"type": "video", "video": str(video)}]}], media_dir=None
    )

    assert rollout_media_payload(multimodal_inputs) == {"video_data": [str(video)]}
    with pytest.raises(NotImplementedError, match="Video"):
        load_media_for_processor(multimodal_inputs)


def test_processor_loads_original_pixels_without_resizing(tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGBA", (37, 23), color=(1, 2, 3, 4)).save(path)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio-bytes")

    loaded = load_media_for_processor({"images": [str(path)], "audios": [str(audio)]})

    (image,) = loaded["images"]
    assert image.size == (37, 23) and image.mode == "RGBA"
    assert loaded["audios"] == [b"audio-bytes"]
