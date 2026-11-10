"""Media references: how a dataset's media reaches SGLang as shared paths.

    chat messages ──collect_media_refs──▶ {"images": [p0, p1], "audios": [p2], "videos": [p3]}
         │                                           │
         │ path / file:// URL ─▶ as is               └─rollout_media_payload─▶ {"image_data": [...], ...}  (SGLang)
         │ data URL / bytes / PIL ─▶ written
         ▼   once under media_dir by content hash

The tests pin the order and duplicates of the references, that a path stays as written when it goes through a
symlink, that non-path sources are written once by content hash, and the rejections.
"""

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from miles.utils.media_refs import (
    collect_media_refs,
    has_media,
    has_message_media,
    materialize_media_ref,
    materialize_message_media,
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

    multimodal_inputs = collect_media_refs(messages, media_dir=None)

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
    multimodal_inputs = collect_media_refs(
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

    first = materialize_media_ref(make_source(data), media_dir=str(tmp_path))
    second = materialize_media_ref(make_source(data), media_dir=str(tmp_path))

    assert first == second
    assert Path(first).is_absolute() and Path(first).is_relative_to(tmp_path)
    assert Image.open(first).getpixel((0, 0)) == (255, 0, 0)


def test_a_path_through_a_symlink_stays_as_written(tmp_path):
    """This node may reach the shared disk through a symlink that the engine nodes do not have: a reference that
    followed it would name a path only this node can open."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "a.png").write_bytes(b"a")
    mount = tmp_path / "mount"
    mount.symlink_to(real_dir, target_is_directory=True)

    assert materialize_media_ref(str(mount / "a.png"), media_dir=None) == str(mount / "a.png")
    assert materialize_media_ref(b"b", media_dir=str(mount)).startswith(f"{mount}/")


def test_non_path_source_without_media_dir_is_rejected():
    with pytest.raises(ValueError, match="--shared-media-dir"):
        materialize_media_ref(_png_bytes("red"), media_dir=None)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        materialize_media_ref(str(tmp_path / "missing.png"), media_dir=None)


def test_remote_url_is_rejected_with_the_reason():
    """A URL is not a file, and 'file not found' would send the user looking for a path that never existed."""
    with pytest.raises(ValueError, match="Remote media URLs are not supported"):
        materialize_media_ref("https://example.com/a.png", media_dir=None)


def test_per_part_resize_options_are_rejected(tmp_path):
    """A reference keeps only the source: dropping min_pixels/max_pixels silently would change the pixels."""
    image = tmp_path / "a.png"
    image.write_bytes(_png_bytes("red"))
    messages = [{"role": "user", "content": [{"type": "image", "image": str(image), "max_pixels": 1024}]}]

    with pytest.raises(ValueError, match="max_pixels"):
        collect_media_refs(messages, media_dir=None)


def test_video_refs_reach_the_engine_payload(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mp4")

    multimodal_inputs = collect_media_refs(
        [{"role": "user", "content": [{"type": "video", "video": str(video)}]}], media_dir=None
    )

    assert rollout_media_payload(multimodal_inputs) == {"video_data": [str(video)]}


def test_message_media_become_shared_paths_and_text_messages_stay_as_they_are(tmp_path):
    """The session server stores a conversation's media as shared paths: the same bytes always get the
    same path, so a conversation that resends its screenshots every turn resends the same references."""
    data_url = "data:image/png;base64," + base64.b64encode(_png_bytes("red")).decode()
    text = {"role": "system", "content": "be brief"}
    image_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            {"type": "image", "image": data_url},
        ],
    }
    messages = [text, image_message]
    assert has_message_media(messages) and not has_message_media([text])

    out = materialize_message_media(messages, str(tmp_path))
    assert out[0] is text
    part_url, part_image = out[1]["content"][1:]
    assert part_url["image_url"]["detail"] == "high"
    assert part_url["image_url"]["url"] == part_image["image"]
    assert Path(part_image["image"]).read_bytes() == _png_bytes("red")
    assert image_message["content"][1]["image_url"]["url"] == data_url, "the input is not modified"
    assert materialize_message_media(out, str(tmp_path)) == out, "a stored path is its own reference"
    assert collect_media_refs(out, None) == {"images": [part_image["image"]] * 2}


def test_message_media_options_are_refused_before_anything_is_stored():
    data_url = "data:image/png;base64," + base64.b64encode(_png_bytes("red")).decode()
    part = {"type": "image_url", "image_url": {"url": data_url}, "max_pixels": 1024}
    with pytest.raises(ValueError, match="max_pixels"):
        materialize_message_media([{"role": "user", "content": [part]}], None)


def test_message_media_without_a_shared_dir_are_refused():
    data_url = "data:image/png;base64," + base64.b64encode(_png_bytes("red")).decode()
    with pytest.raises(ValueError, match="--shared-media-dir"):
        materialize_message_media([{"role": "user", "content": [{"type": "image_url", "image_url": data_url}]}], None)
