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
from types import SimpleNamespace

import pytest
from PIL import Image

from miles.utils.media_refs import (
    collect_media_refs,
    has_media,
    materialize_media_ref,
    release_run_media,
    rollout_media_payload,
    run_media_dir,
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


def test_run_media_is_written_under_the_run_and_released_without_touching_shared_paths(tmp_path):
    shared = tmp_path / "dataset.png"
    shared.write_bytes(_png_bytes("red"))
    args = SimpleNamespace(shared_media_dir=str(tmp_path / "media"), run_uuid="run-1")

    written = materialize_media_ref(_png_bytes("blue"), run_media_dir(args))
    kept = materialize_media_ref(str(shared), run_media_dir(args))
    assert Path(written).is_relative_to(tmp_path / "media" / "run-1")

    release_run_media(args)
    assert not Path(written).exists()
    assert Path(kept).is_file()


def test_runs_without_a_shared_media_dir_have_no_run_media():
    args = SimpleNamespace(shared_media_dir=None, run_uuid="run-1")
    assert run_media_dir(args) is None
    release_run_media(args)
