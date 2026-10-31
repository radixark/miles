"""Images on the TITO session path (miles.rollout.session.media): the store,
the mm:// normalization of stored messages, the restored engine body, the
placeholder expansion and the trainer's PNG format."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from miles.rollout.session.media import (
    MEDIA_URL_PREFIX,
    MediaStore,
    expand_image_placeholders,
    media_references,
    normalize_messages,
    restore_messages,
    train_inputs,
)

PAD = 151655


class FakeProcessor:
    """Counts like a Qwen vision processor: 16 px patches merged 2x2, so a
    64x64 image is a 1x4x4 grid of 16 patches and 4 tokens."""

    def __init__(self):
        self.image_processor = self

    def __call__(self, images, return_tensors="np"):
        grids = [(1, image.height // 16, image.width // 16) for image in images]
        return {"image_grid_thw": np.asarray(grids, dtype=np.int64)}

    def _get_num_multimodal_tokens(self, image_sizes):
        counts = [(h // 16) * (w // 16) // 4 for h, w in image_sizes]
        return SimpleNamespace(num_image_tokens=counts)


def _png(width=64, height=64, color=(1, 2, 3)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _data_url(png: bytes, media_type="image/png") -> str:
    return f"data:{media_type};base64," + base64.b64encode(png).decode()


def _image_message(url: str, text="look") -> dict:
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}],
    }


def test_the_store_keeps_one_entry_per_distinct_image_with_the_processor_count():
    store = MediaStore(FakeProcessor(), PAD)
    png = _png()
    ref = store.register(_data_url(png))
    assert ref.startswith(MEDIA_URL_PREFIX)
    assert store.register(_data_url(png)) == ref, "the same bytes register once"
    entry = store.get(ref)
    assert entry.png == png and entry.grid_thw == (1, 4, 4) and entry.num_tokens == 4
    other = store.register(_data_url(_png(width=128)))
    assert other != ref and store.get(other).num_tokens == 8
    assert len(store.entries) == 2


def test_a_snapshot_keeps_the_images_registered_before_it():
    """The samples worker reads a snapshot; an image a later chat registers
    must not appear in it, and the snapshot shares the stored entries."""
    store = MediaStore(FakeProcessor(), PAD)
    first = store.register(_data_url(_png()))
    copy = store.snapshot()
    second = store.register(_data_url(_png(width=128)))
    assert list(copy.entries) == [first[len(MEDIA_URL_PREFIX) :]]
    assert len(store.entries) == 2
    assert copy.get(first) is store.get(first)
    with pytest.raises(KeyError):
        copy.get(second)


def test_a_jpeg_is_stored_as_png_and_the_engine_still_gets_the_client_bytes():
    store = MediaStore(FakeProcessor(), PAD)
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (9, 9, 9)).save(buffer, format="JPEG")
    url = _data_url(buffer.getvalue(), "image/jpeg")
    ref = store.register(url)
    entry = store.get(ref)
    assert entry.png.startswith(b"\x89PNG") and entry.data_url == url


def test_a_store_without_processor_or_placeholder_refuses_images():
    with pytest.raises(ValueError, match="cannot carry images"):
        MediaStore(None, PAD).register(_data_url(_png()))
    with pytest.raises(ValueError, match="cannot carry images"):
        MediaStore(FakeProcessor(), None).register(_data_url(_png()))
    with pytest.raises(ValueError, match="data URLs"):
        MediaStore(FakeProcessor(), PAD).register("https://example.com/a.png")


def test_normalize_and_restore_round_trip_and_leave_text_messages_alone():
    store = MediaStore(FakeProcessor(), PAD)
    url = _data_url(_png())
    text = {"role": "user", "content": "hello"}
    messages = [text, _image_message(url)]
    normalized = normalize_messages(messages, store)
    assert normalized[0] is text, "a text message is passed through by identity"
    ref = normalized[1]["content"][1]["image_url"]["url"]
    assert ref.startswith(MEDIA_URL_PREFIX) and messages[1]["content"][1]["image_url"]["url"] == url
    assert media_references(normalized) == [ref]
    assert normalize_messages(normalized, store) == normalized, "references normalize to themselves"
    restored = restore_messages(normalized, store)
    assert restored[1]["content"][1]["image_url"]["url"] == url and restored[0] is text
    assert media_references(messages) == [url]


def test_expand_image_placeholders_repeats_each_placeholder_by_its_count():
    assert expand_image_placeholders([1, PAD, 2, PAD, 3], [3, 1], PAD) == [1, PAD, PAD, PAD, 2, PAD, 3]
    assert expand_image_placeholders([1, 2], [], PAD) == [1, 2]
    with pytest.raises(ValueError, match="placeholder"):
        expand_image_placeholders([1, PAD], [3, 3], PAD)


def test_train_inputs_pack_the_pngs_with_lengths_and_grids():
    store = MediaStore(FakeProcessor(), PAD)
    first, second = _png(), _png(width=128, color=(7, 7, 7))
    entries = [store.get(store.register(_data_url(first))), store.get(store.register(_data_url(second)))]
    packed = train_inputs(entries)
    assert packed["image_png"].dtype == np.uint8 and packed["image_png"].tobytes() == first + second
    assert packed["image_png_lengths"].tolist() == [len(first), len(second)]
    assert packed["image_grid_thw"].tolist() == [[1, 4, 4], [1, 4, 8]]
    assert train_inputs([])["image_grid_thw"].shape == (0, 3)


def test_unreadable_bytes_and_bad_base64_are_client_errors():
    store = MediaStore(FakeProcessor(), PAD)
    with pytest.raises(ValueError, match="not a readable image"):
        store.register("data:image/png;base64," + base64.b64encode(b"not an image at all").decode())
    with pytest.raises(ValueError, match="not valid base64"):
        store.register("data:image/png;base64,@@@not-base64@@@")


class TestExpandedLength:
    """``expanded_length`` counts a token list the way the engine does: each
    image placeholder at the image's token count, images taken in message
    order; it is what the addition-R3 row offsets are computed in."""

    PAD = 151655

    def _store_with(self, *pngs):
        from miles.rollout.session.media import MediaStore, normalize_messages

        store = MediaStore(FakeProcessor(), self.PAD)
        messages = normalize_messages([_image_message(_data_url(png)) for png in pngs], store)
        return store, messages

    def test_placeholders_count_at_the_images_token_counts_in_message_order(self):
        from miles.rollout.session.media import expanded_length

        store, messages = self._store_with(_png(), _png(width=128))  # 4 and 8 tokens
        assert expanded_length([1, self.PAD, 2, self.PAD, 3], messages, store) == 5 - 2 + 4 + 8

    def test_a_prefix_with_fewer_placeholders_uses_the_first_images(self):
        from miles.rollout.session.media import expanded_length

        store, messages = self._store_with(_png(), _png(width=128))
        # A checkpoint that predates the second image: one placeholder, the first image's count.
        assert expanded_length([1, self.PAD, 2], messages, store) == 3 - 1 + 4

    def test_text_only_servers_and_ids_without_placeholders_count_as_they_are(self):
        from miles.rollout.session.media import MediaStore, expanded_length

        store, messages = self._store_with(_png())
        assert expanded_length([1, 2, 3], messages, store) == 3
        assert expanded_length([1, self.PAD, 3], messages, None) == 3
        flat = MediaStore(FakeProcessor(), self.PAD, expands=False)
        assert expanded_length([1, self.PAD, 3], messages, flat) == 3

    def test_more_placeholders_than_images_is_refused(self):
        from miles.rollout.session.media import expanded_length

        store, messages = self._store_with(_png())
        with pytest.raises(ValueError, match="2 image placeholder"):
            expanded_length([1, self.PAD, self.PAD], messages, store)
