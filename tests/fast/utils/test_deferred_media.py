from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

from PIL import Image

from miles.utils.processing_utils import (
    call_processor_cached,
    deferred_media_cache_key,
    extract_deferrable_media_refs,
    resolve_deferred_media,
)


class _QwenStyleProcessor:
    """No extract_media, patch-size image_processor: takes the qwen_vl_utils path."""

    image_processor = SimpleNamespace(patch_size=14)


def _prompt_with(*parts):
    return [{"role": "user", "content": list(parts)}]


def _image_part(value):
    return {"type": "image", "image": value}


def test_defers_local_paths_and_file_uris():
    prompt = _prompt_with(
        {"type": "text", "text": "describe"},
        _image_part("/data/page_1.png"),
        _image_part("file:///data/page_2.png"),
    )
    refs = extract_deferrable_media_refs(prompt, _QwenStyleProcessor())
    assert refs == {"image": ["/data/page_1.png", "file:///data/page_2.png"]}


def test_no_media_returns_none_not_empty_dict():
    prompt = _prompt_with({"type": "text", "text": "just text"})
    assert extract_deferrable_media_refs(prompt, _QwenStyleProcessor()) is None
    assert extract_deferrable_media_refs("a plain string prompt", _QwenStyleProcessor()) is None


def test_inline_and_remote_media_decode_eagerly():
    for value in (
        "data:image/png;base64,iVBORw0KGgo=",
        "https://example.com/page.png",
        Image.new("RGB", (4, 4)),  # already-decoded object, nothing to reopen
    ):
        prompt = _prompt_with(_image_part(value))
        assert extract_deferrable_media_refs(prompt, _QwenStyleProcessor()) is None


def test_unmodelled_part_types_decode_eagerly():
    # A partial refs dict would silently drop the unmodelled media, so any
    # image_url / audio / unknown part must send the whole prompt eager.
    for part in (
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
        {"type": "audio", "audio": "/data/clip.wav"},
    ):
        prompt = _prompt_with(_image_part("/data/page_1.png"), part)
        assert extract_deferrable_media_refs(prompt, _QwenStyleProcessor()) is None


def test_extract_media_processor_decodes_eagerly():
    class _ExtractMediaProcessor:
        def extract_media(self, prompt):
            raise AssertionError("must not be reached from the extractor")

    prompt = _prompt_with(_image_part("/data/page_1.png"))
    assert extract_deferrable_media_refs(prompt, _ExtractMediaProcessor()) is None


def test_resolve_matches_eager_output_and_caches_per_media_identity(tmp_path):
    from miles.utils.processing_utils import process_vision_info

    path = tmp_path / "page.png"
    Image.new("RGB", (64, 64), color=(200, 10, 10)).save(path)

    prompt = _prompt_with({"type": "text", "text": "describe"}, _image_part(str(path)))
    processor = _QwenStyleProcessor()
    refs = extract_deferrable_media_refs(prompt, processor)

    resolved = resolve_deferred_media(refs, processor)
    eager = process_vision_info(prompt, processor)
    assert len(resolved["images"]) == len(eager["images"]) == 1
    assert resolved["images"][0].size == eager["images"][0].size

    # The n samples of a GRPO group share refs; the resolve must be shared too.
    assert resolve_deferred_media(refs, processor) is resolved
    assert resolve_deferred_media(dict(refs), processor) is resolved


def test_call_processor_cached_runs_once_per_group():
    calls = []

    def processor(text, **kwargs):
        calls.append(text)
        return {"input_ids": [[1, 2, 3]], "pixel_values": object()}

    refs = {"image": ["/data/page_1.png"]}
    key = deferred_media_cache_key(refs)
    inputs = {"images": [object()]}

    first = call_processor_cached(processor, "prompt-text", inputs, key)
    second = call_processor_cached(processor, "prompt-text", inputs, key)
    assert first is second
    assert len(calls) == 1

    call_processor_cached(processor, "other-prompt", inputs, key)
    assert len(calls) == 2
