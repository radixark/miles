from collections.abc import Sequence
from typing import Any

from PIL import Image

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.errors import UpstreamResponseError
from miles.rollout.session.token_space import MM_TOKEN_EXPANSION_START_LEN
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType
from miles.utils.media_refs import collect_media_refs
from miles.utils.processing_utils import load_processor

COUNTED_TITO_MODELS = frozenset({TITOTokenizerType.QWEN3_VL.value})


class ImageTokenCounter:
    def __init__(self, processor: Any, placeholder_id: int) -> None:
        self.processor = processor
        self.placeholder_id = placeholder_id
        self._counts: dict[str, int] = {}

    def count(self, path: str) -> int:
        cached = self._counts.get(path)
        if cached is not None:
            return cached
        with Image.open(path) as handle:
            size = (handle.height, handle.width)
        count = int(self.processor._get_num_multimodal_tokens(image_sizes=[size]).num_image_tokens[0])
        if count <= 0:
            raise ValueError(f"{path} ({size[1]}x{size[0]}) expands into {count} tokens")
        self._counts[path] = count
        return count


def build_image_token_counter(config: SessionServerConfig, tokenizer: Any) -> ImageTokenCounter | None:
    if not config.multimodal_token_space_processing or config.tito_model not in COUNTED_TITO_MODELS:
        return None
    processor = load_processor(config.hf_checkpoint, trust_remote_code=True)
    if processor is None:
        raise ValueError(
            f"--tito-model {config.tito_model} needs an image processor, but {config.hf_checkpoint!r} has none"
        )
    placeholder_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    if placeholder_id is None or placeholder_id == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer does not know the image placeholder {processor.image_token!r}")
    return ImageTokenCounter(processor, int(placeholder_id))


def expand_input_ids(ids: Sequence[int], counts: Sequence[int], placeholder_id: int) -> list[int]:
    found = sum(1 for token_id in ids if token_id == placeholder_id)
    if found != len(counts):
        raise ValueError(f"prompt has {found} image placeholder token(s) but {len(counts)} image(s) were provided")
    expanded: list[int] = []
    next_image = 0
    for token_id in ids:
        if token_id == placeholder_id:
            expanded.extend([placeholder_id] * counts[next_image])
            next_image += 1
        else:
            expanded.append(token_id)
    return expanded


def expected_engine_prompt(request_body: dict[str, Any], counter: ImageTokenCounter) -> list[int]:
    start = request_body[MM_TOKEN_EXPANSION_START_LEN]
    ids = request_body["input_ids"]
    new_placeholders = ids[start:].count(counter.placeholder_id)
    images = (collect_media_refs(request_body["messages"], None) or {}).get("images", [])
    if new_placeholders > len(images):
        raise UpstreamResponseError(
            f"the request expands {new_placeholders} image placeholder(s) but its messages carry {len(images)} image(s)"
        )
    new_images = images[len(images) - new_placeholders :]
    counts = [counter.count(path) for path in new_images]
    return ids[:start] + expand_input_ids(ids[start:], counts, counter.placeholder_id)


def assert_engine_expansion_matches(
    request_body: dict[str, Any], choice: dict[str, Any], counter: ImageTokenCounter
) -> None:
    if MM_TOKEN_EXPANSION_START_LEN not in request_body:
        return
    effective = choice["prompt_token_ids"]
    expected = expected_engine_prompt(request_body, counter)
    if effective == expected:
        return
    first_diff = next(
        (i for i, (a, b) in enumerate(zip(effective, expected, strict=False)) if a != b),
        min(len(effective), len(expected)),
    )
    raise UpstreamResponseError(
        f"the engine's expanded prompt differs from the session server's: engine has {len(effective)} tokens, "
        f"expected {len(expected)} (first difference at index {first_diff}); the engine's image processor "
        "disagrees with --hf-checkpoint's, or the images were dropped on the way"
    )
