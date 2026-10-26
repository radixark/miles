import pytest
import torch
from sglang.srt.multimodal.modality import Modality

from miles.backends.training_utils.data import token_space_media_process

_IMAGE_TOKEN = 9


class _Strategy:
    mm_token_expansion_params = {"image_token_id": _IMAGE_TOKEN, "video_token_id": 8, "audio_token_id": None}

    @staticmethod
    def compute_mrope_positions(input_ids, media_features):
        return torch.arange(len(input_ids)).expand(3, -1), None


class _RecordingProcessor:
    strategy = _Strategy()

    def __init__(self):
        self.calls = []

    async def process(self, **kwargs):
        self.calls.append(kwargs)
        num_tokens = len(kwargs["input_ids"])
        return {
            "input_ids": torch.tensor([kwargs["input_ids"]]),
            "attention_mask": torch.ones(1, num_tokens),
            "pixel_values": torch.full((2, 3), 2.0),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
            "mm_modalities": [Modality.IMAGE, Modality.IMAGE],
            "mm_spans": [[(1, 3)], [(3, 5)]],
            # Two images squeeze four tokens into two mRoPE steps, so the text after them is shifted by -2.
            "mrope_positions": torch.tensor([[0, 1, 1, 2, 2]] * 3),
            "mrope_position_delta": torch.tensor([[-2]]),
        }


@pytest.fixture
def processor(monkeypatch):
    processor = _RecordingProcessor()
    monkeypatch.setattr(token_space_media_process, "_build_processor", lambda *args: processor)
    token_space_media_process._processor_future.cache_clear()
    yield processor
    token_space_media_process._processor_future.cache_clear()


def _rebuild(multimodal_inputs, tokens):
    pending = token_space_media_process.start_multimodal_train_inputs(
        "ckpt", None, multimodal_inputs, tokens, device=torch.device("cpu")
    )
    return [token_space_media_process.resolve_train_inputs(p) for p in pending]


def test_trainer_processes_the_media_prefix_without_expanding_it(processor):
    (train_inputs,) = _rebuild([{"images": ["/shared/a.png", "/shared/b.png"]}], [torch.tensor([1, 9, 9, 9, 9, 2, 5])])

    (call,) = processor.calls
    # Only the tokens up to the last placeholder go to the processor; they are already expanded.
    assert call["input_ids"] == [1, 9, 9, 9, 9]
    assert call["mm_token_expansion_start_len"] == 5
    assert call["image_data"] == ["/shared/a.png", "/shared/b.png"]
    assert call["image_device"] == "cpu"
    assert set(train_inputs) == {"pixel_values", "image_grid_thw", "mm_image_positions", "mrope_position_ids"}
    # Every tower output row, at the tokens of its media's half-open spans.
    assert train_inputs["mm_image_positions"].tolist() == [1, 2, 3, 4]


def test_mrope_continues_past_the_last_media_per_sample(processor):
    shared = {"images": ["/shared/a.png", "/shared/b.png"]}
    first, second = _rebuild(
        [shared, dict(shared)], [torch.tensor([1, 9, 9, 9, 9, 2, 5]), torch.tensor([1, 9, 9, 9, 9, 7])]
    )

    # One rebuild for the shared prefix; each sample extends it with index + delta over its own tokens.
    assert len(processor.calls) == 1
    assert first["pixel_values"] is second["pixel_values"]
    assert first["mrope_position_ids"][:, 0].tolist() == [0, 1, 1, 2, 2, 3, 4]
    assert second["mrope_position_ids"][:, 0].tolist() == [0, 1, 1, 2, 2, 3]


def test_samples_with_the_same_media_after_different_text_rebuild_apart(processor):
    shared = {"images": ["/shared/a.png", "/shared/b.png"]}
    _rebuild([shared, dict(shared)], [torch.tensor([1, 9, 9, 9, 9, 2]), torch.tensor([3, 9, 9, 9, 9, 2])])

    # The media sits at the same place only when the text before it matches.
    assert len(processor.calls) == 2


def test_text_only_samples_carry_mrope(processor):
    (train_inputs,) = _rebuild([None], [torch.tensor([1, 2])])

    assert processor.calls == []
    assert train_inputs["mrope_position_ids"].tolist() == [[0, 0, 0], [1, 1, 1]]
