import torch

from miles.backends.training_utils.data import token_space_media_process


class _RecordingProcessor:
    def __init__(self):
        self.calls = []

    async def load_and_process_token_space_async(self, **kwargs):
        self.calls.append(kwargs)
        num_images = len(kwargs["image_data"])
        return {
            "input_ids": torch.tensor([kwargs["input_ids"]]),
            "attention_mask": torch.ones(1, len(kwargs["input_ids"])),
            "pixel_values": torch.full((num_images, 3), float(num_images)),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * num_images),
        }


def test_trainer_rebuilds_tensors_without_expanding_the_rollout_tokens(monkeypatch):
    processor = _RecordingProcessor()
    monkeypatch.setattr(
        token_space_media_process, "_token_space_processor", lambda hf_checkpoint, mm_process_config_json: processor
    )
    multimodal_inputs = [{"images": ["/shared/a.png", "/shared/b.png"]}, None]
    tokens = [torch.tensor([1, 9, 9, 9, 9, 2, 5]), torch.tensor([1, 2])]

    train_inputs = token_space_media_process.compute_multimodal_train_inputs(
        "ckpt", None, multimodal_inputs, tokens, device=torch.device("cpu")
    )

    (call,) = processor.calls
    # The rollout tokens are already expanded, so expansion starts past their end.
    assert call["input_ids"] == [1, 9, 9, 9, 9, 2, 5]
    assert call["mm_token_expansion_start_len"] == 7
    assert call["image_data"] == ["/shared/a.png", "/shared/b.png"]
    assert call["image_device"] == "cpu"
    assert train_inputs[1] is None
    assert set(train_inputs[0]) == {"pixel_values", "image_grid_thw"}
    assert torch.equal(train_inputs[0]["image_grid_thw"], torch.tensor([[1, 2, 2], [1, 2, 2]]))
