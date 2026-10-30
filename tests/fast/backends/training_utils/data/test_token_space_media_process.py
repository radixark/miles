import pytest
import torch
from tests.fast.fixtures.sglang_config_fixtures import with_parser_defaults_and_sglang_config
from transformers import BaseImageProcessor

from miles.backends.training_utils.data import token_space_media_process
from miles.backends.training_utils.data.token_space_media_process import RolloutEngineProcessorArgs


class _RecordingProcessor:
    def __init__(self):
        self.calls = []

    async def process(self, **kwargs):
        self.calls.append(kwargs)
        num_images = len(kwargs["image_data"])
        return {
            "input_ids": torch.tensor([kwargs["input_ids"]]),
            "attention_mask": torch.ones(1, len(kwargs["input_ids"])),
            "pixel_values": torch.full((num_images, 3), float(num_images)),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * num_images),
            "mm_modalities": ["image"] * num_images,
            "mm_spans": [(1, 3), (3, 5)],
        }


def test_trainer_rebuilds_tensors_without_expanding_the_rollout_tokens(monkeypatch):
    processor = _RecordingProcessor()
    monkeypatch.setattr(
        token_space_media_process,
        "_trainer_processor",
        lambda rollout_engine_args, device: token_space_media_process._TrainerProcessor(
            core=processor, image_device="cuda:3"
        ),
    )
    multimodal_inputs = [{"images": ["/shared/a.png", "/shared/b.png"]}, None]
    tokens = [torch.tensor([1, 9, 9, 9, 9, 2, 5]), torch.tensor([1, 2])]

    train_inputs = token_space_media_process.compute_multimodal_train_inputs(
        object(),
        multimodal_inputs,
        tokens,
        device=torch.device("cpu"),
    )

    (call,) = processor.calls
    # The rollout tokens are already expanded, so expansion starts past their end.
    assert call["input_ids"] == [1, 9, 9, 9, 9, 2, 5]
    assert call["mm_token_expansion_start_len"] == 7
    assert call["image_data"] == ["/shared/a.png", "/shared/b.png"]
    # As the engine's wrapper: the derived image device, and no video device.
    assert call["image_device"] == "cuda:3"
    assert "video_device" not in call
    assert train_inputs[1] is None
    assert set(train_inputs[0]) == {"pixel_values", "image_grid_thw"}
    assert torch.equal(train_inputs[0]["image_grid_thw"], torch.tensor([[1, 2, 2], [1, 2, 2]]))


def test_rollout_engine_args_read_the_parsed_engine_config():
    """Every value comes from the engines' resolved server args; the tokenizer defaults to the model, as in SGLang."""
    values = with_parser_defaults_and_sglang_config(
        {
            "hf_checkpoint": "/ckpt",
            "rollout_num_gpus": 2,
            "rollout_num_gpus_per_engine": 1,
            "num_gpus_per_node": 8,
            "sglang_revision": "v1",
            "sglang_image_processor_backend": "pil",
            "sglang_mm_process_config": {"video": {"fps": 2}},
        }
    )

    rollout_engine_args = RolloutEngineProcessorArgs.from_sglang_config(values["sglang"])

    assert (rollout_engine_args.model_path, rollout_engine_args.tokenizer_path) == ("/ckpt", "/ckpt")
    assert rollout_engine_args.revision == "v1"
    assert rollout_engine_args.image_processor_backend == "pil"
    assert rollout_engine_args.mm_process_config_json == '{"video": {"fps": 2}}'


class _ImageProcessor(BaseImageProcessor):
    pass


@pytest.mark.parametrize(
    ("backend", "image_processor", "expected"),
    [("auto", _ImageProcessor(), "cuda:3"), ("pil", _ImageProcessor(), None), ("auto", object(), None)],
    ids=["image-processor", "pil-backend", "not-an-image-processor"],
)
def test_image_device_follows_the_engine_wrapper(monkeypatch, backend, image_processor, expected):
    """A trainer that preprocesses images on another device than the engines computes other pixels."""
    from sglang.srt.multimodal import serving_preprocessing

    calls = []

    def fake_fast_image_processor_device(processor, *, base_gpu_id, rl_on_policy_target, true_on_policy_enabled):
        calls.append(base_gpu_id)
        return f"cuda:{base_gpu_id}"

    monkeypatch.setattr(serving_preprocessing, "fast_image_processor_device", fake_fast_image_processor_device)
    hf_processor = type("Processor", (), {"image_processor": image_processor})()

    image_device = token_space_media_process._image_device(
        hf_processor, image_processor_backend=backend, device=torch.device("cuda", 3)
    )

    assert image_device == expected
    assert calls == ([3] if expected else [])
