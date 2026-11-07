import threading

import pytest
import torch
from sglang.srt.multimodal.modality import Modality
from tests.fast.fixtures.sglang_config_fixtures import with_parser_defaults_and_sglang_config
from transformers import BaseImageProcessor

from miles.backends.training_utils.data import token_space_media_process
from miles.backends.training_utils.data.token_space_media_process import RolloutEngineProcessorArgs
from miles.utils.external_utils.command_utils.common import encode_pseudo_file


class _TextMropeStrategy:
    @staticmethod
    def compute_mrope_positions(output):
        return torch.arange(len(output["input_ids"])).expand(3, -1), None


class _RecordingProcessor:
    strategy = _TextMropeStrategy()

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
            "mrope_positions": torch.arange(3 * num_tokens).view(3, num_tokens),
            "mrope_position_delta": torch.tensor([[0]]),
        }


def _rebuild(monkeypatch, multimodal_inputs, tokens, *, with_sglang_placement=True):
    processor = _RecordingProcessor()
    monkeypatch.setattr(
        token_space_media_process,
        "_trainer_processor",
        lambda *_: token_space_media_process._TrainerProcessor(core=processor, image_device="cuda:3"),
    )
    pending = token_space_media_process.compute_multimodal_train_inputs(
        object(),
        multimodal_inputs,
        tokens,
        mm_processor_worker_num=1,
        mm_io_worker_num=0,
        with_sglang_placement=with_sglang_placement,
        device=torch.device("cpu"),
    )
    train_inputs = token_space_media_process.TokenSpaceTrainInputs(
        token_space_media_process.resolved_multimodal_train_inputs(pending.multimodal_train_inputs),
        (
            None
            if pending.mrope_position_ids is None
            else token_space_media_process.resolved_mrope_position_ids(pending.mrope_position_ids)
        ),
    )
    return processor, train_inputs


def test_trainer_rebuilds_tensors_without_expanding_the_rollout_tokens(monkeypatch):
    processor, train_inputs = _rebuild(
        monkeypatch, [{"images": ["/shared/a.png", "/shared/b.png"]}], [torch.tensor([1, 9, 9, 9, 9, 2, 5])]
    )

    (call,) = processor.calls
    # The rollout tokens are already expanded, so expansion starts past their end.
    assert call["input_ids"] == [1, 9, 9, 9, 9, 2, 5]
    assert call["mm_token_expansion_start_len"] == 7
    assert call["image_data"] == ["/shared/a.png", "/shared/b.png"]
    # As the engine's wrapper: the derived image device, and no video device.
    assert call["image_device"] == "cuda:3"
    assert "video_device" not in call
    assert call["compute_mrope_positions"] is True
    assert torch.equal(train_inputs.multimodal_train_inputs[0]["image_grid_thw"], torch.tensor([[1, 2, 2], [1, 2, 2]]))


def test_rebuild_runs_in_the_background(monkeypatch):
    """compute_multimodal_train_inputs returns before the processor ran; resolving an entry waits for it.

    main thread:  compute(...) --returns build futures--> ... resolved_*(...) --waits--> payload
    background:                 process() blocks here ---release---> build returned
    """
    release = threading.Event()
    processor = _RecordingProcessor()
    unblocked_process = processor.process

    async def blocking_process(**kwargs):
        release.wait()
        return await unblocked_process(**kwargs)

    processor.process = blocking_process
    monkeypatch.setattr(
        token_space_media_process,
        "_trainer_processor",
        lambda *_: token_space_media_process._TrainerProcessor(core=processor, image_device=None),
    )

    pending = token_space_media_process.compute_multimodal_train_inputs(
        object(),
        [{"images": ["/shared/a.png", "/shared/b.png"]}],
        [torch.tensor([1, 9, 9, 9, 9, 2, 5])],
        mm_processor_worker_num=1,
        mm_io_worker_num=0,
        with_sglang_placement=True,
        device=torch.device("cpu"),
    )

    (build,) = pending.multimodal_train_inputs
    assert pending.mrope_position_ids == [build]
    assert not build.done()
    release.set()
    assert set(build.result().multimodal_train_inputs) == {"pixel_values", "image_grid_thw", "mm_image_positions"}
    assert build.result().mrope_position_ids.shape == (7, 3)


class _PositionsThatFailToCopy:
    """Stands in for the [3, num_tokens] positions whose device copy is the last step of a rebuild."""

    @property
    def T(self):
        return self

    def contiguous(self):
        return self

    def to(self, device):
        raise RuntimeError("device copy failed")


def test_a_failed_rebuild_raises_where_the_sample_is_resolved(monkeypatch):
    """A build that fails on its last step, the device copy of the positions, raises at the consumer.

    background:   rebuild(): tensors built, then positions.to(device) raises ----> the executor's future holds it
    main thread:  build.result() raises it; so do resolved_multimodal_train_inputs and resolved_mrope_position_ids

    Hand-settled futures left this build pending forever: the copy ran after the try that settled them.
    """
    monkeypatch.setattr(
        token_space_media_process,
        "_trainer_processor",
        lambda *_: token_space_media_process._TrainerProcessor(core=_RecordingProcessor(), image_device=None),
    )
    monkeypatch.setattr(
        token_space_media_process,
        "_sample_train_inputs",
        lambda *args, **kwargs: ({"pixel_values": torch.zeros(1)}, _PositionsThatFailToCopy()),
    )

    pending = token_space_media_process.compute_multimodal_train_inputs(
        object(),
        [{"images": ["/shared/a.png"]}],
        [torch.tensor([1, 9, 2])],
        mm_processor_worker_num=1,
        mm_io_worker_num=0,
        with_sglang_placement=True,
        device=torch.device("cpu"),
    )

    (build,) = pending.multimodal_train_inputs
    with pytest.raises(RuntimeError, match="device copy failed"):
        build.result(timeout=5)
    with pytest.raises(RuntimeError, match="device copy failed"):
        token_space_media_process.resolved_multimodal_train_inputs(pending.multimodal_train_inputs)
    with pytest.raises(RuntimeError, match="device copy failed"):
        token_space_media_process.resolved_mrope_position_ids(pending.mrope_position_ids)


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

    rollout_engine_args = RolloutEngineProcessorArgs.from_sglang_config(values["sglang"], hf_checkpoint="/ckpt")

    assert (rollout_engine_args.model_path, rollout_engine_args.tokenizer_path) == ("/ckpt", "/ckpt")
    assert rollout_engine_args.revision == "v1"
    assert rollout_engine_args.image_processor_backend == "pil"
    assert rollout_engine_args.mm_process_config_json == '{"video": {"fps": 2}}'


def test_rollout_engine_args_take_the_actor_model():
    """A frozen reference model with its own checkpoint does not become the model the trainer rebuilds for."""
    values = with_parser_defaults_and_sglang_config(
        {
            "hf_checkpoint": "/ckpt",
            "rollout_num_gpus": 2,
            "rollout_num_gpus_per_engine": 1,
            "num_gpus_per_node": 8,
            "sglang_image_processor_backend": "pil",
            "sglang_config": encode_pseudo_file(
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 1\n"
                "  - name: ref\n"
                "    model_path: /ref\n"
                "    update_weights: false\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 1\n"
            ),
        }
    )

    rollout_engine_args = RolloutEngineProcessorArgs.from_sglang_config(values["sglang"], hf_checkpoint="/ckpt")

    assert (rollout_engine_args.model_path, rollout_engine_args.tokenizer_path) == ("/ckpt", "/ckpt")
    assert rollout_engine_args.image_processor_backend == "pil"


def test_rollout_engine_args_fall_back_to_the_checkpoint_without_engines():
    """--debug-train-only starts no engine; the replayed samples came from the trainer's own checkpoint."""
    values = with_parser_defaults_and_sglang_config({"hf_checkpoint": "/ckpt", "starts_inference_engines": False})

    rollout_engine_args = RolloutEngineProcessorArgs.from_sglang_config(values["sglang"], hf_checkpoint="/ckpt")

    assert values["sglang"].models == []
    assert (rollout_engine_args.model_path, rollout_engine_args.tokenizer_path) == ("/ckpt", "/ckpt")


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


def test_placement_and_mrope_come_from_sglang(monkeypatch):
    _, train_inputs = _rebuild(
        monkeypatch,
        [{"images": ["/shared/a.png", "/shared/b.png"]}, None],
        [torch.tensor([1, 9, 9, 9, 9, 2, 5]), torch.tensor([1, 2])],
    )

    media, text = train_inputs.multimodal_train_inputs
    assert set(media) == {"pixel_values", "image_grid_thw", "mm_image_positions"}
    assert text is None
    # Every tower output row, at the tokens of its media's half-open spans.
    assert media["mm_image_positions"].tolist() == [1, 2, 3, 4]
    # The positions travel apart from the processor tensors, token-major, text-only samples included.
    media_positions, text_positions = train_inputs.mrope_position_ids
    assert media_positions.shape == (7, 3)
    assert media_positions[:, 1].tolist() == list(range(7, 14))
    assert text_positions.tolist() == [[0, 0, 0], [1, 1, 1]]


def test_models_that_place_media_themselves_get_only_the_processor_tensors(monkeypatch):
    """Megatron-Bridge and torch-native models take no placement kwargs and compute their own positions."""
    processor, train_inputs = _rebuild(
        monkeypatch,
        [{"images": ["/shared/a.png", "/shared/b.png"]}, None],
        [torch.tensor([1, 9, 9, 9, 9, 2, 5]), torch.tensor([1, 2])],
        with_sglang_placement=False,
    )

    assert processor.calls[0]["compute_mrope_positions"] is False
    assert set(train_inputs.multimodal_train_inputs[0]) == {"pixel_values", "image_grid_thw"}
    assert train_inputs.multimodal_train_inputs[1] is None
    assert train_inputs.mrope_position_ids is None
