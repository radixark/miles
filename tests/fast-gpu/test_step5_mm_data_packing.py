"""Step5 image samples packed into micro-batches: positions offset into the packed sequence, text-only batches."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

import pytest  # noqa: E402
import torch  # noqa: E402
from tests.fast.backends.training_utils.step5_mm_rollout import (  # noqa: E402
    IMAGE,
    MODEL_INPUT_KEYS,
    rollout_data,
    set_trivial_parallel_state,
)

from miles.backends.training_utils.data.multimodal import expand_multimodal_rollout_data_in_place  # noqa: E402
from miles.backends.training_utils.data.rollout import DataIterator, get_batch  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="get_batch packs on the current CUDA device")


@pytest.fixture(autouse=True)
def _trivial_parallel_state():
    set_trivial_parallel_state()


_BATCH_KEYS = ["tokens", "loss_masks", "total_lengths", "response_lengths", "multimodal_train_inputs"]


def test_a_mixed_micro_batch_offsets_positions_into_the_packed_sequence():
    data = rollout_data(device="cuda")
    expand_multimodal_rollout_data_in_place(data)

    batch = get_batch(DataIterator(data, micro_batch_size=3), _BATCH_KEYS)

    inputs = batch["multimodal_train_inputs"]
    assert tuple(inputs) == MODEL_INPUT_KEYS
    assert inputs["step5_image_positions"].numel() == sum(int((t == IMAGE).sum()) for t in data["tokens"])
    assert (batch["tokens"][0][inputs["step5_image_positions"]] == IMAGE).all()
    assert inputs["step5_pixel_values"].shape[0] == 3
    assert inputs["step5_num_patches"].tolist() == [0, 1, 0]


def test_a_micro_batch_of_text_samples_hands_the_model_no_image_inputs():
    data = rollout_data(device="cuda")
    expand_multimodal_rollout_data_in_place(data)

    batch = get_batch(DataIterator(data, micro_batch_indices=[[1], [0, 2]]), _BATCH_KEYS)

    assert batch["multimodal_train_inputs"] == {}
