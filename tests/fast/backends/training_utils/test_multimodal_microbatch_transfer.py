"""Tests for per-microbatch multimodal tensor collation and deferred transfer."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import numpy as np
import pytest
import torch

from miles.backends.training_utils.mm_data import collate_multimodal_train_inputs, materialize_multimodal_inputs


def test_collate_multimodal_train_inputs_uses_only_selected_microbatch() -> None:
    """The active microbatch is the only multimodal payload collated for forward."""
    full_payload = [
        {
            "pixel_values": torch.tensor([[1.0], [2.0]]),
            "image_grid_thw": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        },
        {
            "pixel_values": torch.tensor([[99.0]]),
            "image_grid_thw": torch.tensor([[9, 9, 9]]),
        },
        {
            "pixel_values": torch.tensor([[3.0], [4.0], [5.0]]),
            "image_grid_thw": torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]]),
        },
    ]
    selected_microbatch = [full_payload[0], full_payload[2]]

    multimodal_data, multimodal_num_items = collate_multimodal_train_inputs(
        selected_microbatch,
        torch.device("cpu"),
    )

    torch.testing.assert_close(multimodal_data["pixel_values"], torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]]))
    torch.testing.assert_close(
        multimodal_data["image_grid_thw"],
        torch.tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]),
    )
    assert multimodal_num_items == {"pixel_values": [2, 3], "image_grid_thw": [2, 3]}


def test_collate_multimodal_train_inputs_handles_numpy_and_none_entries() -> None:
    """Deferred CPU materialization can leave numpy arrays mixed with empty samples."""
    microbatch = [
        None,
        {
            "pixel_values": np.array([[1.0, 2.0]], dtype=np.float32),
            "image_grid_thw": np.array([[1, 2, 3]], dtype=np.int64),
        },
        {
            "pixel_values": torch.tensor([[3.0, 4.0]], dtype=torch.float32),
            "image_grid_thw": torch.tensor([[4, 5, 6]], dtype=torch.int64),
        },
    ]

    multimodal_data, multimodal_num_items = collate_multimodal_train_inputs(microbatch, torch.device("cpu"))

    torch.testing.assert_close(multimodal_data["pixel_values"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch.testing.assert_close(multimodal_data["image_grid_thw"], torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert multimodal_num_items == {"pixel_values": [1, 1], "image_grid_thw": [1, 1]}


def test_collate_multimodal_train_inputs_handles_text_only_microbatch() -> None:
    """Text-only microbatches should not add multimodal model kwargs."""
    multimodal_data, multimodal_num_items = collate_multimodal_train_inputs([None, None], torch.device("cpu"))

    assert multimodal_data == {}
    assert multimodal_num_items == {}


def test_collate_multimodal_train_inputs_fails_loudly_on_bad_values() -> None:
    """Unexpected multimodal value types should fail before model forward."""
    with pytest.raises(TypeError, match="Expected multimodal value"):
        collate_multimodal_train_inputs([{"pixel_values": [[1.0, 2.0]]}], torch.device("cpu"))


def test_materialize_multimodal_inputs_keeps_deferred_tensors_on_cpu() -> None:
    """Deferred actor materialization resolves arrays to CPU tensors without CUDA placement."""
    materialized = materialize_multimodal_inputs(
        [
            {
                "pixel_values": np.array([[1.0, 2.0]], dtype=np.float32),
                "image_grid_thw": np.array([[1, 1, 1]], dtype=np.int64),
            },
            None,
        ],
        device=torch.device("cpu"),
    )

    assert len(materialized) == 2
    assert materialized[1] is None
    assert materialized[0] is not None
    torch.testing.assert_close(materialized[0]["pixel_values"], torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    torch.testing.assert_close(materialized[0]["image_grid_thw"], torch.tensor([[1, 1, 1]], dtype=torch.int64))
    assert materialized[0]["pixel_values"].device.type == "cpu"
    assert materialized[0]["pixel_values"].is_pinned() is torch.cuda.is_available()


def test_materialize_multimodal_inputs_copies_numpy_values() -> None:
    """Resolved tensors should not alias mutable arrays owned by the rollout side."""
    pixel_values = np.array([[1.0, 2.0]], dtype=np.float32)
    materialized = materialize_multimodal_inputs([{"pixel_values": pixel_values}], device=torch.device("cpu"))

    pixel_values[0, 0] = 99.0

    assert materialized[0] is not None
    torch.testing.assert_close(materialized[0]["pixel_values"], torch.tensor([[1.0, 2.0]], dtype=torch.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for the deferred host->device path")
def test_deferred_materialize_then_collate_lands_on_cuda() -> None:
    """End-to-end deferred path smoke test: pinned-CPU materialize, then collate onto CUDA.

    No other test exercises the actual host->device path (the rest collate onto
    CPU). This confirms the deferred pipeline — materialize on pinned CPU, then
    collate onto CUDA — yields a correctly concatenated payload resident on the
    device. It does not assert the copy is non_blocking (async vs sync is not
    observable from the output); that property is guaranteed structurally by
    moving each pinned tensor over before the concat in
    ``_cat_multimodal_tensors_for_forward``.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    materialized = materialize_multimodal_inputs(
        [
            {"pixel_values": np.array([[1.0], [2.0]], dtype=np.float32)},
            {"pixel_values": np.array([[3.0]], dtype=np.float32)},
        ],
        device=torch.device("cpu"),
    )
    # Deferred payload is pinned CPU before collation.
    assert materialized[0] is not None and materialized[0]["pixel_values"].device.type == "cpu"

    multimodal_data, multimodal_num_items = collate_multimodal_train_inputs(materialized, device)

    assert multimodal_data["pixel_values"].device.type == "cuda"
    torch.testing.assert_close(
        multimodal_data["pixel_values"].cpu(),
        torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32),
    )
    assert multimodal_num_items == {"pixel_values": [2, 1]}
