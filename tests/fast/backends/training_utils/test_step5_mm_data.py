"""Step5 trainer-side expansion: the engine-expanded prompt swapped in, image positions, idempotence, text samples."""

import pytest
import torch
from tests.fast.backends.training_utils.step5_mm_rollout import (
    IMAGE,
    MODEL_INPUT_KEYS,
    rollout_data,
    set_trivial_parallel_state,
)

from miles.backends.training_utils.data.multimodal import expand_multimodal_rollout_data_in_place
from miles.utils.types import STEP5_EXPANDED_PROMPT_KEY


@pytest.fixture(autouse=True)
def _trivial_parallel_state():
    set_trivial_parallel_state()


def test_the_expanded_prompt_replaces_the_request_prompt():
    data = rollout_data()
    expanded_prompts = [
        mm[STEP5_EXPANDED_PROMPT_KEY].tolist() if mm else None for mm in data["multimodal_train_inputs"]
    ]

    expand_multimodal_rollout_data_in_place(data)

    assert data["tokens"][0].tolist() == expanded_prompts[0] + [20, 21, 22]
    assert data["tokens"][1].tolist() == [5, 6, 8, 23, 24]
    assert data["tokens"][2].tolist() == expanded_prompts[2] + [25]
    assert data["total_lengths"] == [len(t) for t in data["tokens"]]
    assert data["response_lengths"] == [3, 2, 1]


def test_image_samples_carry_exactly_the_model_inputs_and_text_samples_none():
    data = rollout_data()

    expand_multimodal_rollout_data_in_place(data)

    image_inputs, text_inputs, other_image_inputs = data["multimodal_train_inputs"]
    assert tuple(image_inputs) == tuple(other_image_inputs) == MODEL_INPUT_KEYS
    assert text_inputs is None


def test_positions_point_at_the_image_tokens():
    data = rollout_data()

    expand_multimodal_rollout_data_in_place(data)

    for token_tensor, mm in zip(data["tokens"], data["multimodal_train_inputs"], strict=True):
        positions = mm["step5_image_positions"] if mm is not None else torch.empty(0, dtype=torch.long)
        assert (token_tensor[positions] == IMAGE).all()
        assert positions.numel() == (token_tensor == IMAGE).sum()


def test_expanding_twice_changes_nothing():
    data = rollout_data()
    expand_multimodal_rollout_data_in_place(data)
    tokens = [t.clone() for t in data["tokens"]]
    inputs = [dict(mm) if mm is not None else None for mm in data["multimodal_train_inputs"]]

    expand_multimodal_rollout_data_in_place(data)

    assert all(torch.equal(a, b) for a, b in zip(tokens, data["tokens"], strict=True))
    for before, after in zip(inputs, data["multimodal_train_inputs"], strict=True):
        assert (before is None) == (after is None)
        assert before is None or (before.keys() == after.keys() and all(before[k] is after[k] for k in before))


def test_context_parallelism_is_refused():
    set_trivial_parallel_state(cp_size=2)
    with pytest.raises(AssertionError, match="CP>1"):
        expand_multimodal_rollout_data_in_place(rollout_data())
