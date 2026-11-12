import random

import pytest
import torch

from miles.backends.training_utils.data import multimodal
from miles.backends.training_utils.data.multimodal import (
    INKLING_AUDIO_SENTINEL_ID,
    INKLING_IMAGE_SENTINEL_ID,
    KIMI_VL_MEDIA_TOKEN_ID,
    expand_multimodal_rollout_data_in_place,
    trained_total_lengths,
)
from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.parallel import GroupInfo, ParallelState


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    group = GroupInfo(rank=0, size=1, group=None)
    state = ParallelState(
        intra_dp=group, intra_dp_cp=group, cp=group, tp=group, pp=group, ep=group, etp=group, indep_dp=group
    )
    monkeypatch.setattr(multimodal, "get_parallel_state", lambda: state)


def _rollout_data(samples: list[tuple[list[int], int, dict | None]]) -> dict:
    return {
        "tokens": [torch.tensor(tokens) for tokens, _, _ in samples],
        "loss_masks": [torch.ones(response_length, dtype=torch.int32) for _, response_length, _ in samples],
        "total_lengths": [len(tokens) for tokens, _, _ in samples],
        "response_lengths": [response_length for _, response_length, _ in samples],
        "multimodal_train_inputs": [None if mm is None else dict(mm) for _, _, mm in samples],
    }


def _expanded_lengths(samples: list[tuple[list[int], int, dict | None]]) -> list[int]:
    rollout_data = _rollout_data(samples)
    expand_multimodal_rollout_data_in_place(rollout_data)
    batch = DataIterator(rollout_data, micro_batch_size=len(samples)).get_next(list(rollout_data))
    expand_multimodal_rollout_data_in_place(batch)
    assert [len(tokens) for tokens in batch["tokens"]] == rollout_data["total_lengths"]
    return rollout_data["total_lengths"]


def _planned_lengths(samples: list[tuple[list[int], int, dict | None]]) -> list[int]:
    return trained_total_lengths([tokens for tokens, _, _ in samples], [mm for _, _, mm in samples])


def _place(rng: random.Random, tokens: list[int], token_id: int, count: int, *, end: int) -> None:
    for position in rng.sample(range(end), count):
        tokens[position] = token_id


def _kimi_sample(rng: random.Random) -> tuple[list[int], int, dict | None]:
    grids = [[1, rng.choice([2, 4, 6, 8]), rng.choice([2, 4, 6, 8])] for _ in range(rng.randint(0, 3))]
    placeholders = rng.choice([0, len(grids), sum((h // 2) * (w // 2) for _, h, w in grids)])
    prompt_length, response_length = rng.randint(4, 20) + placeholders, rng.randint(1, 20)
    tokens = [rng.randint(1, 1000) for _ in range(prompt_length + response_length)]
    if not grids:
        return tokens, response_length, rng.choice([None, {"pixel_values": torch.zeros(1, 3)}])
    _place(rng, tokens, KIMI_VL_MEDIA_TOKEN_ID, placeholders, end=len(tokens))
    return tokens, response_length, {"grid_thws": torch.tensor(grids)}


def _inkling_sample(rng: random.Random) -> tuple[list[int], int, dict | None]:
    prompt_length, response_length = rng.randint(6, 20), rng.randint(1, 20)
    tokens = [rng.randint(1, 1000) for _ in range(prompt_length + response_length)]
    images, audios = rng.randint(0, 2), rng.randint(0, 2)
    if images + audios == 0:
        return (
            tokens,
            response_length,
            rng.choice([None, {"mm_vision_num_patches": torch.tensor([], dtype=torch.long)}]),
        )
    positions = rng.sample(range(prompt_length), images + audios)
    for position in positions[:images]:
        tokens[position] = INKLING_IMAGE_SENTINEL_ID
    for position in positions[images:]:
        tokens[position] = INKLING_AUDIO_SENTINEL_ID
    mm = {}
    if images:
        mm["mm_vision_num_patches"] = torch.tensor([rng.randint(1, 30) for _ in range(images)])
    if audios:
        mm["mm_audio_num_tokens"] = torch.tensor([rng.randint(1, 30) for _ in range(audios)])
    return tokens, response_length, mm


@pytest.mark.parametrize("seed", range(40))
def test_kimi_batches_are_planned_at_their_expanded_lengths(seed: int) -> None:
    rng = random.Random(seed)
    samples = [_kimi_sample(rng) for _ in range(rng.randint(1, 6))]

    assert _planned_lengths(samples) == _expanded_lengths(samples)


@pytest.mark.parametrize("seed", range(40))
def test_inkling_batches_are_planned_at_their_expanded_lengths(seed: int) -> None:
    rng = random.Random(seed)
    samples = [_inkling_sample(rng) for _ in range(rng.randint(1, 6))]

    assert _planned_lengths(samples) == _expanded_lengths(samples)


@pytest.mark.parametrize("placeholders", [2, 3, 5])
def test_kimi_rejects_placeholder_counts_that_match_neither_grids_nor_tokens(placeholders: int) -> None:
    samples = [([1] + [KIMI_VL_MEDIA_TOKEN_ID] * placeholders + [2, 3], 1, {"grid_thws": torch.tensor([[1, 4, 4]])})]

    with pytest.raises(ValueError, match="media placeholders"):
        _planned_lengths(samples)
    with pytest.raises(ValueError, match="media placeholders"):
        expand_multimodal_rollout_data_in_place(_rollout_data(samples))


def test_media_expanded_at_rollout_keeps_its_length() -> None:
    samples = [
        ([1, 2, 3, 4], 2, {"pixel_values": torch.zeros(4, 3), "image_grid_thw": torch.tensor([[1, 2, 2]])}),
        ([5, 6, 7], 1, None),
    ]

    assert _planned_lengths(samples) == [4, 3] == _expanded_lengths(samples)
    assert trained_total_lengths([[1, 2], [3]], None) == [2, 1]


def test_kimi_placeholder_grows_by_its_grid() -> None:
    tokens = [1, KIMI_VL_MEDIA_TOKEN_ID, 2, 3]

    assert trained_total_lengths([tokens], [{"grid_thws": torch.tensor([[1, 8, 6]])}]) == [4 - 1 + 4 * 3]
