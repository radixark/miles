import copy

import pytest
import torch

from miles.backends.training_utils.data import multimodal
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.utils.types import WeightVersionSpan, WeightVersionsPerCall


def _parallel_state(cp_size: int = 1) -> ParallelState:
    trivial = GroupInfo(rank=0, size=1, group=None)
    cp = GroupInfo(rank=0, size=cp_size, group=None)
    return ParallelState(
        intra_dp=trivial,
        intra_dp_cp=cp,
        cp=cp,
        tp=trivial,
        pp=trivial,
        ep=trivial,
        etp=trivial,
        indep_dp=trivial,
    )


def _validate_remapped_calls(calls: list[dict], *, num_tokens: int) -> list[WeightVersionsPerCall]:
    parsed = [WeightVersionsPerCall.from_dict(call) for call in calls]
    previous_end = 0
    for call in parsed:
        call.validate(num_tokens=num_tokens)
        for span in call.spans:
            assert previous_end <= span.abs_start < span.abs_end <= num_tokens
            previous_end = span.abs_end
    return parsed


class TestKimiVlWeightVersionRemap:
    def test_prompt_image_expansion_shifts_output_spans_and_output_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(multimodal, "get_parallel_state", lambda: _parallel_state(cp_size=1))

        # Pre-expansion: prompt = [10, MEDIA, 20] (len 3), response = [30, 31, 32] (len 3).
        # grid_thws [[1, 4, 4]] -> (4//2)*(4//2) = 4 image tokens -> delta = +3.
        # Post-expansion: prompt len 6, response tokens at [6, 9).
        call = WeightVersionsPerCall(
            spans=[
                WeightVersionSpan(version="1", abs_start=3, abs_end=5),
                WeightVersionSpan(version="2", abs_start=5, abs_end=6),
            ],
            prefill_spans=[
                WeightVersionSpan(version="0", abs_start=0, abs_end=2),
                WeightVersionSpan(version="1", abs_start=2, abs_end=6),
            ],
            output_start=3,
            prompt_tokens=6,
        )
        original_call_dict = call.to_dict()
        snapshot = copy.deepcopy(original_call_dict)

        rollout_data = {
            "tokens": [torch.tensor([10, multimodal.KIMI_VL_MEDIA_TOKEN_ID, 20, 30, 31, 32])],
            "loss_masks": [torch.tensor([1, 1, 1], dtype=torch.int)],
            "total_lengths": [6],
            "response_lengths": [3],
            "multimodal_train_inputs": [{"grid_thws": torch.tensor([[1, 4, 4]])}],
            "weight_versions": [[original_call_dict]],
        }

        multimodal.expand_multimodal_rollout_data_in_place(rollout_data)

        assert rollout_data["total_lengths"] == [9]
        assert rollout_data["response_lengths"] == [3]
        assert rollout_data["tokens"][0][6:9].tolist() == [30, 31, 32]
        assert rollout_data["weight_versions"] == [
            [
                {
                    "spans": [
                        {"version": "1", "abs_start": 6, "abs_end": 8},
                        {"version": "2", "abs_start": 8, "abs_end": 9},
                    ],
                    # Engine prefill_spans and prompt_tokens are already in expanded engine coordinates.
                    "prefill_spans": [
                        {"version": "0", "abs_start": 0, "abs_end": 2},
                        {"version": "1", "abs_start": 2, "abs_end": 6},
                    ],
                    "output_start": 6,
                    "prompt_tokens": 6,
                }
            ]
        ]
        # Original input dict is not mutated in place.
        assert original_call_dict == snapshot
        _validate_remapped_calls(rollout_data["weight_versions"][0], num_tokens=9)

    def test_multi_turn_with_media_in_prompt_and_later_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(multimodal, "get_parallel_state", lambda: _parallel_state(cp_size=1))

        # Pre-expansion (len 10, prompt_len 3, response_len 7):
        #   prompt: [10, MEDIA(4 tok, +3), 11]           -> [0, 3)
        #   turn 1 output: [20, 21]                      -> [3, 5)
        #   tool obs (loss_mask=0): [30, MEDIA(6 tok, +5), 31] -> [5, 8)
        #   turn 2 output: [40, 41]                      -> [8, 10)
        # Post-expansion (len 18, prompt_len 6, response_len 12):
        #   turn 1 output shifts by +3 -> [6, 8), output_start=6
        #   turn 2 output shifts by +3 + +5 = +8 -> [16, 18), output_start=16
        turn1 = WeightVersionsPerCall(
            spans=[WeightVersionSpan(version="1", abs_start=3, abs_end=5)],
            prefill_spans=[WeightVersionSpan(version="1", abs_start=0, abs_end=6)],
            output_start=3,
            prompt_tokens=6,
        )
        turn2 = WeightVersionsPerCall(
            spans=[WeightVersionSpan(version="2", abs_start=8, abs_end=10)],
            prefill_spans=[
                WeightVersionSpan(version="1", abs_start=0, abs_end=8),
                WeightVersionSpan(version="2", abs_start=8, abs_end=16),
            ],
            output_start=8,
            prompt_tokens=16,
        )
        rollout_data = {
            "tokens": [
                torch.tensor(
                    [
                        10,
                        multimodal.KIMI_VL_MEDIA_TOKEN_ID,
                        11,
                        20,
                        21,
                        30,
                        multimodal.KIMI_VL_MEDIA_TOKEN_ID,
                        31,
                        40,
                        41,
                    ]
                )
            ],
            "loss_masks": [torch.tensor([1, 1, 0, 0, 0, 1, 1], dtype=torch.int)],
            "total_lengths": [10],
            "response_lengths": [7],
            "multimodal_train_inputs": [{"grid_thws": torch.tensor([[1, 4, 4], [1, 4, 6]])}],
            "weight_versions": [[turn1.to_dict(), turn2.to_dict()]],
        }

        multimodal.expand_multimodal_rollout_data_in_place(rollout_data)

        assert rollout_data["total_lengths"] == [18]
        assert rollout_data["response_lengths"] == [12]
        assert rollout_data["tokens"][0][6:8].tolist() == [20, 21]
        assert rollout_data["tokens"][0][16:18].tolist() == [40, 41]
        parsed = _validate_remapped_calls(rollout_data["weight_versions"][0], num_tokens=18)
        assert parsed[0].output_start == 6
        assert parsed[0].spans == [WeightVersionSpan(version="1", abs_start=6, abs_end=8)]
        assert parsed[1].output_start == 16
        assert parsed[1].spans == [WeightVersionSpan(version="2", abs_start=16, abs_end=18)]

    def test_idempotent_when_called_twice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(multimodal, "get_parallel_state", lambda: _parallel_state(cp_size=1))

        rollout_data = {
            "tokens": [torch.tensor([1, multimodal.KIMI_VL_MEDIA_TOKEN_ID, 2, 3])],
            "loss_masks": [torch.tensor([1, 1], dtype=torch.int)],
            "total_lengths": [4],
            "response_lengths": [2],
            "multimodal_train_inputs": [{"grid_thws": torch.tensor([[1, 4, 4]])}],
            "weight_versions": [
                [
                    WeightVersionsPerCall(
                        spans=[WeightVersionSpan(version="5", abs_start=2, abs_end=4)],
                        output_start=2,
                    ).to_dict()
                ]
            ],
        }

        multimodal.expand_multimodal_rollout_data_in_place(rollout_data)
        after_first = copy.deepcopy(rollout_data["weight_versions"])
        multimodal.expand_multimodal_rollout_data_in_place(rollout_data)

        assert rollout_data["weight_versions"] == after_first
        assert rollout_data["weight_versions"][0][0]["spans"] == [{"version": "5", "abs_start": 5, "abs_end": 7}]
        assert rollout_data["weight_versions"][0][0]["output_start"] == 5

    def test_mixed_batch_preserves_text_only_and_empty_weight_version_samples(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(multimodal, "get_parallel_state", lambda: _parallel_state(cp_size=1))

        mm_call = WeightVersionsPerCall(
            spans=[WeightVersionSpan(version="1", abs_start=2, abs_end=4)],
            output_start=2,
        ).to_dict()
        text_call = WeightVersionsPerCall(
            spans=[WeightVersionSpan(version="2", abs_start=1, abs_end=3)],
            output_start=1,
        ).to_dict()
        rollout_data = {
            "tokens": [
                torch.tensor([multimodal.KIMI_VL_MEDIA_TOKEN_ID, 10, 20, 21]),
                torch.tensor([5, 6, 7]),
                torch.tensor([multimodal.KIMI_VL_MEDIA_TOKEN_ID, 30, 31]),
            ],
            "loss_masks": [
                torch.tensor([1, 1], dtype=torch.int),
                torch.tensor([1, 1], dtype=torch.int),
                torch.tensor([1, 1], dtype=torch.int),
            ],
            "total_lengths": [4, 3, 3],
            "response_lengths": [2, 2, 2],
            "multimodal_train_inputs": [
                {"grid_thws": torch.tensor([[1, 4, 4]])},
                None,
                {"grid_thws": torch.tensor([[1, 2, 4]])},
            ],
            "weight_versions": [[mm_call], [text_call], []],
        }

        multimodal.expand_multimodal_rollout_data_in_place(rollout_data)

        assert rollout_data["total_lengths"] == [7, 3, 4]
        assert rollout_data["weight_versions"][0][0]["spans"] == [{"version": "1", "abs_start": 5, "abs_end": 7}]
        assert rollout_data["weight_versions"][0][0]["output_start"] == 5
        assert rollout_data["weight_versions"][1] == [text_call]
        assert rollout_data["weight_versions"][2] == []

    def test_extra_trailing_placeholders_beyond_grid_count_do_not_shift(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(multimodal, "get_parallel_state", lambda: _parallel_state(cp_size=1))

        # 2 placeholders, only 1 grid (grid_idx=0 at pos=0 expands by +3; pos=2 is skipped).
        rollout_data = {
            "tokens": [
                torch.tensor([multimodal.KIMI_VL_MEDIA_TOKEN_ID, 10, multimodal.KIMI_VL_MEDIA_TOKEN_ID, 20, 21])
            ],
            "loss_masks": [torch.tensor([1, 1], dtype=torch.int)],
            "total_lengths": [5],
            "response_lengths": [2],
            "multimodal_train_inputs": [{"grid_thws": torch.tensor([[1, 4, 4]])}],
            "weight_versions": [
                [
                    WeightVersionsPerCall(
                        spans=[WeightVersionSpan(version="3", abs_start=3, abs_end=5)],
                        output_start=3,
                    ).to_dict()
                ]
            ],
        }

        multimodal.expand_multimodal_rollout_data_in_place(rollout_data)

        assert rollout_data["total_lengths"] == [8]
        assert rollout_data["tokens"][0][6:8].tolist() == [20, 21]
        assert rollout_data["weight_versions"][0][0]["spans"] == [{"version": "3", "abs_start": 6, "abs_end": 8}]
        assert rollout_data["weight_versions"][0][0]["output_start"] == 6


class TestInklingWeightVersionRemap:
    def test_image_and_audio_sentinels_remap_weight_versions_and_stay_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(multimodal, "get_parallel_state", lambda: _parallel_state(cp_size=1))

        # Pre-expansion: prompt = [1, IMG(-101, 4 patches -> +3), 2, AUD(-102, 3 tokens -> +2), 3] (len 5)
        # response = [10, 11, 12] (len 3), total = 8.
        # Post-expansion: prompt len = 10, response at [10, 13), total = 13.
        call1 = WeightVersionsPerCall(
            spans=[WeightVersionSpan(version="4", abs_start=5, abs_end=7)],
            prefill_spans=[WeightVersionSpan(version="3", abs_start=0, abs_end=10)],
            output_start=5,
            prompt_tokens=10,
        )
        call2 = WeightVersionsPerCall(
            spans=[WeightVersionSpan(version="5", abs_start=7, abs_end=8)],
            prefill_spans=[
                WeightVersionSpan(version="3", abs_start=0, abs_end=10),
                WeightVersionSpan(version="4", abs_start=10, abs_end=12),
            ],
            output_start=7,
            prompt_tokens=12,
        )
        rollout_data = {
            "tokens": [
                torch.tensor(
                    [
                        1,
                        multimodal.INKLING_IMAGE_SENTINEL_ID,
                        2,
                        multimodal.INKLING_AUDIO_SENTINEL_ID,
                        3,
                        10,
                        11,
                        12,
                    ]
                )
            ],
            "loss_masks": [torch.tensor([1, 1, 1], dtype=torch.int)],
            "total_lengths": [8],
            "response_lengths": [3],
            "multimodal_train_inputs": [
                {
                    "mm_vision_num_patches": torch.tensor([4]),
                    "mm_audio_num_tokens": torch.tensor([3]),
                }
            ],
            "weight_versions": [[call1.to_dict(), call2.to_dict()]],
        }

        multimodal.expand_multimodal_rollout_data_in_place(rollout_data)

        assert rollout_data["total_lengths"] == [13]
        assert rollout_data["tokens"][0][10:13].tolist() == [10, 11, 12]
        parsed = _validate_remapped_calls(rollout_data["weight_versions"][0], num_tokens=13)
        assert parsed[0] == WeightVersionsPerCall(
            spans=[WeightVersionSpan(version="4", abs_start=10, abs_end=12)],
            prefill_spans=[WeightVersionSpan(version="3", abs_start=0, abs_end=10)],
            output_start=10,
            prompt_tokens=10,
        )
        assert parsed[1] == WeightVersionsPerCall(
            spans=[WeightVersionSpan(version="5", abs_start=12, abs_end=13)],
            prefill_spans=[
                WeightVersionSpan(version="3", abs_start=0, abs_end=10),
                WeightVersionSpan(version="4", abs_start=10, abs_end=12),
            ],
            output_start=12,
            prompt_tokens=12,
        )

        # Calling a second time must be a no-op because sentinels are already replaced.
        after_first = copy.deepcopy(rollout_data["weight_versions"])
        multimodal.expand_multimodal_rollout_data_in_place(rollout_data)
        assert rollout_data["weight_versions"] == after_first

    def test_mixed_inkling_and_text_samples(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(multimodal, "get_parallel_state", lambda: _parallel_state(cp_size=1))

        mm_call = WeightVersionsPerCall(
            spans=[WeightVersionSpan(version="1", abs_start=2, abs_end=4)],
            output_start=2,
        ).to_dict()
        text_call = WeightVersionsPerCall(
            spans=[WeightVersionSpan(version="1", abs_start=2, abs_end=4)],
            output_start=2,
        ).to_dict()
        rollout_data = {
            "tokens": [
                torch.tensor([multimodal.INKLING_IMAGE_SENTINEL_ID, 9, 10, 11]),
                torch.tensor([5, 6, 7, 8]),
            ],
            "loss_masks": [
                torch.tensor([1, 1], dtype=torch.int),
                torch.tensor([1, 1], dtype=torch.int),
            ],
            "total_lengths": [4, 4],
            "response_lengths": [2, 2],
            "multimodal_train_inputs": [
                {"mm_vision_num_patches": torch.tensor([5])},
                None,
            ],
            "weight_versions": [[mm_call], [text_call]],
        }

        multimodal.expand_multimodal_rollout_data_in_place(rollout_data)

        assert rollout_data["total_lengths"] == [8, 4]
        assert rollout_data["weight_versions"][0][0]["spans"] == [{"version": "1", "abs_start": 6, "abs_end": 8}]
        assert rollout_data["weight_versions"][0][0]["output_start"] == 6
        assert rollout_data["weight_versions"][1] == [text_call]


class TestRemapWeightVersionRepresentations:
    def test_supports_dataclass_legacy_list_and_unstamped_calls(self) -> None:
        expansions = [(1, 3)]

        dataclass_call = WeightVersionsPerCall(
            spans=[WeightVersionSpan(version="7", abs_start=3, abs_end=5)],
            prefill_spans=[WeightVersionSpan(version="6", abs_start=0, abs_end=6)],
            output_start=3,
            prompt_tokens=6,
        )
        legacy_list_call = [{"version": "7", "abs_start": 3, "abs_end": 5}]
        unstamped_dict_call = {"spans": [], "prefill_spans": [], "output_start": None, "prompt_tokens": None}

        remapped = multimodal._remap_weight_versions_for_sample(
            [dataclass_call, legacy_list_call, unstamped_dict_call],
            expansions,
        )

        assert remapped == [
            WeightVersionsPerCall(
                spans=[WeightVersionSpan(version="7", abs_start=6, abs_end=8)],
                prefill_spans=[WeightVersionSpan(version="6", abs_start=0, abs_end=6)],
                output_start=6,
                prompt_tokens=6,
            ),
            [{"version": "7", "abs_start": 6, "abs_end": 8}],
            {"spans": [], "prefill_spans": [], "output_start": None, "prompt_tokens": None},
        ]
