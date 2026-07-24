from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils import data as data_utils
from miles.backends.training_utils import mm_data
from miles.backends.training_utils.loss_hub.opd import apply_opd_kl_to_advantages
from miles.backends.training_utils.parallel import GroupInfo, ParallelState

_ROLLOUT_LOG_PROBS = torch.tensor([-0.2, -1.3, -0.7, -2.1, -0.4, -3.2, -1.8])
_OPD_VALUES = torch.tensor([0.1, 0.9, -0.3, 1.7, -1.1, 0.4, 2.3])


def _parallel_state(*, cp_size: int, cp_rank: int = 0) -> ParallelState:
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    return ParallelState(
        intra_dp=trivial_group,
        intra_dp_cp=GroupInfo(rank=cp_rank, size=cp_size, group=None),
        cp=GroupInfo(rank=cp_rank, size=cp_size, group=None),
        tp=trivial_group,
        pp=trivial_group,
        ep=trivial_group,
        etp=trivial_group,
        indep_dp=trivial_group,
    )


def _args(qkv_format: str) -> Namespace:
    return Namespace(
        enable_witness=False,
        qkv_format=qkv_format,
        data_pad_size_multiplier=16,
        compress_ratios=[],
        true_on_policy_mode=False,
        bf16=False,
        fp16=False,
    )


def _load_rollout_data(
    monkeypatch: pytest.MonkeyPatch,
    *,
    qkv_format: str,
    cp_size: int,
    cp_rank: int,
    opd_key: str,
) -> dict:
    parallel_state = _parallel_state(cp_size=cp_size, cp_rank=cp_rank)
    rollout_data = {
        "tokens": [list(range(11))],
        "loss_masks": [[1] * 7],
        "total_lengths": [11],
        "response_lengths": [7],
        "rollout_log_probs": [_ROLLOUT_LOG_PROBS.tolist()],
        opd_key: [_OPD_VALUES.clone()],
    }

    monkeypatch.setattr(data_utils, "process_rollout_data", lambda *args, **kwargs: rollout_data)
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    return data_utils.get_rollout_data(_args(qkv_format), object())


@pytest.mark.parametrize("opd_key", ["teacher_log_probs", "opd_reverse_kl"])
@pytest.mark.parametrize(
    ("qkv_format", "cp_size", "cp_rank", "expected_indices"),
    [
        ("thd", 1, 0, [0, 1, 2, 3, 4, 5, 6]),
        ("thd", 2, 0, [6]),
        ("thd", 2, 1, [0, 1, 2, 3, 4, 5]),
        ("bshd", 2, 0, [0]),
        ("bshd", 2, 1, [1, 2, 3, 4, 5, 6]),
    ],
)
def test_sglang_opd_response_fields_follow_rollout_log_prob_cp_slice(
    monkeypatch: pytest.MonkeyPatch,
    opd_key: str,
    qkv_format: str,
    cp_size: int,
    cp_rank: int,
    expected_indices: list[int],
) -> None:
    rollout_data = _load_rollout_data(
        monkeypatch,
        qkv_format=qkv_format,
        cp_size=cp_size,
        cp_rank=cp_rank,
        opd_key=opd_key,
    )

    expected_indices_tensor = torch.tensor(expected_indices)
    torch.testing.assert_close(
        rollout_data["rollout_log_probs"][0],
        _ROLLOUT_LOG_PROBS[expected_indices_tensor],
    )
    torch.testing.assert_close(
        rollout_data[opd_key][0],
        _OPD_VALUES[expected_indices_tensor],
    )
    assert rollout_data[opd_key][0].dtype == torch.float32
    assert rollout_data[opd_key][0].device.type == "cpu"

    advantages = [torch.ones(len(expected_indices), dtype=torch.float32)]
    student_log_probs = [rollout_data[opd_key][0] + 0.25]
    apply_opd_kl_to_advantages(
        Namespace(opd_type="sglang", opd_kl_coef=0.5),
        rollout_data,
        advantages,
        student_log_probs,
    )

    if opd_key == "teacher_log_probs":
        torch.testing.assert_close(advantages[0], torch.full_like(advantages[0], 0.875))
    else:
        torch.testing.assert_close(
            advantages[0],
            1.0 - 0.5 * rollout_data[opd_key][0],
        )


def test_multimodal_cp_reslices_precomputed_opd_reverse_kl(monkeypatch: pytest.MonkeyPatch) -> None:
    rollout_data = {
        "tokens": [torch.tensor([mm_data.KIMI_VL_MEDIA_TOKEN_ID, 1, 2])],
        "loss_masks": [torch.ones(2, dtype=torch.int)],
        "total_lengths": [3],
        "response_lengths": [2],
        "multimodal_train_inputs": [{"grid_thws": torch.tensor([[1, 4, 4]])}],
        "opd_reverse_kl": [torch.tensor([0.1, 0.2])],
    }
    gathered_keys = []

    monkeypatch.setattr(mm_data, "get_parallel_state", lambda: _parallel_state(cp_size=2))

    def _all_gather(value: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        gathered_keys.append(value)
        return value

    monkeypatch.setattr(mm_data, "all_gather_with_cp", _all_gather)
    monkeypatch.setattr(mm_data, "slice_log_prob_with_cp", lambda value, *args, **kwargs: value)

    mm_data.expand_multimodal_rollout_data_in_place(rollout_data)

    assert len(gathered_keys) == 1
    assert gathered_keys[0] is rollout_data["opd_reverse_kl"][0]
