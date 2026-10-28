from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch


@pytest.fixture
def provider_parallel_state(monkeypatch: pytest.MonkeyPatch) -> None:
    group = SimpleNamespace(rank=0)
    monkeypatch.setattr(
        "miles.backends.training_utils.parallel.get_parallel_state",
        lambda: SimpleNamespace(pp=group, tp=group, cp=group, intra_dp=group),
    )


def _legacy_provider(pre_process: bool, post_process: bool) -> _ProviderModel:
    return _ProviderModel(pre_process=pre_process, post_process=post_process, learning_rate=None, vp_stage=None)


def _provider_with_args(pre_process: bool, post_process: bool, args: Any) -> _ProviderModel:
    return _ProviderModel(pre_process=pre_process, post_process=post_process, learning_rate=args.lr, vp_stage=None)


def _provider_with_stage(pre_process: bool, post_process: bool, vp_stage: int | None) -> _ProviderModel:
    return _ProviderModel(pre_process=pre_process, post_process=post_process, learning_rate=None, vp_stage=vp_stage)


def _provider_with_args_and_stage(
    pre_process: bool, post_process: bool, args: Any, vp_stage: int | None
) -> _ProviderModel:
    return _ProviderModel(pre_process=pre_process, post_process=post_process, learning_rate=args.lr, vp_stage=vp_stage)


class _ProviderModel(torch.nn.Module):
    def __init__(
        self, *, pre_process: bool, post_process: bool, learning_rate: float | None, vp_stage: int | None
    ) -> None:
        super().__init__()
        self.pre_process = pre_process
        self.post_process = post_process
        self.learning_rate = learning_rate
        self.vp_stage = vp_stage
