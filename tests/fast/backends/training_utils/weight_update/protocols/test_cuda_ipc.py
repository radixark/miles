from argparse import Namespace
from unittest.mock import Mock

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.cuda_ipc import UpdateWeightFromTensor


def _protocol(*, offload_train: bool) -> UpdateWeightFromTensor:
    protocol = object.__new__(UpdateWeightFromTensor)
    protocol.args = Namespace(offload_train=offload_train)
    return protocol


class TestAfterEnginesResumed:
    def test_a_resident_trainer_returns_its_ipc_blocks_to_the_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without offload the cached IPC blocks are safe to release once the engines have consumed them."""
        ipc_collect = Mock()
        empty_cache = Mock()
        monkeypatch.setattr(torch.cuda, "ipc_collect", ipc_collect)
        monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

        _protocol(offload_train=False).after_engines_resumed()

        ipc_collect.assert_called_once_with()
        empty_cache.assert_called_once_with()

    def test_an_offloaded_trainer_leaves_the_paused_allocations_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With offload the trainer may hold paused memory-saver blocks, which empty_cache would free twice."""
        ipc_collect = Mock()
        empty_cache = Mock()
        monkeypatch.setattr(torch.cuda, "ipc_collect", ipc_collect)
        monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

        _protocol(offload_train=True).after_engines_resumed()

        ipc_collect.assert_not_called()
        empty_cache.assert_not_called()
