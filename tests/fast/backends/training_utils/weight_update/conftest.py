from argparse import Namespace
from collections.abc import Callable

import pytest

from miles.backends.training_utils.weight_update.protocols import cuda_ipc
from miles.backends.training_utils.weight_update.protocols.cuda_ipc import UpdateWeightFromTensor


@pytest.fixture
def colocated_protocol(monkeypatch: pytest.MonkeyPatch) -> Callable[..., UpdateWeightFromTensor]:
    monkeypatch.setattr(cuda_ipc.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(cuda_ipc.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(cuda_ipc.dist, "new_group", lambda ranks, backend: object())

    def create(*, backup: bool, checker: bool, offload: bool) -> UpdateWeightFromTensor:
        return UpdateWeightFromTensor(
            Namespace(
                rollout_num_gpus_per_engine=1,
                lora_rank=8,
                lora_adapter_path=None,
                lora_base_cpu_backup=backup,
                lora_train_only=False,
                colocate=True,
                offload_rollout=offload,
                check_weight_update_equal=checker,
            )
        )

    return create
