from typing import Any

from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig


def make_sglang_config(**base_args: Any) -> SglangConfig:
    group = ServerGroupConfig(
        worker_type=WorkerType.REGULAR,
        num_gpus=1,
        num_gpus_per_engine=1,
        gpu_offset=0,
        engine_offset=0,
        needs_offload=False,
    )
    model = ModelConfig(name="default", model_path=None, server_groups=[group], update_weights=True)
    return SglangConfig(models=[model], base_args=base_args)
