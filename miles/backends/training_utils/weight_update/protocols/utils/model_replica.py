from collections.abc import Sequence
from dataclasses import dataclass

import torch
from sglang.srt import server_args as server_args_module
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import ParallelismContext, RankParallelismConfig
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.model_loader import get_model
from sglang.srt.model_loader.parameter_mapper import ParameterMapper
from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
)
from miles.utils import async_utils
from miles.utils.workers.argv_utils import _record_field_names

# where a rank sits in the launch, not how it holds its weights
_PLACEMENT_PARALLELISM_FIELDS = frozenset({"global_rank", "local_rank"})


@dataclass(frozen=True)
class RolloutEngineRankConfig:
    """How one rollout engine rank holds its weights, as the engine reports it.

    Ranks with the same `shard_layout_key` take the same bytes, so one model replica serves them all.
    """

    parallelism: RankParallelismConfig
    server_args: ServerArgs

    @property
    def shard_layout_key(self) -> tuple:
        sharding = {
            name: value
            for name, value in self.parallelism.to_dict().items()
            if name not in _PLACEMENT_PARALLELISM_FIELDS
        }
        return tuple(sorted(sharding.items())), self.server_args.quantization


class ModelReplicas:
    """Model replicas that load HF weights in rollout engine ranks' layouts, one per shard layout.

    The p2p protocol builds one at each connect. The first replica's params, pinned, are the buffer every later
    replica loads into and every write reads from.
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model_replicas_by_shard_layout_key: dict[tuple, torch.nn.Module] = {}
        self.shared_params_dict: dict[str, torch.Tensor] = {}
        self.param_mapper: ParameterMapper | None = None

    def get_or_build(self, config: RolloutEngineRankConfig) -> torch.nn.Module:
        if config.shard_layout_key not in self._model_replicas_by_shard_layout_key:
            self._model_replicas_by_shard_layout_key[config.shard_layout_key] = self._build(config)
        return self._model_replicas_by_shard_layout_key[config.shard_layout_key]

    def _build(self, config: RolloutEngineRankConfig) -> torch.nn.Module:
        model_replica = _build_cpu_replica(config, self._model_path)
        if not self.shared_params_dict:
            for param in model_replica.parameters():
                param.data = param.data.pin_memory()
            self.shared_params_dict = dict(model_replica.named_parameters())
            self.param_mapper = ParameterMapper.from_model(model_replica)
            return model_replica

        for name, param in model_replica.named_parameters():
            assert name in self.shared_params_dict, f"[P2P-Shared] Parameter {name} not found in shared buffers"
            param.data = self.shared_params_dict[name]
        return model_replica


def query_rollout_engine_rank_configs(
    rollout_engines: Sequence[SGLangApiClient], assignments: Sequence[RolloutEngineRankAssignment]
) -> dict[int, RolloutEngineRankConfig]:
    """Returns the config of each rollout engine rank in `assignments`, by rollout engine rank, as its first engine
    reports it."""
    return {
        assignment.rollout_engine_rank: _query_config(
            rollout_engines[assignment.rollout_engine_indices[0]], assignment.rollout_engine_rank
        )
        for assignment in assignments
    }


def create_server_args_from_dict(data_dict: dict) -> ServerArgs:
    valid_fields = set(_record_field_names(ServerArgs))
    filtered_data = {k: v for k, v in data_dict.items() if k in valid_fields}
    return ServerArgs(**filtered_data)


def _query_config(rollout_engine: SGLangApiClient, rollout_engine_rank: int) -> RolloutEngineRankConfig:
    parallelism_info = async_utils.run(rollout_engine.get_parallelism_info(rank=rollout_engine_rank))
    server_info = async_utils.run(rollout_engine.get_server_info())
    return RolloutEngineRankConfig(
        parallelism=RankParallelismConfig.from_dict(parallelism_info),
        server_args=create_server_args_from_dict(server_info),
    )


def _build_cpu_replica(config: RolloutEngineRankConfig, model_path: str) -> torch.nn.Module:
    """Create a CPU model replica that loads the right shard and skips post_load_weights."""
    load_config = LoadConfig(
        load_format="dummy",
        model_loader_extra_config=None,
        rl_quant_profile=config.server_args.rl_quant_profile,
    )
    server_args_module.set_global_server_args_for_scheduler(config.server_args)
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()

    # Monkey-patch the loader-level post_load_weights to no-op BEFORE get_model,
    # because get_model() calls post_load_weights() internally (loader.py:1310)
    # which may invoke CUDA-only kernels (e.g., per_tensor_quant_fp8 for FP8 models).
    # This is safe because the rollout engine runs post_load_weights on its own GPU
    # after RDMA transfer, at end_weight_update.
    from sglang.srt.model_loader import loader as model_loader_module

    original_post_load_weights = model_loader_module.post_load_weights
    model_loader_module.post_load_weights = lambda *args, **kwargs: None
    try:
        with ParallelismContext(config.parallelism):
            model = get_model(
                model_config=ModelConfig(model_path),
                load_config=load_config,
                device_config=DeviceConfig(device="cpu"),
            )
    finally:
        model_loader_module.post_load_weights = original_post_load_weights

    # Also patch the instance method for subsequent load_weights() calls
    # (deepseek_weight_loader.py:342 calls self.post_load_weights() at the end).
    if hasattr(model, "post_load_weights"):
        model.post_load_weights = lambda *args, **kwargs: None

    return model
