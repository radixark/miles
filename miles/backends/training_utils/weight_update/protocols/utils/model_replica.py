from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
from sglang.srt import server_args as server_args_module
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import ParallelismContext, RankParallelismConfig
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.model_loader import get_model
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.parameter_mapper import ParameterMapper
from sglang.srt.runtime_context import get_server_args
from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
)
from miles.utils import async_utils
from miles.utils.workers.argv_utils import _record_field_names

# where a rank sits in the launch, not how it holds its weights
_PLACEMENT_PARALLELISM_FIELDS = frozenset({"global_rank", "local_rank"})

# a multiple of every element size, so a span views as any param's dtype
_SPAN_ALIGNMENT_BYTES = 256


@dataclass(frozen=True)
class RolloutEngineRankConfig:
    """How one rollout engine rank holds its weights, as the engine reports it.

    Ranks with the same `shard_layout_key` take the same bytes, so one model replica serves them all.
    """

    parallelism: RankParallelismConfig
    server_args: ServerArgs

    @property
    def shard_layout_key(self) -> tuple[tuple[str, object], ...]:
        sharding = {
            f"parallelism.{name}": value
            for name, value in self.parallelism.to_dict().items()
            if name not in _PLACEMENT_PARALLELISM_FIELDS
        }
        server_args = {
            f"server_args.{name}": value for name, value in _replica_layout_server_args(self.server_args).items()
        }
        return tuple(sorted((sharding | server_args).items()))


class ParamSpec(NamedTuple):
    """One param as a rollout engine rank's loader writes into it, after `restore_weights_before_loading`."""

    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int

    @classmethod
    def of(cls, tensor: torch.Tensor) -> "ParamSpec":
        span_numel = (
            0
            if tensor.numel() == 0
            else 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
        )
        return cls(tensor.shape, tensor.stride(), tensor.dtype, span_numel * tensor.element_size())


class ModelReplica:
    """An sglang model in one rollout engine rank's layout, without parameter storage, that turns HF weights into
    the bytes that rank's loader would write.

    Its params are 0-size, and their shapes and attributes are those the engine's params have while it loads an
    update. The p2p protocol loads each group of ready params into a staging buffer with `load_into` and writes the
    returned bytes into the rollout engine ranks of this layout.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        built_shapes_by_name: Mapping[str, torch.Size],
        config: RolloutEngineRankConfig,
        *,
        postprocess_device: torch.device,
    ) -> None:
        # some models call it from their own load_weights; the engine runs it after the writes
        if hasattr(model, "post_load_weights"):
            model.post_load_weights = lambda *args, **kwargs: None
        self._model = model
        self._config = config
        self.param_specs = _bring_to_reload_state(model, built_shapes_by_name, config.parallelism, postprocess_device)
        self._params_by_name = dict(model.named_parameters())
        self.param_mapper = ParameterMapper.from_model(model)

    def load_into(
        self, buffer: torch.Tensor, param_names: Sequence[str], hf_tensors: list[tuple[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Loads `hf_tensors`, the HF tensors of exactly `param_names`, into the uint8 `buffer`; returns the bytes
        of each param in `buffer`, by name.

        `pack_into_buffers` gives groups of params that fit one buffer. Raises if loading changed a param in any way
        but its bytes, since a write carries only bytes to the rollout engine.
        """
        # sglang holds one live config per process, and this replica was built under its own
        if get_server_args() is not self._config.server_args:
            _publish_server_args(self._config.server_args)
        spans_by_name = _spans_in_buffer(buffer, param_names, self.param_specs)
        params_by_name = {name: self._params_by_name[name] for name in param_names}
        try:
            for name, param in params_by_name.items():
                spec = self.param_specs[name]
                param.data = torch.as_strided(spans_by_name[name].view(spec.dtype), spec.shape, spec.stride)
            metadata_before_by_name = {name: _param_metadata(param) for name, param in params_by_name.items()}
            with ParallelismContext(self._config.parallelism):
                self._model.load_weights(hf_tensors)
            params_after_by_name = dict(self._model.named_parameters())
            changed = [
                name
                for name, param in params_by_name.items()
                if params_after_by_name[name] is not param or _param_metadata(param) != metadata_before_by_name[name]
            ]
            assert not changed, (
                f"loading changed more than the bytes of {', '.join(changed[:5])} ({len(changed)} in all); the "
                "rollout engine receives only bytes, so its param would keep the old shape, storage or attributes"
            )
        finally:
            for param in params_by_name.values():
                param.data = torch.empty(0, dtype=param.dtype)
        return spans_by_name


class ModelReplicas:
    """Model replicas that load HF weights in rollout engine ranks' layouts, one per shard layout.

    The p2p protocol keeps one for the whole trainer process. The first replica's params, pinned, are the buffer
    every later replica loads into and every write reads from, so the buffer is registered once and stays valid
    across reconnects.
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
            shared = self.shared_params_dict[name]
            assert param.shape == shared.shape and param.dtype == shared.dtype, (
                f"[P2P-Shared] {name} is {tuple(param.shape)} {param.dtype} in the replica for "
                f"{config.parallelism} but {tuple(shared.shape)} {shared.dtype} in the shared buffer"
            )
            param.data = shared
        return model_replica


def query_rollout_engine_rank_configs(
    rollout_engines: Sequence[SGLangApiClient], assignments: Sequence[RolloutEngineRankAssignment]
) -> dict[int, RolloutEngineRankConfig]:
    """Returns the config of each rollout engine rank in `assignments`, by rollout engine rank.

    All rollout engines of one rank must hold it the same way, since one model replica serves them.
    """
    configs_by_rollout_engine_rank = {}
    for assignment in assignments:
        configs = [
            _query_config(rollout_engines[rollout_engine_ind], assignment.rollout_engine_rank)
            for rollout_engine_ind in assignment.rollout_engine_indices
        ]
        for rollout_engine_ind, config in zip(assignment.rollout_engine_indices, configs, strict=True):
            _assert_expert_placement_reproducible(config.server_args, f"rollout engine {rollout_engine_ind}")
        differing_fields = {
            name
            for config in configs[1:]
            for name, _ in set(config.shard_layout_key) ^ set(configs[0].shard_layout_key)
        }
        assert not differing_fields, (
            f"rollout engines {assignment.rollout_engine_indices} hold rank {assignment.rollout_engine_rank} in "
            f"different layouts, so one model replica cannot serve them: they differ in {sorted(differing_fields)}"
        )
        configs_by_rollout_engine_rank[assignment.rollout_engine_rank] = configs[0]
    return configs_by_rollout_engine_rank


def assert_replica_matches_shard(
    model_replica: torch.nn.Module, published_nbytes_by_name: Mapping[str, int], published_by: str
) -> None:
    """The replica must hold exactly the weights a rollout engine rank publishes, each in the published number of
    bytes; otherwise what it loads cannot be written into that rank's memory."""
    replica_nbytes_by_name = {
        name: param.numel() * param.element_size() for name, param in model_replica.named_parameters()
    }
    mismatches = [
        f"{name} is {replica_nbytes_by_name.get(name)} bytes here, {published_nbytes_by_name.get(name)} there"
        for name in sorted(replica_nbytes_by_name.keys() | published_nbytes_by_name.keys())
        if replica_nbytes_by_name.get(name) != published_nbytes_by_name.get(name)
    ]
    assert not mismatches, (
        f"the model replica does not match the weights {published_by} publishes: "
        f"{', '.join(mismatches[:5])} ({len(mismatches)} in all)"
    )


def build_model_replica(config: RolloutEngineRankConfig, model_path: str) -> ModelReplica:
    """Builds the model replica of `config`'s layout. `ModelReplicas` calls it once per layout; it uses this
    process's GPU for about one module's params, freed before it returns."""
    _publish_server_args(config.server_args)
    with ParallelismContext(config.parallelism):
        model, built_shapes_by_name = DefaultModelLoader(LoadConfig()).initialize_model_without_storage(
            model_config=ModelConfig.from_server_args(config.server_args, model_path=model_path),
            device=torch.device("cpu"),
        )
    return ModelReplica(
        model, built_shapes_by_name, config, postprocess_device=torch.device("cuda", torch.cuda.current_device())
    )


def pack_into_buffers(
    param_names: Iterable[str], param_specs: Mapping[str, ParamSpec], buffer_bytes: int
) -> Iterator[list[str]]:
    """Splits `param_names`, in order, into groups that `ModelReplica.load_into` can each load into one staging
    buffer of `buffer_bytes`."""
    group, group_end = [], 0
    for name in param_names:
        nbytes = param_specs[name].nbytes
        assert nbytes <= buffer_bytes, f"{name} takes {nbytes} bytes, more than a {buffer_bytes}-byte staging buffer"
        start = _align_span_start(group_end)
        if group and start + nbytes > buffer_bytes:
            yield group
            group, start = [], 0
        group.append(name)
        group_end = start + nbytes
    if group:
        yield group


def _replica_layout_server_args(server_args: ServerArgs) -> dict[str, object]:
    """The server args that change the bytes a model replica writes and may differ between the rollout engines of
    one model, by PD role or a server group's sglang overrides. The other args that shape a replica, such as the
    model, dtype and quantization, are the same for every rollout engine of a model."""
    names = (
        # sharding that RankParallelismConfig does not carry
        "enable_dp_lm_head",
        "moe_dense_tp_size",
        "dcp_size",
        # MoE structure: expert count, shared-expert fusion and its sharding
        "moe_a2a_backend",
        "moe_runner_backend",
        "disable_shared_experts_fusion",
        "enforce_shared_experts_fusion",
        "enable_two_batch_overlap",
        "enable_single_batch_overlap",
        "enable_waterfill",
        "disable_flashinfer_cutlass_moe_fp4_allgather",
        # layouts postprocess leaves in the reload state
        "fp8_gemm_runner_backend",
        "fp4_gemm_runner_backend",
        "flashinfer_mxfp4_moe_precision",
        "enable_w4a4_mxfp4_megamoe",
        "flashinfer_a2a_dispatch_type",
    )
    return {name: getattr(server_args, name) for name in names}


def _assert_expert_placement_reproducible(server_args: ServerArgs, rollout_engine: str) -> None:
    # the engine places these experts by its expert-location metadata, runtime rebalancing or CPU offload; a model
    # replica loads every expert into its default slot
    placement_fields = [
        name
        for name, is_set in (
            ("ep_num_redundant_experts", server_args.ep_num_redundant_experts != 0),
            ("init_expert_location", server_args.init_expert_location != "trivial"),
            ("enable_eplb", server_args.enable_eplb),
            ("ep_join_mode", server_args.ep_join_mode is not None),
            ("elastic_ep_initial_size", server_args.elastic_ep_initial_size is not None),
            ("dwdp_size", server_args.dwdp_size != 1),
            ("kt_weight_path", server_args.kt_weight_path is not None),
        )
        if is_set
    ]
    assert not placement_fields, (
        f"{rollout_engine} places experts by {', '.join(placement_fields)}, which a model replica does not "
        "reproduce, so p2p would write experts into the wrong slots. Update its weights with another "
        "--update-weight-transfer-mode."
    )


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
    _publish_server_args(config.server_args)

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


def _publish_server_args(server_args: ServerArgs) -> None:
    # model construction and quant methods read these process-wide settings
    server_args_module.set_global_server_args_for_scheduler(server_args)
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()


def _bring_to_reload_state(
    model: torch.nn.Module,
    built_shapes_by_name: Mapping[str, torch.Size],
    parallelism: RankParallelismConfig,
    postprocess_device: torch.device,
) -> dict[str, ParamSpec]:
    """Runs the engine's startup postprocess and session restore on `model`, one module at a time on zero tensors,
    and returns the spec of each param afterwards, by name. Every param ends 0-size on the CPU."""
    built_shapes_by_param_id = {id(param): built_shapes_by_name[name] for name, param in model.named_parameters()}
    reload_specs_by_param_id = {}
    for module_name, module in model.named_modules():
        # modules without one keep their params as built
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            continue
        for param in module.parameters(recurse=False):
            param.data = torch.zeros(built_shapes_by_param_id[id(param)], dtype=param.dtype, device=postprocess_device)
        with ParallelismContext(parallelism):
            quant_method.process_weights_after_loading(module)
            # as the engine does: duck-typed quant methods have no restore
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.restore_weights_before_loading(module)
        # one entry per Parameter, though postprocess may register one under two names
        for param_name, param in module.named_parameters(recurse=False):
            assert param.device.type != "meta", (
                f"the postprocess of {module_name} left {param_name} on meta; it must have read a tensor that "
                "construction without storage does not allocate"
            )
            reload_specs_by_param_id[id(param)] = ParamSpec.of(param)
            param.data = torch.empty(0, dtype=param.dtype)

    param_specs = {}
    for name, param in model.named_parameters():
        if id(param) in reload_specs_by_param_id:
            param_specs[name] = reload_specs_by_param_id[id(param)]
        else:
            param_specs[name] = ParamSpec.of(torch.empty(built_shapes_by_name[name], dtype=param.dtype, device="meta"))
    return param_specs


def _align_span_start(offset: int) -> int:
    return -(-offset // _SPAN_ALIGNMENT_BYTES) * _SPAN_ALIGNMENT_BYTES


def _spans_in_buffer(
    buffer: torch.Tensor, param_names: Sequence[str], param_specs: Mapping[str, ParamSpec]
) -> dict[str, torch.Tensor]:
    spans_by_name, end = {}, 0
    for name in param_names:
        start = _align_span_start(end)
        end = start + param_specs[name].nbytes
        assert end <= buffer.numel(), (
            f"{', '.join(param_names)} do not fit a {buffer.numel()}-byte staging buffer; group them with "
            "pack_into_buffers"
        )
        spans_by_name[name] = buffer[start:end]
    return spans_by_name


def _param_metadata(param: torch.nn.Parameter) -> tuple:
    attributes = {
        key: value if isinstance(value, bool | int | float | str | None) else id(value)
        for key, value in vars(param).items()
    }
    return param.shape, param.stride(), param.dtype, param.data_ptr(), attributes
