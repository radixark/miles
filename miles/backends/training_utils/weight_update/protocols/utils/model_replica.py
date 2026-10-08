import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import NamedTuple

import torch
from sglang.srt import server_args as server_args_module
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import ParallelismContext, RankParallelismConfig
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.moe.utils import (
    draft_model_build_scope,
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_server_args
from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols.utils.loader_probe import (
    HfNameMapping,
    HfTensorSpec,
    probe_loader_writes,
)
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
)
from miles.utils import async_utils
from miles.utils.workers.argv_utils import _record_field_names

logger = logging.getLogger(__name__)

# where a rank sits in the launch, not how it holds its weights
_PLACEMENT_PARALLELISM_FIELDS = frozenset({"global_rank", "local_rank"})

# a multiple of every element size, so the bytes of any param view as its dtype
_PARAM_ALIGNMENT_BYTES = 256

_SPECULATIVE_ALGORITHMS_WITHOUT_DRAFT_MODEL = frozenset({"NGRAM", "UNO"})


@dataclass(frozen=True)
class RolloutEngineRankConfig:
    """Reported weight layout for one target or draft runner at a rollout rank.

    Configs with the same `shard_layout_key` take the same bytes, so one model replica serves them all.
    """

    runner_role: str
    parallelism: RankParallelismConfig
    server_args: ServerArgs

    @property
    def shard_layout_key(self) -> tuple[tuple[str, object], ...]:
        sharding_fields = {
            f"parallelism.{name}": value
            for name, value in self.parallelism.to_dict().items()
            if name not in _PLACEMENT_PARALLELISM_FIELDS
        }
        server_args_fields = {
            f"server_args.{name}": value for name, value in _get_shard_layout_server_args(self.server_args).items()
        }
        return tuple(sorted((sharding_fields | server_args_fields | {"runner_role": self.runner_role}).items()))


def _get_shard_layout_server_args(server_args: ServerArgs) -> dict[str, object]:
    """The server args that change the bytes a model replica writes and may differ between the rollout engines of
    one model, by PD role or a server group's sglang overrides. The other args that shape a replica, such as the
    model path and its config overrides, are the same for every rollout engine of a model."""
    field_names = (
        # precision: a server group may quantize the same weights its own way
        "quantization",
        "dtype",
        # sharding that RankParallelismConfig does not carry
        "enable_dp_lm_head",
        "moe_dense_tp_size",
        "dcp_size",
        # MoE structure: expert count, shared-expert fusion and its sharding
        "moe_a2a_backend",
        "moe_runner_backend",
        "speculative_moe_a2a_backend",
        "speculative_moe_runner_backend",
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
    return {name: getattr(server_args, name) for name in field_names}


def query_rollout_engine_rank_configs(
    rollout_engines: Sequence[SGLangApiClient],
    assignments: Sequence[RolloutEngineRankAssignment],
    runner_role: str,
) -> dict[int, RolloutEngineRankConfig]:
    """Query `runner_role` configs by rollout rank, requiring matching layouts across engines.

    All rollout engines of one rank must hold it the same way, since one model replica serves them.
    """
    configs_by_rollout_engine_rank = {}
    for assignment in assignments:
        configs = [
            _query_config(rollout_engines[rollout_engine_ind], assignment.rollout_engine_rank, runner_role)
            for rollout_engine_ind in assignment.rollout_engine_indices
        ]
        for rollout_engine_ind, config in zip(assignment.rollout_engine_indices, configs, strict=True):
            _assert_expert_placement_reproducible(config.server_args, rollout_engine_ind)
        differing_fields = {
            name
            for config in configs[1:]
            for name, _ in set(config.shard_layout_key) ^ set(configs[0].shard_layout_key)
        }
        assert not differing_fields, (
            f"rollout engines {assignment.rollout_engine_indices} hold the {runner_role} of rank "
            f"{assignment.rollout_engine_rank} in different layouts, so one model replica cannot serve them: they "
            f"differ in {sorted(differing_fields)}"
        )
        configs_by_rollout_engine_rank[assignment.rollout_engine_rank] = configs[0]
    return configs_by_rollout_engine_rank


def _query_config(
    rollout_engine: SGLangApiClient, rollout_engine_rank: int, runner_role: str
) -> RolloutEngineRankConfig:
    parallelism_info = async_utils.run(rollout_engine.get_parallelism_info(rank=rollout_engine_rank, role=runner_role))
    return RolloutEngineRankConfig(
        runner_role=runner_role,
        parallelism=RankParallelismConfig.from_dict(parallelism_info),
        server_args=_query_server_args(rollout_engine),
    )


def _query_server_args(rollout_engine: SGLangApiClient) -> ServerArgs:
    return create_server_args_from_dict(async_utils.run(rollout_engine.get_server_info()))


def create_server_args_from_dict(data_dict: dict) -> ServerArgs:
    valid_fields = set(_record_field_names(ServerArgs))
    filtered_data = {k: v for k, v in data_dict.items() if k in valid_fields}
    return ServerArgs(**filtered_data)


def _assert_expert_placement_reproducible(server_args: ServerArgs, rollout_engine_ind: int) -> None:
    # the engine places these experts by its expert-location metadata, runtime rebalancing or CPU offload; a model
    # replica loads every expert into its default slot
    unreproducible_fields = [
        name
        for name, is_in_use in (
            ("ep_num_redundant_experts", server_args.ep_num_redundant_experts != 0),
            ("init_expert_location", server_args.init_expert_location != "trivial"),
            ("enable_eplb", server_args.enable_eplb),
            ("ep_join_mode", server_args.ep_join_mode is not None),
            ("elastic_ep_initial_size", server_args.elastic_ep_initial_size is not None),
            ("dwdp_size", server_args.dwdp_size != 1),
            ("kt_weight_path", server_args.kt_weight_path is not None),
        )
        if is_in_use
    ]
    assert not unreproducible_fields, (
        f"rollout engine {rollout_engine_ind} places experts by {', '.join(unreproducible_fields)}, which a model "
        "replica does not reproduce, so p2p would write experts into the wrong slots. Update its weights with "
        "another --update-weight-transfer-mode."
    )


def query_runner_roles(
    rollout_engines: Sequence[SGLangApiClient], assignments: Sequence[RolloutEngineRankAssignment], selector: str
) -> tuple[str, ...]:
    """Return target-first update roles, requiring the same roles on every assigned engine."""
    distinct_runner_roles = {
        _select_runner_roles(_query_server_args(rollout_engines[rollout_engine_ind]), selector)
        for assignment in assignments
        for rollout_engine_ind in assignment.rollout_engine_indices
    }
    assert len(distinct_runner_roles) == 1, (
        f"the rollout engines run different model runners {sorted(distinct_runner_roles)}, but one update writes the "
        "same runners on each; a draft would go unwritten on some engines or be queried on engines without one"
    )
    (runner_roles,) = distinct_runner_roles
    return runner_roles


def _select_runner_roles(server_args: ServerArgs, selector: str) -> tuple[str, ...]:
    if selector != "all" or server_args.speculative_algorithm in (None, *_SPECULATIVE_ALGORITHMS_WITHOUT_DRAFT_MODEL):
        return ("target",)
    _assert_draft_is_target_mtp(server_args)
    return ("target", "draft")


def _assert_draft_is_target_mtp(server_args: ServerArgs) -> None:
    if server_args.speculative_algorithm != "EAGLE" or server_args.speculative_draft_model_path not in (
        None,
        server_args.model_path,
    ):
        raise NotImplementedError(
            f"the rollout engines draft with {server_args.speculative_algorithm} from "
            f"{server_args.speculative_draft_model_path}, but p2p updates only a draft that is the target model's own "
            "MTP layer (EAGLE)"
        )
    if server_args.enable_multi_layer_eagle:
        raise NotImplementedError(
            "multi-layer EAGLE runs a draft runner per MTP layer; p2p updates one draft runner per rollout engine rank"
        )


class TransferBufferParamLayout(NamedTuple):
    """How one param's bytes are laid out in a transfer buffer: as the engine's param is after
    `restore_weights_before_loading`, which is what its loader writes into."""

    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    occupied_nbytes: int

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TransferBufferParamLayout":
        return cls(tensor.shape, tensor.stride(), tensor.dtype, _compute_occupied_nbytes(tensor))


def _compute_occupied_nbytes(tensor: torch.Tensor) -> int:
    """The bytes a tensor's elements cover in memory, from its first element to its last; `numel × element_size`
    when it is contiguous, more when its strides leave gaps."""
    if tensor.numel() == 0:
        return 0
    last_element_offset = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
    return (last_element_offset + 1) * tensor.element_size()


class ModelReplica:
    """Convert HF weights into one rollout rank's reload layout using its sglang loader.

    Parameters retain their reload metadata but have storage only while bound to a transfer buffer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        built_shapes_by_name: Mapping[str, torch.Size],
        config: RolloutEngineRankConfig,
        *,
        postprocess_device: torch.device,
        transfer_buffer_device: torch.device,
    ) -> None:
        # some models call it from their own load_weights; the engine runs it after the writes
        if hasattr(model, "post_load_weights"):
            model.post_load_weights = lambda *args, **kwargs: None
        self._model = model
        self._config = config
        # the trainer's HF tensors are on the GPU postprocess runs on
        self._hf_tensor_device = postprocess_device
        self._transfer_buffer_device = transfer_buffer_device
        self.transfer_buffer_param_layouts = _bring_to_reload_state(
            model, built_shapes_by_name, config.parallelism, postprocess_device
        )
        self._params_by_name = dict(model.named_parameters())

    def map_hf_names(self, hf_tensor_specs: Mapping[str, HfTensorSpec]) -> HfNameMapping:
        """Probe HF-to-parameter dependencies and allocate scratch storage for buffers the loader writes."""
        if get_server_args() is not self._config.server_args:
            _publish_server_args(self._config.server_args)
        with ParallelismContext(self._config.parallelism):
            loader_writes = probe_loader_writes(
                self._model, self.transfer_buffer_param_layouts, hf_tensor_specs, device=self._hf_tensor_device
            )
        if loader_writes.buffer_names:
            logger.info(
                f"the {self._config.runner_role} loader also fills {len(loader_writes.buffer_names)} buffers, e.g. "
                f"{sorted(loader_writes.buffer_names)[:3]}; p2p writes only params, so the rollout engine derives "
                "these itself"
            )
        _give_buffers_scratch_storage(self._model, loader_writes.buffer_names, self._transfer_buffer_device)
        return loader_writes.hf_name_mapping

    def load_into(
        self, buffer: torch.Tensor, param_names: Sequence[str], hf_tensors: list[tuple[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Loads `hf_tensors`, the HF tensors of exactly `param_names`, into the uint8 `buffer`; returns the bytes
        of each param in `buffer`, by name.

        `pack_into_buffers` gives groups of params that fit one buffer. Raises if loading changed a param in any way
        but its bytes, since a write carries only bytes to the rollout engine.
        """
        # sglang keeps one live config per process; load under the one this replica was built with
        if get_server_args() is not self._config.server_args:
            _publish_server_args(self._config.server_args)
        param_bytes_by_name = _slice_buffer_by_param(buffer, param_names, self.transfer_buffer_param_layouts)
        with self._bind_params_to_bytes(param_bytes_by_name) as bound_params_by_name:
            state_before_load_by_name = {
                name: _get_param_state_besides_bytes(param) for name, param in bound_params_by_name.items()
            }
            with ParallelismContext(self._config.parallelism):
                self._model.load_weights(hf_tensors)
            self._assert_params_changed_only_bytes(bound_params_by_name, state_before_load_by_name)
        return param_bytes_by_name

    @contextmanager
    def _bind_params_to_bytes(
        self, param_bytes_by_name: Mapping[str, torch.Tensor]
    ) -> Iterator[dict[str, torch.nn.Parameter]]:
        bound_params_by_name = {name: self._params_by_name[name] for name in param_bytes_by_name}
        try:
            for name, param in bound_params_by_name.items():
                param.data = _view_bytes_as_param(param_bytes_by_name[name], self.transfer_buffer_param_layouts[name])
            yield bound_params_by_name
        finally:
            for param in bound_params_by_name.values():
                param.data = torch.empty(0, dtype=param.dtype)

    def _assert_params_changed_only_bytes(
        self, bound_params_by_name: Mapping[str, torch.nn.Parameter], state_before_load_by_name: Mapping[str, tuple]
    ) -> None:
        params_after_load_by_name = dict(self._model.named_parameters())
        changed_param_names = [
            name
            for name, param in bound_params_by_name.items()
            if params_after_load_by_name[name] is not param
            or _get_param_state_besides_bytes(param) != state_before_load_by_name[name]
        ]
        assert not changed_param_names, (
            f"loading changed more than the bytes of {', '.join(changed_param_names[:5])} "
            f"({len(changed_param_names)} in all); the rollout engine receives only bytes, so its param would keep "
            "the old shape, storage or attributes"
        )


def _give_buffers_scratch_storage(model: torch.nn.Module, buffer_names: frozenset[str], device: torch.device) -> None:
    # preserve aliases for buffers registered under multiple names
    meta_buffer_ids = {id(buffer) for name, buffer in model.named_buffers() if name in buffer_names and buffer.is_meta}
    scratch_buffers_by_id = {}
    for module in model.modules():
        for local_name, buffer in module._buffers.items():
            if buffer is None or id(buffer) not in meta_buffer_ids:
                continue
            if id(buffer) not in scratch_buffers_by_id:
                scratch_buffers_by_id[id(buffer)] = torch.empty_strided(
                    buffer.shape, buffer.stride(), dtype=buffer.dtype, device=device
                )
            module._buffers[local_name] = scratch_buffers_by_id[id(buffer)]


def _bring_to_reload_state(
    model: torch.nn.Module,
    built_shapes_by_name: Mapping[str, torch.Size],
    parallelism: RankParallelismConfig,
    postprocess_device: torch.device,
) -> dict[str, TransferBufferParamLayout]:
    """Runs the engine's startup postprocess and session restore on `model`, one module at a time on zero tensors,
    and returns the transfer buffer layout of each param afterwards, by name. Every param ends 0-size on the CPU."""
    built_shapes_by_param_id = {id(param): built_shapes_by_name[name] for name, param in model.named_parameters()}
    reload_layouts_by_param_id = {}
    for module_name, module in model.named_modules():
        # modules without one keep their params as built
        if getattr(module, "quant_method", None) is not None:
            reload_layouts_by_param_id |= _bring_module_to_reload_state(
                module_name, module, built_shapes_by_param_id, parallelism, postprocess_device
            )

    param_layouts = {}
    for name, param in model.named_parameters():
        if id(param) in reload_layouts_by_param_id:
            param_layouts[name] = reload_layouts_by_param_id[id(param)]
        else:
            built_param = torch.empty(built_shapes_by_name[name], dtype=param.dtype, device="meta")
            param_layouts[name] = TransferBufferParamLayout.from_tensor(built_param)
    return param_layouts


def _bring_module_to_reload_state(
    module_name: str,
    module: torch.nn.Module,
    built_shapes_by_param_id: Mapping[int, torch.Size],
    parallelism: RankParallelismConfig,
    postprocess_device: torch.device,
) -> dict[int, TransferBufferParamLayout]:
    for param in module.parameters(recurse=False):
        param.data = torch.zeros(built_shapes_by_param_id[id(param)], dtype=param.dtype, device=postprocess_device)
    with ParallelismContext(parallelism):
        module.quant_method.process_weights_after_loading(module)
        # as the engine does: duck-typed quant methods have no restore
        if isinstance(module.quant_method, QuantizeMethodBase):
            module.quant_method.restore_weights_before_loading(module)

    reload_layouts_by_param_id = {}
    # one entry per Parameter, though postprocess may register one under two names
    for param_name, param in module.named_parameters(recurse=False):
        assert param.device.type != "meta", (
            f"the postprocess of {module_name} left {param_name} on meta; it must have read a tensor that "
            "construction without storage does not allocate"
        )
        reload_layouts_by_param_id[id(param)] = TransferBufferParamLayout.from_tensor(param)
        param.data = torch.empty(0, dtype=param.dtype)
    return reload_layouts_by_param_id


def _slice_buffer_by_param(
    buffer: torch.Tensor, param_names: Sequence[str], param_layouts: Mapping[str, TransferBufferParamLayout]
) -> dict[str, torch.Tensor]:
    """Lays `param_names` out one after another in the uint8 `buffer`, each start aligned; returns the bytes each
    param occupies, by name."""
    param_bytes_by_name, param_end_offset = {}, 0
    for name in param_names:
        param_start_offset = _align_param_start(param_end_offset)
        param_end_offset = param_start_offset + param_layouts[name].occupied_nbytes
        assert param_end_offset <= buffer.numel(), (
            f"{', '.join(param_names)} do not fit a {buffer.numel()}-byte transfer buffer; group them with "
            "pack_into_buffers"
        )
        param_bytes_by_name[name] = buffer[param_start_offset:param_end_offset]
    return param_bytes_by_name


def _align_param_start(offset: int) -> int:
    return -(-offset // _PARAM_ALIGNMENT_BYTES) * _PARAM_ALIGNMENT_BYTES


def _view_bytes_as_param(param_bytes: torch.Tensor, param_layout: TransferBufferParamLayout) -> torch.Tensor:
    return torch.as_strided(param_bytes.view(param_layout.dtype), param_layout.shape, param_layout.stride)


def _get_param_state_besides_bytes(param: torch.nn.Parameter) -> tuple:
    attributes = {
        key: value if isinstance(value, bool | int | float | str | None) else id(value)
        for key, value in vars(param).items()
    }
    return param.shape, param.stride(), param.dtype, param.data_ptr(), attributes


def build_model_replica(
    config: RolloutEngineRankConfig, model_path: str, *, transfer_buffer_device: torch.device
) -> ModelReplica:
    """Builds the model replica of `config`'s layout. The build holds about one module's params at a time on this
    process's GPU and frees them before it returns."""
    _publish_server_args(config.server_args)
    is_draft = config.runner_role == "draft"
    with _runner_build_context(is_draft):
        with ParallelismContext(config.parallelism):
            model, built_shapes_by_name = DefaultModelLoader(LoadConfig()).initialize_model_without_storage(
                model_config=ModelConfig.from_server_args(
                    config.server_args, model_path=model_path, is_draft_model=is_draft
                ),
                device=torch.device("cpu"),
            )
        return ModelReplica(
            model,
            built_shapes_by_name,
            config,
            postprocess_device=torch.device("cuda", torch.cuda.current_device()),
            transfer_buffer_device=transfer_buffer_device,
        )


@contextmanager
def _runner_build_context(is_draft: bool) -> Iterator[None]:
    # as EAGLEWorkerV2 builds its draft: the speculative MoE backends and the draft's shared-experts fusion
    with ExitStack() as stack:
        if is_draft:
            stack.enter_context(speculative_moe_backend_context())
            stack.enter_context(speculative_moe_a2a_backend_context())
            stack.enter_context(draft_model_build_scope())
        yield


def _publish_server_args(server_args: ServerArgs) -> None:
    # model construction and quant methods read these process-wide settings
    server_args_module.set_global_server_args_for_scheduler(server_args)
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()


def pack_into_buffers(
    param_groups: Iterable[tuple[str, ...]],
    param_layouts: Mapping[str, TransferBufferParamLayout],
    buffer_nbytes: int,
) -> Iterator[list[tuple[str, ...]]]:
    """Pack complete parameter groups into buffers, preserving order and parameter alignment."""
    packed_groups, end_offset = [], 0
    for param_group in param_groups:
        group_nbytes = compute_param_group_nbytes(param_group, param_layouts)
        assert group_nbytes <= buffer_nbytes, f"{param_group} need {group_nbytes} bytes, a buffer has {buffer_nbytes}"
        group_end_offset = _end_offset_after(param_group, param_layouts, start_offset=end_offset)
        if packed_groups and group_end_offset > buffer_nbytes:
            yield packed_groups
            packed_groups, group_end_offset = [], group_nbytes
        packed_groups.append(param_group)
        end_offset = group_end_offset
    if packed_groups:
        yield packed_groups


def compute_param_group_nbytes(
    param_group: Iterable[str], param_layouts: Mapping[str, TransferBufferParamLayout]
) -> int:
    """Return the group's byte size, including alignment between parameters."""
    return _end_offset_after(param_group, param_layouts, start_offset=0)


def _end_offset_after(
    param_names: Iterable[str], param_layouts: Mapping[str, TransferBufferParamLayout], start_offset: int
) -> int:
    end_offset = start_offset
    for name in param_names:
        end_offset = _align_param_start(end_offset) + param_layouts[name].occupied_nbytes
    return end_offset


class ModelReplicas:
    """Cache one runner role's replicas and HF mappings by shard layout for the trainer process's lifetime.

    All replicas must share a transfer-buffer layout so the sender can reuse one packing across ranks.
    """

    def __init__(self, model_path: str, transfer_buffer_device: torch.device) -> None:
        self._model_path = model_path
        self._transfer_buffer_device = transfer_buffer_device
        self._model_replicas_by_shard_layout_key: dict[tuple, ModelReplica] = {}
        self._mapped_shard_layout_keys: set[tuple] = set()
        self.transfer_buffer_param_layouts: dict[str, TransferBufferParamLayout] = {}
        self.hf_name_mapping: HfNameMapping | None = None

    def get_or_build(self, config: RolloutEngineRankConfig) -> ModelReplica:
        if config.shard_layout_key not in self._model_replicas_by_shard_layout_key:
            self._model_replicas_by_shard_layout_key[config.shard_layout_key] = self._build(config)
        return self._model_replicas_by_shard_layout_key[config.shard_layout_key]

    def _build(self, config: RolloutEngineRankConfig) -> ModelReplica:
        model_replica = build_model_replica(
            config, self._model_path, transfer_buffer_device=self._transfer_buffer_device
        )
        if not self.transfer_buffer_param_layouts:
            self.transfer_buffer_param_layouts = model_replica.transfer_buffer_param_layouts
        differing_param_names = [
            name
            for name in sorted(
                self.transfer_buffer_param_layouts.keys() | model_replica.transfer_buffer_param_layouts.keys()
            )
            if self.transfer_buffer_param_layouts.get(name) != model_replica.transfer_buffer_param_layouts.get(name)
        ]
        assert not differing_param_names, (
            f"the model replica for {config.parallelism} lays out {', '.join(differing_param_names[:5])} "
            f"({len(differing_param_names)} in all) differently from this sender's other replicas, but one packing of "
            "ready params serves all of them"
        )
        return model_replica

    def map_hf_names(self, hf_tensor_specs: Mapping[str, HfTensorSpec]) -> None:
        """Merge new replicas' mappings; expert-parallel ranks load different subsets of HF tensors."""
        for shard_layout_key, model_replica in self._model_replicas_by_shard_layout_key.items():
            if shard_layout_key in self._mapped_shard_layout_keys:
                continue
            hf_name_mapping = model_replica.map_hf_names(hf_tensor_specs)
            self.hf_name_mapping = (
                hf_name_mapping if self.hf_name_mapping is None else self.hf_name_mapping.union(hf_name_mapping)
            )
            self._mapped_shard_layout_keys.add(shard_layout_key)


def assert_replica_matches_shard(
    model_replica: ModelReplica, published_nbytes_by_name: Mapping[str, int], published_by: str
) -> None:
    """The replica must hold exactly the weights a rollout engine rank publishes, each in the published number of
    bytes; otherwise what it loads cannot be written into that rank's memory."""
    replica_nbytes_by_name = {
        name: layout.occupied_nbytes for name, layout in model_replica.transfer_buffer_param_layouts.items()
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
