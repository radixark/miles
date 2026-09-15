import functools
import glob
import json
import os
from collections.abc import Iterator

import safetensors
import torch
import torch.distributed as dist

from miles.backends.fsdp_utils.dtensor import gather_full_param
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups
from miles.backends.training_utils.weight_update.hf_weight_iterator.checkpoint_towers import (
    iter_checkpoint_tower_units,
)
from miles.utils.hf_config import load_hf_config


class TitanHfWeightIterator(HfWeightIteratorBase):
    forced_placement = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._engine_dtypes = _checkpoint_dtypes(self.args.hf_checkpoint)

    def _iter_hf_param_units(self, weights, *, materialize):
        for name, tensor in hf_weights(self.model, complete_across_pp=self.placement.gather_pp):
            if materialize:
                yield [(name, self._to_engine_dtype(name, tensor))]
        yield from iter_checkpoint_tower_units(self.args.hf_checkpoint, materialize=materialize)

    def _to_engine_dtype(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        target = self._engine_dtypes.get(name)
        if target is None or tensor.dtype == target or not tensor.is_floating_point():
            return tensor
        return tensor.to(target)

    def _hf_atomic_update_groups(self):
        q_lora_rank = getattr(load_hf_config(self.args.hf_checkpoint), "q_lora_rank", None) or None
        return get_hf_atomic_update_groups(self.model_name, q_lora_rank=q_lora_rank)

    def _iter_hf_adapter_units(self, lora_name, adapter, *, materialize):
        raise NotImplementedError("the torchtitan backend has no LoRA")


def hf_weights(trainer, *, complete_across_pp: bool = True) -> Iterator[tuple[str, torch.Tensor]]:
    offloaded = next(trainer.model_parts[0].parameters()).device.type == "cpu"
    if offloaded:
        for part in trainer.model_parts:
            part.cuda()
    try:
        yield from _hf_weights_on_device(trainer, complete_across_pp=complete_across_pp)
    finally:
        if offloaded:
            for part in trainer.model_parts:
                part.cpu()
            torch.cuda.empty_cache()


@functools.cache
def _stage_process_groups(stage_ranks: tuple[tuple[int, ...], ...]) -> tuple:
    return tuple(dist.new_group(list(ranks)) for ranks in stage_ranks)


_GROUPED_EXPERTS = "moe.routed_experts.inner_experts"


def _hf_weights_on_device(trainer, *, complete_across_pp: bool) -> Iterator[tuple[str, torch.Tensor]]:
    sd_adapter = getattr(trainer.checkpointer, "sd_adapter", None)
    if sd_adapter is None:
        sd_adapter = trainer.config.model_spec.state_dict_adapter(trainer.model_config, trainer.config.hf_assets_path)
    state = {k: v for part in trainer.model_parts for k, v in part.state_dict().items()}
    layout = _StageLayout.gather(trainer, complete_across_pp=complete_across_pp)

    dense = {k: v for k, v in state.items() if _GROUPED_EXPERTS not in k}
    yield from layout.stream(sd_adapter.to_hf(dense))

    mine = sorted(k for k in state if _GROUPED_EXPERTS in k)
    everyone: list = [None] * layout.world
    dist.all_gather_object(everyone, mine)
    for key in sorted(set().union(*everyone)):
        local = sd_adapter.to_hf({key: gather_full_param(state[key])}) if key in state else {}
        yield from layout.stream(local)


class _StageLayout:
    def __init__(self, *, world: int, my_rank: int, device, audience: list[int], broadcast_group):
        self.world = world
        self.my_rank = my_rank
        self.device = device
        self.audience = set(audience)
        self.broadcast_group = broadcast_group

    @classmethod
    def gather(cls, trainer, *, complete_across_pp: bool) -> "_StageLayout":
        world = dist.get_world_size()
        my_rank = dist.get_rank()
        stage_of: list = [None] * world
        pp_mesh = trainer.parallel_dims.get_optional_mesh("pp")
        my_stage_id = dist.get_rank(group=pp_mesh.get_group()) if pp_mesh is not None else 0
        dist.all_gather_object(stage_of, my_stage_id)
        stage_groups: dict[int, list[int]] = {}
        for rank, stage in enumerate(stage_of):
            stage_groups.setdefault(stage, []).append(rank)
        my_stage = stage_of[my_rank]
        if complete_across_pp:
            audience, broadcast_group = list(range(world)), None
        else:
            groups = _stage_process_groups(tuple(tuple(ranks) for _, ranks in sorted(stage_groups.items())))
            audience, broadcast_group = stage_groups[my_stage], groups[sorted(stage_groups).index(my_stage)]
        return cls(
            world=world, my_rank=my_rank, device=trainer.device, audience=audience, broadcast_group=broadcast_group
        )

    def stream(self, local: dict[str, torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
        local_meta = {name: (tuple(t.shape), str(t.dtype)) for name, t in local.items()}
        gathered: list = [None] * self.world
        dist.all_gather_object(gathered, local_meta)

        if all(meta.keys() == local_meta.keys() for meta in gathered):
            for name in sorted(local):
                yield name, gather_full_param(local[name])
            return

        owners: dict[str, list[int]] = {}
        specs: dict[str, tuple] = {}
        for rank, meta in enumerate(gathered):
            for name, (shape, dtype) in meta.items():
                owners.setdefault(name, []).append(rank)
                if specs.setdefault(name, (shape, dtype)) != (shape, dtype):
                    raise RuntimeError(f"ranks disagree on the shape/dtype of {name}")

        names = [name for name in sorted(owners) if self.audience.intersection(owners[name])]
        for name in names:
            shape, dtype = specs[name]
            holders = [rank for rank in owners[name] if rank in self.audience]
            if self.my_rank in holders:
                tensor = gather_full_param(local[name]).contiguous()
            else:
                tensor = torch.empty(shape, dtype=getattr(torch, dtype.split(".")[-1]), device=self.device)
            dist.broadcast(tensor, src=holders[0], group=self.broadcast_group)
            yield name, tensor
            del tensor


_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def _checkpoint_dtypes(hf_checkpoint: str) -> dict[str, torch.dtype]:
    index_path = os.path.join(hf_checkpoint, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = sorted(os.path.basename(p) for p in glob.glob(os.path.join(hf_checkpoint, "*.safetensors")))

    dtypes: dict[str, torch.dtype] = {}
    for shard in shards:
        with safetensors.safe_open(os.path.join(hf_checkpoint, shard), framework="pt") as handle:
            for name in handle.keys():
                dtype = _SAFETENSORS_DTYPES.get(handle.get_slice(name).get_dtype())
                if dtype is not None:
                    dtypes[name] = dtype
    return dtypes
