"""torchtitan's hook into the shared weight-update machinery.

Transport, engine session, bucketing and atomic groups live in
``training_utils/weight_update``; the backend supplies the HF weight iterator.
``hf_weights`` produces its stream: the trainer's shards become HF-named full
tensors through the model's own ``state_dict_adapter.to_hf``, reassembled
across dp/tp/ep and, when the protocol's placement asks for it, completed
across pipeline stages, one tensor resident at a time.
"""

import glob
import json
import os
from argparse import Namespace
from collections.abc import Iterator

import safetensors
import torch
import torch.distributed as dist

from miles.backends.fsdp_utils.dtensor import gather_full_param
from miles.backends.training_utils.weight_update.hf_weight_iterator import (
    HfWeightIteratorBase,
    WeightUpdatePlacement,
    resolve_placement,
)
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups


class TitanHfWeightIterator(HfWeightIteratorBase):
    """Streams a TitanTrainer's weights as HF-named tensors; ``model`` is the trainer."""

    forced_placement = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._engine_dtypes = _checkpoint_dtypes(self.args.hf_checkpoint)

    def _iter_hf_param_units(self, weights, *, materialize):
        for name, tensor in hf_weights(self.model, complete_across_pp=self.placement.gather_pp):
            if materialize:
                yield [(name, self._to_engine_dtype(name, tensor))]

    def _to_engine_dtype(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Cast a master weight to the dtype the checkpoint holds for that tensor."""
        target = self._engine_dtypes.get(name)
        if target is None or tensor.dtype == target or not tensor.is_floating_point():
            return tensor
        return tensor.to(target)

    def _hf_atomic_update_groups(self):
        q_lora_rank = getattr(self.model.model_config, "q_lora_rank", None) or None
        return get_hf_atomic_update_groups(self.model_name, q_lora_rank=q_lora_rank)

    def _iter_hf_adapter_units(self, lora_name, adapter, *, materialize):
        raise NotImplementedError("the torchtitan backend has no LoRA")


def hf_weights(trainer, *, complete_across_pp: bool = True) -> Iterator[tuple[str, torch.Tensor]]:
    """HF-named full tensors, one resident at a time; an offloaded model is brought back
    to the device for the duration because the adapter's collectives need the meshes."""
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


def _stage_group(trainer, stage_groups: dict[int, list[int]], my_stage: int):
    """This rank's pipeline-stage process group, created once for the trainer's lifetime."""
    if getattr(trainer, "_stage_groups", None) is None:
        trainer._stage_groups = {stage: dist.new_group(ranks) for stage, ranks in sorted(stage_groups.items())}
    return trainer._stage_groups[my_stage]


def _hf_weights_on_device(trainer, *, complete_across_pp: bool) -> Iterator[tuple[str, torch.Tensor]]:
    sd_adapter = getattr(trainer.checkpointer, "sd_adapter", None)
    if sd_adapter is None:
        sd_adapter = trainer.config.model_spec.state_dict_adapter(trainer.model_config, trainer.config.hf_assets_path)
    local = sd_adapter.to_hf({k: v for part in trainer.model_parts for k, v in part.state_dict().items()})

    world = dist.get_world_size()
    local_meta = {name: (tuple(t.shape), str(t.dtype)) for name, t in local.items()}
    gathered: list = [None] * world
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
        audience, broadcast_group = stage_groups[my_stage], _stage_group(trainer, stage_groups, my_stage)

    audience_set = set(audience)
    names = [name for name in sorted(owners) if audience_set.intersection(owners[name])]
    for name in names:
        shape, dtype = specs[name]
        holders = [rank for rank in owners[name] if rank in audience_set]
        if my_rank in holders:
            tensor = gather_full_param(local[name]).contiguous()
        else:
            tensor = torch.empty(shape, dtype=getattr(torch, dtype.split(".")[-1]), device=trainer.device)
        dist.broadcast(tensor, src=holders[0], group=broadcast_group)
        yield name, tensor
        del tensor


_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def _checkpoint_dtypes(hf_checkpoint: str) -> dict[str, torch.dtype]:
    """Per-tensor dtypes from the checkpoint's safetensors headers; no tensor data is read."""
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


def get_hf_weight_iterator(
    args: Namespace,
    trainer,
    *,
    required_placement: WeightUpdatePlacement,
    model_name: str,
    quantization_config: dict | None,
) -> HfWeightIteratorBase:
    return TitanHfWeightIterator(
        args,
        trainer,
        placement=resolve_placement(required_placement, TitanHfWeightIterator.forced_placement),
        model_name=model_name,
        quantization_config=quantization_config,
    )
