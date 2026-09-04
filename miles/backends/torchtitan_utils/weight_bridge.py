"""torchtitan's hook into the shared weight-update machinery.

Transport, engine session, bucketing and atomic groups all live in
``training_utils/weight_update``; the one thing a backend supplies is an HF
weight iterator. torchtitan's is ``TitanHfWeightIterator``, fed by
``hf_weights``: the trainer's DTensor shards become HF-named full tensors
through the model's own ``state_dict_adapter.to_hf`` -- the same mapping its
checkpointer used to load the weights, run in reverse -- reassembled across
dp/tp/ep and, when the protocol's placement asks for it, completed across
pipeline stages, one tensor resident at a time.
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
    """Streams a TitanTrainer's weights as HF-named tensors.

    ``model`` is the trainer rather than a module: it owns the model parts and
    the adapter. dp/tp/ep shards are always reassembled; whether a pipeline
    stage's layers are broadcast to the ranks that lack them follows the
    protocol's placement, because completing the stream on every rank means
    every rank materializes the whole model.
    """

    # No forced placement: the protocol decides. gather_pp=True has every rank
    # materialize the whole model, which a pipelined 30B does not survive;
    # gather_pp=False lets each stage stream its own layers.
    forced_placement = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._engine_dtypes = _checkpoint_dtypes(self.args.hf_checkpoint)

    def _iter_hf_param_units(self, weights, *, materialize):
        # ``weights`` is None by contract here: the trainer reads its live parts.
        # Non-materializing ranks still walk the stream, since producing each
        # tensor is a collective every rank has to join.
        for name, tensor in hf_weights(self.model, complete_across_pp=self.placement.gather_pp):
            if materialize:
                yield [(name, self._to_engine_dtype(name, tensor))]

    def _to_engine_dtype(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Cast a master weight to the dtype the engine holds for that tensor.

        Mixed precision keeps torchtitan's parameters in fp32 and casts to the
        compute dtype on the fly, so the state dict hands out fp32 while the
        checkpoint the engine mirrors is mostly bf16. Sending fp32 doubles every
        transfer, and disk-delta -- the one protocol that reconciles the stream
        against the checkpoint's own bytes -- refuses it outright.

        Per tensor rather than per model, because a checkpoint mixes dtypes: a
        blanket bf16 cast turns qwen3.5's fp32 log scales into something the
        checkpoint never held. A name the checkpoint does not carry is left
        alone rather than guessed at.
        """
        target = self._engine_dtypes.get(name)
        if target is None or tensor.dtype == target or not tensor.is_floating_point():
            return tensor
        return tensor.to(target)

    def _hf_atomic_update_groups(self):
        # DeepSeek's MLA down-projections are fused by sglang from two HF
        # tensors; whether the model has them is a fact of its architecture.
        q_lora_rank = getattr(self.model.model_config, "q_lora_rank", None) or None
        return get_hf_atomic_update_groups(self.model_name, q_lora_rank=q_lora_rank)

    def _iter_hf_adapter_units(self, lora_name, adapter, *, materialize):
        raise NotImplementedError("the torchtitan backend has no LoRA")


def hf_weights(trainer, *, complete_across_pp: bool = True) -> Iterator[tuple[str, torch.Tensor]]:
    """HF-named tensors, materialized one at a time, for the engine push.

    The weight transport requires every rank in an IPC gather group to
    stream the same tensor sequence. dp/tp shards reassemble via
    ``gather_full_param``; under PP each tensor lives on exactly one
    stage, so it is additionally broadcast over the pp mesh -- after which
    every rank yields the identical full stream and the transport stays
    PP-oblivious. One tensor is resident at a time either way.

    An offloaded model comes back to the device for the duration: unlike
    the plain state dicts FSDP streams, titan's fused-QKV save hooks run
    DTensor collectives inside ``state_dict()`` itself, and the meshes
    have no CPU backend. Weights-only occupancy is strictly below the
    training peak, so whenever training fits, this does.
    """
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
    """This rank's pipeline-stage process group, created once.

    Weights are pushed on every optimizer step, and ``new_group`` builds a
    fresh communicator each time it is called and never releases one, so
    creating these per push exhausts NCCL's device buffers within a few
    updates ("Failed to CUDA calloc"). Creation is collective, so every rank
    builds every stage's group in the same order.
    """
    if getattr(trainer, "_stage_groups", None) is None:
        trainer._stage_groups = {stage: dist.new_group(ranks) for stage, ranks in sorted(stage_groups.items())}
    return trainer._stage_groups[my_stage]


def _hf_weights_on_device(trainer, *, complete_across_pp: bool) -> Iterator[tuple[str, torch.Tensor]]:
    # The checkpointer only builds its adapter when checkpointing is
    # enabled; weight streaming needs the mapping regardless.
    sd_adapter = getattr(trainer.checkpointer, "sd_adapter", None)
    if sd_adapter is None:
        sd_adapter = trainer.config.model_spec.state_dict_adapter(trainer.model_config, trainer.config.hf_assets_path)
    local = sd_adapter.to_hf({k: v for part in trainer.model_parts for k, v in part.state_dict().items()})

    # Which ranks hold which key. Two parallelisms make the export
    # rank-partial: a pipeline stage exports only its own layers, and under
    # expert parallelism the adapter names each rank's experts by their
    # global index, so every rank exports a different slice of them.
    # DTensor.shape is already the global shape, so the metadata describes
    # the post-gather tensor.
    world = dist.get_world_size()
    local_meta = {name: (tuple(t.shape), str(t.dtype)) for name, t in local.items()}
    gathered: list = [None] * world
    dist.all_gather_object(gathered, local_meta)

    if all(meta.keys() == local_meta.keys() for meta in gathered):
        # Every rank exports the same keys: dp/tp/fsdp sharding is internal
        # to each tensor and gather_full_param resolves it.
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
    # Who shares my pipeline stage. Completion is always needed *within* a
    # stage -- expert parallelism has each rank export a different slice of
    # the experts, so no single rank holds a stage's whole set -- and the
    # placement only decides whether it also crosses stages.
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
        # Every holder joins the gather -- they are exactly the ranks the
        # tensor's own mesh spans, so the collective is complete. The
        # lowest of them then broadcasts to the ranks that lack it;
        # replicas (data parallelism) hold identical values after a step,
        # so which holder broadcasts does not matter, only that it is
        # agreed on.
        if my_rank in holders:
            tensor = gather_full_param(local[name]).contiguous()
        else:
            tensor = torch.empty(shape, dtype=getattr(torch, dtype.split(".")[-1]), device=trainer.device)
        dist.broadcast(tensor, src=holders[0], group=broadcast_group)
        yield name, tensor
        # Drop this generator's reference as soon as the consumer has taken
        # the tensor: holding it until the next iteration keeps one extra
        # full tensor alive per unit, and the consumer's bucket is the only
        # thing that should own it.
        del tensor


_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def _checkpoint_dtypes(hf_checkpoint: str) -> dict[str, torch.dtype]:
    """Per-tensor dtypes from the checkpoint's safetensors headers.

    A checkpoint is not one dtype: qwen3.5 keeps its linear-attention log
    scales and biases in fp32 beside bf16 weights, so casting everything to a
    single model-wide dtype corrupts exactly those. Only the headers are read,
    never the tensor data.
    """
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
