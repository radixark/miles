from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from miles.utils.types import SampleLineage

_ROW_WIDTH = 5


class ModelCompanion(torch.nn.Module):
    def __init__(self, *, pipeline_rank: int, chunk_index: int, replica_id: tuple[int, ...]) -> None:
        super().__init__()
        self.pipeline_rank = pipeline_rank
        self.chunk_index = chunk_index
        self.replica_id = replica_id
        self.sample_consumptions = torch.nn.Parameter(
            torch.empty((0, _ROW_WIDTH), dtype=torch.int64, device="cpu"), requires_grad=False
        )
        self.sample_consumptions.miles_dynamic_shape = True
        self.weight_version = torch.nn.Parameter(torch.zeros((), dtype=torch.int64, device="cpu"), requires_grad=False)

    def record_sample_consumptions(self, samples: Iterable[SampleLineage], *, is_skipped: bool = False) -> None:
        counts = _RowCodec.read(self.sample_consumptions)
        counts.update(_Row(identity=sample, is_skipped=is_skipped) for sample in samples)
        _RowCodec.write(target=self.sample_consumptions, counts=counts)

    def snapshot_sample_consumptions(self, *, is_skipped: bool) -> dict[SampleLineage, int]:
        return {
            row.identity: count
            for row, count in _RowCodec.read(self.sample_consumptions).items()
            if row.is_skipped == is_skipped
        }

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> dict[str, Any]:
        from megatron.core.dist_checkpointing.mapping import ShardedObject

        return {
            f"{prefix}{name}": ShardedObject(
                key=f"{prefix}pp{self.pipeline_rank}.chunk{self.chunk_index}.{name}",
                data=parameter.detach().clone(),
                global_shape=(1,),
                global_offset=(0,),
                replica_id=self.replica_id,
            )
            for name, parameter in self.named_parameters()
        }

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> "ModelCompanion":
        return self

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        _resize_when_load_from_state_dict(self, state_dict, "sample_consumptions", prefix=prefix)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class ModelCompanionInstallationUtils:
    @staticmethod
    def install(model: torch.nn.Module, *, chunk_index: int) -> None:
        from miles.backends.training_utils.parallel import get_parallel_state

        parallel = get_parallel_state()
        model.add_module(
            "model_companion",
            ModelCompanion(
                pipeline_rank=parallel.pp.rank,
                chunk_index=chunk_index,
                replica_id=(parallel.tp.rank, parallel.cp.rank, parallel.intra_dp.rank),
            ),
        )

    @staticmethod
    def is_companion_parameter(name: str) -> bool:
        return "model_companion" in name.split(".")

    @staticmethod
    @contextmanager
    def hide(model: Sequence[torch.nn.Module]) -> Iterator[None]:
        children = [
            (parent, name, child)
            for chunk in model
            for parent in list(chunk.modules())
            for name, child in list(parent.named_children())
            if isinstance(child, ModelCompanion)
        ]
        for parent, name, _ in children:
            delattr(parent, name)
        try:
            yield
        finally:
            for parent, name, child in children:
                parent.add_module(name, child)


class ModelCompanionSampleConsumptionUtils:
    @staticmethod
    def record(
        model: Sequence[torch.nn.Module], samples: Iterable[SampleLineage], *, is_skipped: bool = False
    ) -> None:
        identities = list(samples)
        for companion in _get_companions_of_model(model):
            companion.record_sample_consumptions(identities, is_skipped=is_skipped)

    @staticmethod
    def snapshot(model: Sequence[torch.nn.Module], *, is_skipped: bool) -> dict[SampleLineage, int]:
        companions = _get_companions_of_model(model)
        assert companions, "Model is missing its CPU training witness"
        snapshots = [companion.snapshot_sample_consumptions(is_skipped=is_skipped) for companion in companions]
        assert all(snapshot == snapshots[0] for snapshot in snapshots[1:]), "CPU witness model chunks diverged"
        return snapshots[0]


class ModelCompanionWeightVersionUtils:
    @staticmethod
    def weight_version(model: Sequence[torch.nn.Module]) -> int:
        companions = _get_companions_of_model(model)
        assert companions, "Model is missing its companion"
        return ModelCompanionWeightVersionUtils.from_params(
            parameter for companion in companions for parameter in companion.named_parameters()
        )

    @staticmethod
    def from_params(parameters: Iterable[tuple[str, torch.Tensor]]) -> int:
        versions = {int(parameter.item()) for name, parameter in parameters if name.split(".")[-1] == "weight_version"}
        assert len(versions) == 1, f"Model companion versions diverged: {versions}"
        return versions.pop()

    @staticmethod
    def bump_weight_version(model: Sequence[torch.nn.Module]) -> None:
        companions = _get_companions_of_model(model)
        if companions:
            ModelCompanionWeightVersionUtils.weight_version(model)
        for companion in companions:
            companion.weight_version.add_(1)


class SampleIdentityExtractor:
    @staticmethod
    def get_consumed_sample_identities(
        *, start_offset: int, end_offset: int, data: dict[str, Any], micro_batch_indices: list[list[int]] | None
    ) -> list[SampleLineage]:
        indices = (
            [index for batch in micro_batch_indices[start_offset:end_offset] for index in batch]
            if micro_batch_indices is not None
            else list(range(start_offset, end_offset))
        )
        return [
            SampleLineage(
                source_sample_index=data["lineage_source_sample_indices"][index],
                output_index=data["lineage_output_indices"][index],
                output_count=data["lineage_output_counts"][index],
            )
            for index in indices
        ]

    @staticmethod
    def gather_sample_identities(local: list[SampleLineage]) -> list[SampleLineage]:
        from miles.backends.training_utils.parallel import get_parallel_state

        parallel = get_parallel_state()
        if parallel.effective_dp.size == 1:
            return local
        gathered: list[list[SampleLineage] | None] = [None] * parallel.effective_dp.size
        dist.all_gather_object(gathered, local, group=parallel.effective_dp.gloo_group)
        return [identity for identities in gathered if identities is not None for identity in identities]


def _resize_when_load_from_state_dict(
    module: torch.nn.Module, state_dict: dict[str, torch.Tensor], name: str, *, prefix: str
) -> None:
    incoming = state_dict[f"{prefix}{name}"]
    assert incoming.dtype == torch.int64 and incoming.ndim == 2 and incoming.shape[1] == _ROW_WIDTH
    module.get_parameter(name).resize_(incoming.shape)


def _get_companions_of_model(model: Sequence[torch.nn.Module]) -> list[ModelCompanion]:
    return [module for chunk in model for module in chunk.modules() if isinstance(module, ModelCompanion)]


@dataclass(frozen=True)
class _Row:
    identity: SampleLineage
    is_skipped: bool


class _RowCodec:
    @staticmethod
    def read(sample_consumptions: torch.Tensor) -> Counter[_Row]:
        return Counter(
            {
                _Row(
                    identity=SampleLineage(source_sample_index=source, output_index=row, output_count=output_count),
                    is_skipped=bool(is_skipped),
                ): count
                for source, row, output_count, count, is_skipped in sample_consumptions.tolist()
            }
        )

    @staticmethod
    def write(*, target: torch.Tensor, counts: Counter[_Row]) -> None:
        values = [
            [
                row.identity.source_sample_index,
                row.identity.output_index,
                row.identity.output_count,
                count,
                row.is_skipped,
            ]
            for row, count in counts.items()
        ]
        source = torch.tensor(values, dtype=torch.int64, device="cpu").reshape((-1, _ROW_WIDTH))
        target.resize_(source.shape)
        target.copy_(source)
