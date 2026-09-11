from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

_ROW_WIDTH = 5


@dataclass(frozen=True)
class TrainingSampleIdentity:
    source_sample_index: int
    row_index: int
    row_count: int


class ModelCompanion(torch.nn.Module):
    def __init__(self, *, pipeline_rank: int = 0, chunk_index: int = 0, replica_id: tuple[int, ...] = (0,)) -> None:
        super().__init__()
        self.pipeline_rank = pipeline_rank
        self.chunk_index = chunk_index
        self.replica_id = replica_id
        self.rows = torch.nn.Parameter(
            torch.empty((0, _ROW_WIDTH), dtype=torch.int64, device="cpu"), requires_grad=False
        )
        self.weight_version = torch.nn.Parameter(torch.zeros((), dtype=torch.int64, device="cpu"), requires_grad=False)

    def record(self, samples: Iterable[TrainingSampleIdentity], *, is_skipped: bool = False) -> None:
        counts = _RowCodec.read(self.rows)
        counts.update((sample, is_skipped) for sample in samples)
        _RowCodec.write(target=self.rows, counts=counts)

    def snapshot(self, *, is_skipped: bool = False) -> dict[TrainingSampleIdentity, int]:
        return {
            identity: count
            for (identity, skipped), count in _RowCodec.read(self.rows).items()
            if skipped == is_skipped
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
        for name, parameter in self.named_parameters():
            incoming = state_dict[f"{prefix}{name}"]
            assert incoming.dtype == torch.int64
            if name == "weight_version":
                assert incoming.ndim == 0 and incoming.item() >= 0
            else:
                assert incoming.ndim == 2 and incoming.shape[1] == _ROW_WIDTH
            parameter.resize_(incoming.shape)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class ModelCompanionUtils:
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
    def weight_version(model: Sequence[torch.nn.Module]) -> int:
        companions = _model_companions(model)
        assert companions, "Model is missing its companion"
        return ModelCompanionUtils.version_from_parameters(
            ("model_companion.weight_version", companion.weight_version) for companion in companions
        )

    @staticmethod
    def version_from_parameters(parameters: Iterable[tuple[str, torch.Tensor]]) -> int:
        versions = {int(parameter.item()) for name, parameter in parameters if name.endswith(".weight_version")}
        assert len(versions) == 1, f"Model companion versions diverged: {versions}"
        return versions.pop()

    @staticmethod
    def bump_weight_version(model: Sequence[torch.nn.Module]) -> None:
        companions = _model_companions(model)
        if companions:
            ModelCompanionUtils.weight_version(model)
        for companion in companions:
            companion.weight_version.add_(1)

    @staticmethod
    def record(
        model: Sequence[torch.nn.Module], samples: Iterable[TrainingSampleIdentity], *, is_skipped: bool = False
    ) -> None:
        identities = list(samples)
        for companion in _model_companions(model):
            companion.record(identities, is_skipped=is_skipped)

    @staticmethod
    def snapshot(model: Sequence[torch.nn.Module], *, is_skipped: bool = False) -> dict[TrainingSampleIdentity, int]:
        companions = _model_companions(model)
        assert companions, "Model is missing its CPU training witness"
        snapshots = [companion.snapshot(is_skipped=is_skipped) for companion in companions]
        assert all(snapshot == snapshots[0] for snapshot in snapshots[1:]), "CPU witness model chunks diverged"
        return snapshots[0]

    @staticmethod
    def is_parameter_name(name: str) -> bool:
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


def _model_companions(model: Sequence[torch.nn.Module]) -> list[ModelCompanion]:
    return [module for chunk in model for module in chunk.modules() if isinstance(module, ModelCompanion)]


class _RowCodec:
    @staticmethod
    def read(rows: torch.Tensor) -> Counter[tuple[TrainingSampleIdentity, bool]]:
        return Counter(
            {
                (
                    TrainingSampleIdentity(source_sample_index=source, row_index=row, row_count=row_count),
                    bool(is_skipped),
                ): count
                for source, row, row_count, count, is_skipped in rows.tolist()
            }
        )

    @staticmethod
    def write(*, target: torch.Tensor, counts: Counter[tuple[TrainingSampleIdentity, bool]]) -> None:
        values = [
            [identity.source_sample_index, identity.row_index, identity.row_count, count, is_skipped]
            for (identity, is_skipped), count in counts.items()
        ]
        source = torch.tensor(values, dtype=torch.int64, device="cpu").reshape((-1, _ROW_WIDTH))
        target.resize_(source.shape)
        target.copy_(source)
