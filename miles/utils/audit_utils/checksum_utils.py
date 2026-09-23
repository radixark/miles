from typing import Any

from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel

InferenceEngineChecksums = dict[str, str]


class InferenceEngineChecksumSnapshot(FrozenStrictBaseModel):
    cell_id: str = Field(min_length=1)
    workers_hash: str = Field(min_length=1)
    tensor_checksums: InferenceEngineChecksums = Field(min_length=1)


def merge_inference_engine_ranks(engine_body: dict[str, Any]) -> InferenceEngineChecksums:
    # Ranks arrive in non-deterministic (zmq) order under TP>1; sort and prefix each tensor
    # name with rank{r}/ so distinct shards' identically-named tensors never clobber.
    assert engine_body.get("success", False), f"check_weights engine reported failure: {engine_body!r}"
    ranks: list[dict[str, Any]] = engine_body.get("ranks", []) or []
    assert ranks, f"check_weights engine body has no ranks: {engine_body!r}"

    ranks_sorted = sorted(ranks, key=_gpu_rank)

    merged: InferenceEngineChecksums = {}
    for rank_info in ranks_sorted:
        rank = _gpu_rank(rank_info)
        for name, value in rank_info["checksums"].items():
            merged[f"rank{rank}/{name}"] = value
    return merged


def _gpu_rank(rank_info: dict[str, Any]) -> int:
    parallelism_info = rank_info["parallelism_info"]
    gpu_ranks = {role_info["rank"] for role_info in parallelism_info}
    assert len(gpu_ranks) == 1, f"expected one GPU rank across roles, got {gpu_ranks}: {rank_info!r}"
    return next(iter(gpu_ranks))
