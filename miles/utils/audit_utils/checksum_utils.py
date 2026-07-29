from typing import Any

InferenceEngineChecksums = dict[str, str]


def flatten_inference_engine_checksums(check_weights_result: Any) -> list[InferenceEngineChecksums]:
    engine_bodies = _flatten_to_inference_engine_bodies(check_weights_result)
    surviving = [body for body in engine_bodies if body is not None]
    assert surviving, (
        f"check_weights('checksum') returned no non-None engine bodies "
        f"(got {len(engine_bodies)} entries, all None): {check_weights_result!r}"
    )
    return [_merge_inference_engine_ranks(body) for body in surviving]


def _flatten_to_inference_engine_bodies(check_weights_result: Any) -> list[Any]:
    return [engine_body for server_group in check_weights_result for engine_body in server_group]


def _merge_inference_engine_ranks(engine_body: dict[str, Any]) -> InferenceEngineChecksums:
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
