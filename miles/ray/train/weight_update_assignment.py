from miles.ray.rollout.inference_controller import UpdatableEngines


def split_update_targets(info: UpdatableEngines, *, num_trainer_cells: int) -> list[UpdatableEngines]:
    assert num_trainer_cells > 0, "an update needs at least one trainer cell to send the weights"

    total = len(info.engine_cell_ids)
    base, remainder = divmod(total, num_trainer_cells)

    assignments: list[UpdatableEngines] = []
    start = 0
    for index in range(num_trainer_cells):
        stop = start + base + (1 if index < remainder else 0)
        assignments.append(_slice_update_targets(info, start=start, stop=stop))
        start = stop

    assert start == total, f"the assignments cover {start} of {total} inference cells"
    return assignments


def _slice_update_targets(info: UpdatableEngines, *, start: int, stop: int) -> UpdatableEngines:
    cell_ids = list(info.engine_cell_ids[start:stop])
    return UpdatableEngines(
        rollout_engines=list(info.rollout_engines[start:stop]),
        engine_gpu_counts=list(info.engine_gpu_counts[start:stop]),
        engine_gpu_offsets=list(info.engine_gpu_offsets[start:stop]),
        engine_cell_ids=cell_ids,
        snapshot_cell_id_to_hashes={cell_id: info.snapshot_cell_id_to_hashes[cell_id] for cell_id in cell_ids},
    )
