def compute_worker_name(*, pool_id: str, cell_index: int, worker_in_cell_index: int) -> str:
    return f"{pool_id}-{cell_index}-{worker_in_cell_index}"
