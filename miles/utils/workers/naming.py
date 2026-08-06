def compute_worker_name(*, spec_name: str, cell_index: int, worker_in_cell_index: int) -> str:
    return f"{spec_name}-{cell_index}-{worker_in_cell_index}"
