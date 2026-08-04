def compute_cell_id(*, pool_id: str, cell_index: int) -> str:
    return f"{pool_id}-{cell_index}"


# TODO refactor & move later
def compute_worker_name(*, pool_id: str, cell_index: int = 0, worker_in_cell_index: int = 0) -> str:
    return f"{pool_id}-{cell_index}-{worker_in_cell_index}"


# TODO refactor & move later
def parse_worker_name(worker_name: str) -> tuple[str, int, int]:
    pool_id, cell_index, worker_in_cell_index = worker_name.rsplit("-", maxsplit=2)
    return pool_id, int(cell_index), int(worker_in_cell_index)
