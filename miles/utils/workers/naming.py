def compute_cell_id(*, pool_id: str, cell_index: int) -> str:
    return f"{pool_id}-{cell_index}"


# TODO refactor & move later
def compute_worker_name(*, cell_id: str, worker_in_cell_index: int = 0) -> str:
    return f"{cell_id}-{worker_in_cell_index}"


# TODO refactor & move later
def parse_worker_name(worker_name: str) -> tuple[str, int]:
    cell_id, worker_in_cell_index = worker_name.rsplit("-", maxsplit=1)
    return cell_id, int(worker_in_cell_index)
