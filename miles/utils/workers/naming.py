from typing import NamedTuple


class ParsedCellId(NamedTuple):
    pool_id: str
    cell_index: int


def parse_cell_id(cell_id: str) -> ParsedCellId:
    pool_id, cell_index = cell_id.rsplit("-", maxsplit=1)
    return ParsedCellId(pool_id=pool_id, cell_index=int(cell_index))


def compute_cell_id(*, pool_id: str, cell_index: int) -> str:
    return f"{pool_id}-{cell_index}"


# TODO refactor & move later
def compute_worker_name(*, pool_id: str, cell_index: int = 0, worker_in_cell_index: int = 0) -> str:
    return f"{pool_id}-{cell_index}-{worker_in_cell_index}"


# TODO refactor & move later
def parse_worker_name(worker_name: str) -> tuple[str, int, int]:
    pool_id, cell_index, worker_in_cell_index = worker_name.rsplit("-", maxsplit=2)
    return pool_id, int(cell_index), int(worker_in_cell_index)


def cell_id_of_worker(worker_name: str) -> str:
    pool_id, cell_index, _worker_in_cell_index = parse_worker_name(worker_name)
    return compute_cell_id(pool_id=pool_id, cell_index=cell_index)
