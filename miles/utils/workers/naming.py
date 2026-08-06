from typing import NamedTuple


class ParsedCellId(NamedTuple):
    spec_name: str
    cell_index: int


def parse_cell_id(cell_id: str) -> ParsedCellId:
    spec_name, cell_index = cell_id.rsplit("-", maxsplit=1)
    return ParsedCellId(spec_name=spec_name, cell_index=int(cell_index))


def compute_cell_id(*, spec_name: str, cell_index: int) -> str:
    return f"{spec_name}-{cell_index}"


# TODO refactor & move later
def compute_worker_name(*, spec_name: str, cell_index: int = 0, worker_in_cell_index: int = 0) -> str:
    return f"{spec_name}-{cell_index}-{worker_in_cell_index}"


# TODO refactor & move later
def parse_worker_name(worker_name: str) -> tuple[str, int, int]:
    spec_name, cell_index, worker_in_cell_index = worker_name.rsplit("-", maxsplit=2)
    return spec_name, int(cell_index), int(worker_in_cell_index)
