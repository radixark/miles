import re
from typing import NamedTuple

DNS_LABEL_PATTERN: str = r"[a-z0-9]([-a-z0-9]*[a-z0-9])?"
DNS_SUBDOMAIN_PATTERN: str = rf"{DNS_LABEL_PATTERN}(\.{DNS_LABEL_PATTERN})*"
POOL_NAME_MAX_LENGTH = 40
_LONGEST_ENGINE_POOL_ID_AROUND_THE_INSTANCE_ID = "inference-engine--99-99"
DEPLOY_INSTANCE_ID_MAX_LENGTH = POOL_NAME_MAX_LENGTH - len(_LONGEST_ENGINE_POOL_ID_AROUND_THE_INSTANCE_ID)
TRAINER_CONTROLLER_POOL_ID_PREFIX = "trainer-controller-"
TRAINER_ID_MAX_LENGTH = POOL_NAME_MAX_LENGTH - len(TRAINER_CONTROLLER_POOL_ID_PREFIX)
PORT_NAME_MAX_LENGTH = 15
PORT_NAME_PATTERN = r"^([0-9]+-)*[0-9]*[a-z][a-z0-9]*(-[a-z0-9]+)*$"
_PORT_NAME_SEPARATORS = re.compile(r"[^a-z0-9]+")


def compute_port_name(name: str) -> str:
    cleaned = _PORT_NAME_SEPARATORS.sub("-", name.lower()).strip("-")[:PORT_NAME_MAX_LENGTH].rstrip("-")
    assert (
        re.fullmatch(PORT_NAME_PATTERN, cleaned) is not None
    ), f"port name {name!r} shortens to {cleaned!r}, which Kubernetes rejects: it must match {PORT_NAME_PATTERN}"
    return cleaned


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
