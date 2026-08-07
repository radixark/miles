from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class IndepDPInfo:
    cell_id: str
    num_cells: int
    alive_rank: int
    alive_size: int
    quorum_id: int
    alive_cell_ids: list[str]

    @classmethod
    def create_trivial(cls) -> "IndepDPInfo":
        return cls(
            cell_id="trivial-0",
            num_cells=1,
            alive_rank=0,
            alive_size=1,
            quorum_id=0,
            alive_cell_ids=["trivial-0"],
        )

    def __post_init__(self):
        assert self.alive_rank == self.alive_cell_ids.index(self.cell_id)
        assert self.alive_size == len(self.alive_cell_ids)


def create_tcp_store() -> tuple["torch.distributed.TCPStore", str]:
    import ray
    import torch.distributed

    store = torch.distributed.TCPStore(
        host_name="0.0.0.0",
        port=0,
        is_master=True,
        wait_for_workers=False,
    )
    host = ray.util.get_node_ip_address()
    port = store.port
    return store, f"{host}:{port}"
