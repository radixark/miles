from dataclasses import dataclass


@dataclass(frozen=True)
class IndepDPInfo:
    cell_index: int
    num_cells: int
    alive_rank: int
    alive_size: int
    quorum_id: int
    alive_cell_indices: list[int]

    @classmethod
    def create_trivial(cls) -> "IndepDPInfo":
        return cls(
            cell_index=0,
            num_cells=1,
            alive_rank=0,
            alive_size=1,
            quorum_id=0,
            alive_cell_indices=[0],
        )

    def __post_init__(self):
        assert self.alive_rank == self.alive_cell_indices.index(self.cell_index)
        assert self.alive_size == len(self.alive_cell_indices)
