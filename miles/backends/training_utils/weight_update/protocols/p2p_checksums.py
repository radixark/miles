import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch


@dataclass
class P2PChecksumShard:
    cell_id: str
    engine_rank: int
    session_id: str
    expected_names: frozenset[str]
    tensors: dict[str, str] = field(default_factory=dict)


def checksum_transfer_tensors(parameters: dict[str, torch.Tensor], names: Sequence[str]) -> dict[str, str]:
    assert len(names) == len(set(names)), "A P2P bucket repeats a parameter"
    checksums = {}
    for name in names:
        tensor = parameters[name]
        assert tensor.device.type == "cpu", "P2P checksums must describe the registered CPU send buffers"
        checksums[name] = hashlib.sha256(tensor.detach().contiguous().flatten().view(torch.uint8).numpy()).hexdigest()
    return checksums


def merge_transfer_checksums(
    shards: Sequence[P2PChecksumShard], *, healthy_engine_ranks: dict[str, int]
) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, int], list[P2PChecksumShard]] = {}
    for shard in shards:
        if shard.cell_id not in healthy_engine_ranks:
            continue
        assert 0 <= shard.engine_rank < healthy_engine_ranks[shard.cell_id], "Unexpected P2P checksum rank"
        grouped.setdefault((shard.cell_id, shard.engine_rank), []).append(shard)

    result: dict[str, dict[str, dict[str, str]]] = {}
    for cell_id, rank_count in healthy_engine_ranks.items():
        result[cell_id] = {}
        for rank in range(rank_count):
            parts = grouped.get((cell_id, rank), [])
            assert parts, f"Missing P2P checksum sender for {cell_id}/rank{rank}"
            assert len({part.session_id for part in parts}) == 1, f"P2P receiver changed for {cell_id}/rank{rank}"
            expected = parts[0].expected_names
            assert expected, f"P2P receiver registered no parameters for {cell_id}/rank{rank}"
            tensors: dict[str, str] = {}
            for part in parts:
                assert (
                    part.expected_names == expected
                ), f"P2P receiver parameter sets disagree for {cell_id}/rank{rank}"
                assert (
                    not tensors.keys() & part.tensors.keys()
                ), f"Duplicate P2P tensor sender for {cell_id}/rank{rank}"
                tensors.update(part.tensors)
            assert tensors.keys() == expected, f"Incomplete P2P tensor coverage for {cell_id}/rank{rank}"
            result[cell_id][str(rank)] = tensors
    return result
