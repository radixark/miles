import argparse
import json
import logging
from pathlib import Path

from ray.actor import ActorHandle

logger = logging.getLogger(__name__)

INITIAL_DUMP_NAME = "initial"


class EngineChecksumDumper:
    def __init__(self, *, dump_dir: Path, rollout_manager: ActorHandle) -> None:
        self._dump_dir = dump_dir
        self._rollout_manager = rollout_manager

    @staticmethod
    def from_args(args: argparse.Namespace, *, rollout_manager: ActorHandle | None) -> "EngineChecksumDumper | None":
        if args.ci_dump_engine_weight_checksums is None or rollout_manager is None:
            return None
        return EngineChecksumDumper(
            dump_dir=Path(args.ci_dump_engine_weight_checksums),
            rollout_manager=rollout_manager,
        )

    async def dump(self, *, rollout_id: int | None) -> None:
        nested: list[list[list[dict | None]]] = await self._rollout_manager.check_weights.remote(action="checksum")
        engine_responses: list[dict] = [
            response
            for per_server in nested
            for per_group in per_server
            for response in per_group
            if response is not None
        ]
        assert engine_responses, "check_weights('checksum') returned no engine responses"

        rollout_dir = self._dump_dir / (f"rollout_{rollout_id}" if rollout_id is not None else INITIAL_DUMP_NAME)
        rollout_dir.mkdir(parents=True, exist_ok=True)
        for engine_index, response in enumerate(engine_responses):
            path = rollout_dir / f"engine_{engine_index}.json"
            path.write_text(json.dumps(response, indent=2, sort_keys=True))
        logger.info(
            "Dumped engine weight checksums for %d engine(s) to %s",
            len(engine_responses),
            rollout_dir,
        )


def compare_engine_checksum_dumps(
    *,
    baseline_dir: str,
    target_dir: str,
    expected_rollout_names: set[str],
    expected_num_engines: int,
) -> None:
    assert expected_rollout_names, "expected_rollout_names must not be empty"
    assert expected_num_engines > 0, f"expected_num_engines must be positive, got {expected_num_engines}"
    baseline_root = Path(baseline_dir)
    target_root = Path(target_dir)
    assert baseline_root.is_dir(), f"Baseline engine checksum dir does not exist: {baseline_root}"
    assert target_root.is_dir(), f"Target engine checksum dir does not exist: {target_root}"

    baseline_rollouts = set(_list_child_names(baseline_root))
    target_rollouts = set(_list_child_names(target_root))
    assert baseline_rollouts == expected_rollout_names, (
        f"Engine checksum rollout dirs mismatch under {baseline_root}: "
        f"baseline={sorted(baseline_rollouts)} vs expected={sorted(expected_rollout_names)}"
    )
    assert target_rollouts == expected_rollout_names, (
        f"Engine checksum rollout dirs mismatch under {target_root}: "
        f"target={sorted(target_rollouts)} vs expected={sorted(expected_rollout_names)}"
    )

    expected_engine_files = {f"engine_{engine_index}.json" for engine_index in range(expected_num_engines)}
    num_tensor_checksums = 0
    num_engine_files = 0
    for rollout_name in sorted(expected_rollout_names):
        baseline_files = set(_list_child_names(baseline_root / rollout_name))
        target_files = set(_list_child_names(target_root / rollout_name))
        assert baseline_files == expected_engine_files, (
            f"Engine checksum files mismatch for {rollout_name}: "
            f"baseline={sorted(baseline_files)} vs expected={sorted(expected_engine_files)}"
        )
        assert target_files == expected_engine_files, (
            f"Engine checksum files mismatch for {rollout_name}: "
            f"target={sorted(target_files)} vs expected={sorted(expected_engine_files)}"
        )
        for file_name in sorted(expected_engine_files):
            num_engine_files += 1
            num_tensor_checksums += _compare_engine_checksum_file(
                baseline_path=baseline_root / rollout_name / file_name,
                target_path=target_root / rollout_name / file_name,
                context=f"{rollout_name}/{file_name}",
            )

    print(
        f"Engine checksum comparison passed: {len(baseline_rollouts)} rollout(s), "
        f"{num_engine_files} engine file(s), {num_tensor_checksums} tensor checksum(s) compared "
        f"({baseline_root} vs {target_root})"
    )


def _list_child_names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def _compare_engine_checksum_file(*, baseline_path: Path, target_path: Path, context: str) -> int:
    baseline = json.loads(baseline_path.read_text())
    target = json.loads(target_path.read_text())

    assert baseline.get("success") is True, f"{context}: baseline checksum response not successful: {baseline}"
    assert target.get("success") is True, f"{context}: target checksum response not successful: {target}"
    # ranks arrive in non-deterministic zmq order; sort by global rank to pair the same rank.
    baseline_ranks: list[dict] = sorted(baseline["ranks"], key=lambda rank: rank["parallelism_info"]["rank"])
    target_ranks: list[dict] = sorted(target["ranks"], key=lambda rank: rank["parallelism_info"]["rank"])
    assert len(baseline_ranks) == len(
        target_ranks
    ), f"{context}: rank count mismatch: baseline={len(baseline_ranks)} vs target={len(target_ranks)}"

    num_tensor_checksums = 0
    mismatches: list[str] = []
    for rank_index, (baseline_rank, target_rank) in enumerate(zip(baseline_ranks, target_ranks, strict=True)):
        baseline_checksums: dict[str, str] = baseline_rank["checksums"]
        target_checksums: dict[str, str] = target_rank["checksums"]
        assert baseline_checksums, f"{context} rank {rank_index}: baseline has no tensor checksums"
        assert set(baseline_checksums) == set(target_checksums), (
            f"{context} rank {rank_index}: tensor name sets differ: "
            f"baseline-only={sorted(set(baseline_checksums) - set(target_checksums))}, "
            f"target-only={sorted(set(target_checksums) - set(baseline_checksums))}"
        )
        for tensor_name in sorted(baseline_checksums):
            num_tensor_checksums += 1
            if baseline_checksums[tensor_name] != target_checksums[tensor_name]:
                mismatches.append(
                    f"rank {rank_index} tensor '{tensor_name}': "
                    f"baseline={baseline_checksums[tensor_name]} vs target={target_checksums[tensor_name]}"
                )

    assert not mismatches, (
        f"Engine weight checksum mismatch in {context} ({len(mismatches)} tensor(s)); the engine weights "
        f"after this update_weights are not bitwise-identical between baseline and target:\n"
        + "\n".join(f"  - {m}" for m in mismatches)
    )
    return num_tensor_checksums
