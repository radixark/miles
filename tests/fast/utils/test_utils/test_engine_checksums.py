import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from miles.utils.test_utils.engine_checksums import EngineChecksumDumper, compare_engine_checksum_dumps


class _FakeRemoteMethod:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn
        self.calls: list[tuple[tuple, dict]] = []

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))

        async def _run() -> Any:
            return self._fn(*args, **kwargs)

        return _run()


class _FakeRolloutManagerHandle:
    def __init__(self, *, checksum_result: list) -> None:
        self.check_weights = _FakeRemoteMethod(lambda action: checksum_result)


def _make_engine_response(checksums: dict[str, str], *, num_ranks: int = 1) -> dict:
    return {
        "success": True,
        "message": "Success.",
        "ranks": [_make_rank_entry(checksums, rank=rank, num_ranks=num_ranks) for rank in range(num_ranks)],
    }


def _make_rank_entry(checksums: dict[str, str], *, rank: int, num_ranks: int) -> dict:
    parallelism_info = {
        "tp_rank": rank,
        "tp_size": num_ranks,
        "dp_rank": 0,
        "dp_size": 1,
        "pp_rank": 0,
        "pp_size": 1,
        "rank": rank,
    }
    return {"checksums": dict(checksums), "parallelism_info": parallelism_info}


def _make_args(ci_dump_engine_weight_checksums: str | None) -> SimpleNamespace:
    return SimpleNamespace(ci_dump_engine_weight_checksums=ci_dump_engine_weight_checksums)


class TestEngineChecksumDumper:
    async def test_from_args_returns_none_when_flag_unset(self):
        """The dumper is off by default: no flag means no dumper and no engine checksum calls."""
        dumper = EngineChecksumDumper.from_args(_make_args(None), rollout_manager=object())

        assert dumper is None

    async def test_from_args_returns_none_without_rollout_manager(self):
        """No rollout manager (e.g. critic group) means no dumper even when the flag is set."""
        dumper = EngineChecksumDumper.from_args(_make_args("/tmp/whatever"), rollout_manager=None)

        assert dumper is None

    async def test_dump_writes_one_json_per_engine_flattened(self, tmp_path: Path):
        """dump flattens servers/groups/engines, drops None entries, and writes engine_<i>.json under rollout_<id>."""
        response_a = _make_engine_response({"w1": "aa", "w2": "bb"})
        response_b = _make_engine_response({"w1": "cc", "w2": "dd"})
        response_c = _make_engine_response({"w1": "ee", "w2": "ff"})
        manager = _FakeRolloutManagerHandle(checksum_result=[[[response_a, None, response_b]], [[response_c]]])
        dumper = EngineChecksumDumper.from_args(_make_args(str(tmp_path)), rollout_manager=manager)

        await dumper.dump(rollout_id=3)

        rollout_dir = tmp_path / "rollout_3"
        assert sorted(p.name for p in rollout_dir.iterdir()) == ["engine_0.json", "engine_1.json", "engine_2.json"]
        assert json.loads((rollout_dir / "engine_0.json").read_text()) == response_a
        assert json.loads((rollout_dir / "engine_1.json").read_text()) == response_b
        assert json.loads((rollout_dir / "engine_2.json").read_text()) == response_c
        assert manager.check_weights.calls == [((), {"action": "checksum"})]

    async def test_dump_without_rollout_id_writes_initial_dir(self, tmp_path: Path):
        """The pre-loop weight sync (rollout_id=None) is dumped under initial/."""
        manager = _FakeRolloutManagerHandle(checksum_result=[[[_make_engine_response({"w": "aa"})]]])
        dumper = EngineChecksumDumper.from_args(_make_args(str(tmp_path)), rollout_manager=manager)

        await dumper.dump(rollout_id=None)

        assert (tmp_path / "initial" / "engine_0.json").exists()

    async def test_dump_fails_when_no_engine_responses(self, tmp_path: Path):
        """An empty checksum result is a hard failure, not a silent empty dump."""
        manager = _FakeRolloutManagerHandle(checksum_result=[[[None]]])
        dumper = EngineChecksumDumper.from_args(_make_args(str(tmp_path)), rollout_manager=manager)

        with pytest.raises(AssertionError, match="no engine responses"):
            await dumper.dump(rollout_id=0)


def _write_dump(
    root: Path,
    side: str,
    *,
    rollouts: dict[str, list[dict]],
) -> str:
    side_dir = root / side
    for rollout_name, engine_responses in rollouts.items():
        rollout_dir = side_dir / rollout_name
        rollout_dir.mkdir(parents=True)
        for engine_index, response in enumerate(engine_responses):
            (rollout_dir / f"engine_{engine_index}.json").write_text(json.dumps(response))
    return str(side_dir)


_CHECKSUMS = {"model.w1": "0011", "model.w2": "2233"}


class TestCompareEngineChecksumDumps:
    def test_passes_on_identical_dumps(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        """Identical baseline/target trees pass and report the number of compared checksums."""
        rollouts = {
            "initial": [_make_engine_response(_CHECKSUMS), _make_engine_response(_CHECKSUMS)],
            "rollout_1": [_make_engine_response(_CHECKSUMS), _make_engine_response(_CHECKSUMS)],
        }
        baseline = _write_dump(tmp_path, "baseline", rollouts=rollouts)
        target = _write_dump(tmp_path, "target", rollouts=rollouts)

        compare_engine_checksum_dumps(
            baseline_dir=baseline,
            target_dir=target,
            expected_rollout_names={"initial", "rollout_1"},
            expected_num_engines=2,
        )

        out = capsys.readouterr().out
        assert "2 rollout(s), 4 engine file(s), 8 tensor checksum(s)" in out

    def test_fails_on_single_tensor_mismatch_with_tensor_name(self, tmp_path: Path):
        """A single differing tensor hash fails and names the rollout, engine, and tensor."""
        baseline = _write_dump(tmp_path, "baseline", rollouts={"rollout_1": [_make_engine_response(_CHECKSUMS)]})
        target = _write_dump(
            tmp_path,
            "target",
            rollouts={"rollout_1": [_make_engine_response({**_CHECKSUMS, "model.w2": "9999"})]},
        )

        with pytest.raises(AssertionError, match=r"(?s)rollout_1/engine_0\.json.*tensor 'model\.w2'") as exc_info:
            compare_engine_checksum_dumps(
                baseline_dir=baseline,
                target_dir=target,
                expected_rollout_names={"rollout_1"},
                expected_num_engines=1,
            )

        assert "model.w1" not in str(exc_info.value)

    def test_fails_on_missing_engine_file(self, tmp_path: Path):
        """A missing engine file on one side is a hard failure (fail-closed)."""
        baseline = _write_dump(
            tmp_path,
            "baseline",
            rollouts={"rollout_1": [_make_engine_response(_CHECKSUMS), _make_engine_response(_CHECKSUMS)]},
        )
        target = _write_dump(tmp_path, "target", rollouts={"rollout_1": [_make_engine_response(_CHECKSUMS)]})

        with pytest.raises(AssertionError, match="Engine checksum files mismatch"):
            compare_engine_checksum_dumps(
                baseline_dir=baseline,
                target_dir=target,
                expected_rollout_names={"rollout_1"},
                expected_num_engines=2,
            )

    def test_fails_on_rollout_dir_set_mismatch(self, tmp_path: Path):
        """A rollout dir present on only one side is a hard failure (fail-closed)."""
        baseline = _write_dump(
            tmp_path,
            "baseline",
            rollouts={
                "rollout_1": [_make_engine_response(_CHECKSUMS)],
                "rollout_2": [_make_engine_response(_CHECKSUMS)],
            },
        )
        target = _write_dump(tmp_path, "target", rollouts={"rollout_1": [_make_engine_response(_CHECKSUMS)]})

        with pytest.raises(AssertionError, match="rollout dirs mismatch"):
            compare_engine_checksum_dumps(
                baseline_dir=baseline,
                target_dir=target,
                expected_rollout_names={"rollout_1", "rollout_2"},
                expected_num_engines=1,
            )

    def test_fails_on_missing_baseline_dir(self, tmp_path: Path):
        """A missing dump dir fails instead of passing vacuously."""
        target = _write_dump(tmp_path, "target", rollouts={"rollout_1": [_make_engine_response(_CHECKSUMS)]})

        with pytest.raises(AssertionError, match="Baseline engine checksum dir does not exist"):
            compare_engine_checksum_dumps(
                baseline_dir=str(tmp_path / "baseline"),
                target_dir=target,
                expected_rollout_names={"rollout_1"},
                expected_num_engines=1,
            )

    def test_fails_on_empty_baseline_dir(self, tmp_path: Path):
        """An empty dump dir (no rollout dirs) fails instead of passing vacuously."""
        (tmp_path / "baseline").mkdir()
        target = _write_dump(tmp_path, "target", rollouts={"rollout_1": [_make_engine_response(_CHECKSUMS)]})

        with pytest.raises(AssertionError, match="rollout dirs mismatch"):
            compare_engine_checksum_dumps(
                baseline_dir=str(tmp_path / "baseline"),
                target_dir=target,
                expected_rollout_names={"rollout_1"},
                expected_num_engines=1,
            )

    def test_fails_on_tensor_name_set_mismatch(self, tmp_path: Path):
        """Different tensor name sets (e.g. a tensor never pushed) are a hard failure."""
        baseline = _write_dump(tmp_path, "baseline", rollouts={"rollout_1": [_make_engine_response(_CHECKSUMS)]})
        target = _write_dump(
            tmp_path,
            "target",
            rollouts={"rollout_1": [_make_engine_response({"model.w1": "0011"})]},
        )

        with pytest.raises(AssertionError, match="tensor name sets differ"):
            compare_engine_checksum_dumps(
                baseline_dir=baseline,
                target_dir=target,
                expected_rollout_names={"rollout_1"},
                expected_num_engines=1,
            )

    def test_fails_on_empty_checksum_map(self, tmp_path: Path):
        """An engine reporting zero tensors fails instead of passing vacuously."""
        baseline = _write_dump(tmp_path, "baseline", rollouts={"rollout_1": [_make_engine_response({})]})
        target = _write_dump(tmp_path, "target", rollouts={"rollout_1": [_make_engine_response({})]})

        with pytest.raises(AssertionError, match="no tensor checksums"):
            compare_engine_checksum_dumps(
                baseline_dir=baseline,
                target_dir=target,
                expected_rollout_names={"rollout_1"},
                expected_num_engines=1,
            )

    def test_fails_on_unsuccessful_response(self, tmp_path: Path):
        """A response with success=False fails even if checksums happen to match."""
        failed = {**_make_engine_response(_CHECKSUMS), "success": False}
        baseline = _write_dump(tmp_path, "baseline", rollouts={"rollout_1": [failed]})
        target = _write_dump(tmp_path, "target", rollouts={"rollout_1": [failed]})

        with pytest.raises(AssertionError, match="not successful"):
            compare_engine_checksum_dumps(
                baseline_dir=baseline,
                target_dir=target,
                expected_rollout_names={"rollout_1"},
                expected_num_engines=1,
            )

    def test_fails_on_rank_count_mismatch(self, tmp_path: Path):
        """Different per-engine rank counts are a hard failure."""
        baseline = _write_dump(
            tmp_path, "baseline", rollouts={"rollout_1": [_make_engine_response(_CHECKSUMS, num_ranks=2)]}
        )
        target = _write_dump(
            tmp_path, "target", rollouts={"rollout_1": [_make_engine_response(_CHECKSUMS, num_ranks=1)]}
        )

        with pytest.raises(AssertionError, match="rank count mismatch"):
            compare_engine_checksum_dumps(
                baseline_dir=baseline,
                target_dir=target,
                expected_rollout_names={"rollout_1"},
                expected_num_engines=1,
            )

    def test_passes_when_ranks_arrive_in_different_order(self, tmp_path: Path):
        """TP>1 ranks listed in different zmq arrival orders are still paired by global rank."""
        rank_0 = _make_rank_entry({"model.w1": "aaaa"}, rank=0, num_ranks=2)
        rank_1 = _make_rank_entry({"model.w1": "bbbb"}, rank=1, num_ranks=2)
        in_order = {"success": True, "message": "Success.", "ranks": [rank_0, rank_1]}
        reversed_order = {"success": True, "message": "Success.", "ranks": [rank_1, rank_0]}
        baseline = _write_dump(tmp_path, "baseline", rollouts={"rollout_1": [in_order]})
        target = _write_dump(tmp_path, "target", rollouts={"rollout_1": [reversed_order]})

        compare_engine_checksum_dumps(
            baseline_dir=baseline,
            target_dir=target,
            expected_rollout_names={"rollout_1"},
            expected_num_engines=1,
        )

    def test_fails_when_both_sides_symmetrically_miss_expected_rollouts(self, tmp_path: Path):
        """Identical trees that lack an expected rollout dir (e.g. all dumps in initial/) fail."""
        rollouts = {"initial": [_make_engine_response(_CHECKSUMS)]}
        baseline = _write_dump(tmp_path, "baseline", rollouts=rollouts)
        target = _write_dump(tmp_path, "target", rollouts=rollouts)

        with pytest.raises(AssertionError, match="rollout dirs mismatch"):
            compare_engine_checksum_dumps(
                baseline_dir=baseline,
                target_dir=target,
                expected_rollout_names={"initial", "rollout_1"},
                expected_num_engines=1,
            )

    def test_fails_when_both_sides_symmetrically_shrink_engine_set(self, tmp_path: Path):
        """Identical trees with fewer engine files than expected_num_engines fail."""
        rollouts = {"rollout_1": [_make_engine_response(_CHECKSUMS)]}
        baseline = _write_dump(tmp_path, "baseline", rollouts=rollouts)
        target = _write_dump(tmp_path, "target", rollouts=rollouts)

        with pytest.raises(AssertionError, match="Engine checksum files mismatch"):
            compare_engine_checksum_dumps(
                baseline_dir=baseline,
                target_dir=target,
                expected_rollout_names={"rollout_1"},
                expected_num_engines=2,
            )

    def test_fails_on_unexpected_engine_file_name(self, tmp_path: Path):
        """Engine files not named engine_0..engine_{n-1} fail even when both sides agree."""
        rollouts = {"rollout_1": [_make_engine_response(_CHECKSUMS)]}
        baseline = Path(_write_dump(tmp_path, "baseline", rollouts=rollouts))
        target = Path(_write_dump(tmp_path, "target", rollouts=rollouts))
        for side_dir in (baseline, target):
            (side_dir / "rollout_1" / "engine_0.json").rename(side_dir / "rollout_1" / "engine_9.json")

        with pytest.raises(AssertionError, match="Engine checksum files mismatch"):
            compare_engine_checksum_dumps(
                baseline_dir=str(baseline),
                target_dir=str(target),
                expected_rollout_names={"rollout_1"},
                expected_num_engines=1,
            )
