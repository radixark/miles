"""Tests for engine_weight_checksum.flatten_inference_engine_checksums."""

from typing import Any

import pytest

from miles.utils.audit_utils.checksum_utils import flatten_inference_engine_checksums


def _engine_body(*, success: bool, ranks: list[dict[str, Any]] | None) -> dict[str, Any]:
    body: dict[str, Any] = {"success": success, "message": "ok"}
    if ranks is not None:
        body["ranks"] = ranks
    return body


def _rank(rank: int, checksums: dict[str, str]) -> dict[str, Any]:
    return {"checksums": checksums, "parallelism_info": [{"role": "target", "rank": rank}]}


class TestFlattenInferenceEngineChecksums:
    def test_single_group_engine_single_rank(self) -> None:
        """One server-group/engine with one rank yields one prefixed checksum dict."""
        result = [[_engine_body(success=True, ranks=[_rank(0, {"w": "aaa"})])]]
        assert flatten_inference_engine_checksums(result) == [{"rank0/w": "aaa"}]

    def test_multiple_engines_flattened_in_order(self) -> None:
        """Engines across server-groups are flattened into a single ordered list."""
        result = [
            [_engine_body(success=True, ranks=[_rank(0, {"w": "e0"})])],
            [_engine_body(success=True, ranks=[_rank(0, {"w": "e1"})])],
        ]
        assert flatten_inference_engine_checksums(result) == [{"rank0/w": "e0"}, {"rank0/w": "e1"}]

    def test_none_node_rank_payloads_filtered(self) -> None:
        """None engine bodies (non-zero node ranks) are dropped before indexing."""
        result = [[_engine_body(success=True, ranks=[_rank(0, {"w": "e0"})]), None]]
        assert flatten_inference_engine_checksums(result) == [{"rank0/w": "e0"}]

    def test_all_none_fails_loud(self) -> None:
        """A None-only result means the checksum action did nothing, so fail loud."""
        result = [[None, None]]
        with pytest.raises(AssertionError, match="no non-None engine bodies"):
            flatten_inference_engine_checksums(result)

    def test_multi_rank_merged_with_rank_prefix(self) -> None:
        """Multiple ranks of one engine merge into one dict, prefixed by rank to avoid clobber."""
        result = [[_engine_body(success=True, ranks=[_rank(0, {"w": "r0"}), _rank(1, {"w": "r1"})])]]
        assert flatten_inference_engine_checksums(result) == [{"rank0/w": "r0", "rank1/w": "r1"}]

    def test_ranks_out_of_order_sorted_by_parallelism_rank(self) -> None:
        """Ranks arriving out of order (zmq) are sorted by parallelism rank deterministically."""
        out_of_order = [[_engine_body(success=True, ranks=[_rank(1, {"w": "r1"}), _rank(0, {"w": "r0"})])]]
        in_order = [[_engine_body(success=True, ranks=[_rank(0, {"w": "r0"}), _rank(1, {"w": "r1"})])]]
        assert flatten_inference_engine_checksums(out_of_order) == flatten_inference_engine_checksums(in_order)

    def test_engine_failure_fails_loud(self) -> None:
        """An engine reporting success=False fails loud rather than silently dropping it."""
        result = [[_engine_body(success=False, ranks=[_rank(0, {"w": "aaa"})])]]
        with pytest.raises(AssertionError, match="reported failure"):
            flatten_inference_engine_checksums(result)

    def test_engine_without_ranks_fails_loud(self) -> None:
        """A success body with no ranks fails loud (nothing to compare)."""
        result = [[_engine_body(success=True, ranks=None)]]
        with pytest.raises(AssertionError, match="no ranks"):
            flatten_inference_engine_checksums(result)

    def test_multi_role_parallelism_info_collapses_to_one_gpu_rank(self) -> None:
        """Per-role parallelism_info (target + draft) sharing a GPU rank uses that single rank."""
        rank_info = {
            "checksums": {"w": "aaa"},
            "parallelism_info": [{"role": "target", "rank": 0}, {"role": "draft", "rank": 0}],
        }
        result = [[_engine_body(success=True, ranks=[rank_info])]]
        assert flatten_inference_engine_checksums(result) == [{"rank0/w": "aaa"}]

    def test_parallelism_info_rank_disagreement_fails_loud(self) -> None:
        """Roles reporting different GPU ranks for one engine rank is a contract violation."""
        rank_info = {
            "checksums": {"w": "aaa"},
            "parallelism_info": [{"role": "target", "rank": 0}, {"role": "draft", "rank": 1}],
        }
        result = [[_engine_body(success=True, ranks=[rank_info])]]
        with pytest.raises(AssertionError, match="one GPU rank"):
            flatten_inference_engine_checksums(result)
