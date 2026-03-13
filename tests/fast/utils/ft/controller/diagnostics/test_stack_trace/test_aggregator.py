"""Tests for StackTraceAggregator."""

from __future__ import annotations

import pytest

from tests.fast.utils.ft.utils import (
    SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK,
    SAMPLE_PYSPY_THREADS_NORMAL,
    SAMPLE_PYSPY_THREADS_STUCK,
    SAMPLE_PYSPY_THREADS_STUCK_FROM_BACKWARD,
)

from miles.utils.ft.agents.diagnostics.executors.stack_trace import PySpyFrame, PySpyThread
from miles.utils.ft.controller.diagnostics.stack_trace import StackTraceAggregator
from miles.utils.ft.controller.diagnostics.stack_trace.aggregator import StackTraceTieError


class TestStackTraceAggregatorBasic:
    def test_empty_traces_returns_empty(self) -> None:
        agg = StackTraceAggregator()
        result = agg.aggregate(traces={})
        assert result.suspect_node_ids == []

    def test_single_node_returns_empty(self) -> None:
        agg = StackTraceAggregator()
        result = agg.aggregate(traces={"node-0": SAMPLE_PYSPY_THREADS_NORMAL})
        assert result.suspect_node_ids == []

    def test_all_same_traces_returns_empty(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_THREADS_NORMAL,
            "node-1": SAMPLE_PYSPY_THREADS_NORMAL,
            "node-2": SAMPLE_PYSPY_THREADS_NORMAL,
        }
        result = agg.aggregate(traces=traces)
        assert result.suspect_node_ids == []


class TestStackTraceAggregatorSuspectDetection:
    def test_one_different_node_is_suspect(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_THREADS_STUCK,
            "node-1": SAMPLE_PYSPY_THREADS_STUCK,
            "node-2": SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK,
        }
        result = agg.aggregate(traces=traces)
        assert result.suspect_node_ids == ["node-2"]

    def test_two_nodes_all_different_raises_tie(self) -> None:
        """Two nodes with different fingerprints produce a tie (1 vs 1).
        Previously returned [], masking the ambiguity. Now raises StackTraceTieError."""
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_THREADS_STUCK,
            "node-1": SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK,
        }
        with pytest.raises(StackTraceTieError, match="unable to determine majority"):
            agg.aggregate(traces=traces)

    def test_multiple_minority_groups(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_THREADS_NORMAL,
            "node-1": SAMPLE_PYSPY_THREADS_NORMAL,
            "node-2": SAMPLE_PYSPY_THREADS_NORMAL,
            "node-3": SAMPLE_PYSPY_THREADS_STUCK,
            "node-4": SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK,
        }
        result = agg.aggregate(traces=traces)
        assert "node-3" in result.suspect_node_ids
        assert "node-4" in result.suspect_node_ids
        assert "node-0" not in result.suspect_node_ids

    def test_tied_groups_raises_tie(self) -> None:
        """2-2 split produces a tie. Previously returned [], masking the
        ambiguity. Now raises StackTraceTieError."""
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_THREADS_STUCK,
            "node-1": SAMPLE_PYSPY_THREADS_STUCK,
            "node-2": SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK,
            "node-3": SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK,
        }
        with pytest.raises(StackTraceTieError):
            agg.aggregate(traces=traces)

    def test_same_leaf_different_caller_detects_suspect(self) -> None:
        """forward->allreduce (majority) vs backward->allreduce (minority) are distinguished."""
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_THREADS_STUCK,
            "node-1": SAMPLE_PYSPY_THREADS_STUCK,
            "node-2": SAMPLE_PYSPY_THREADS_STUCK_FROM_BACKWARD,
        }
        result = agg.aggregate(traces=traces)
        assert result.suspect_node_ids == ["node-2"]


class TestStackTraceAggregatorFingerprint:
    def test_fingerprint_ignores_line_numbers(self) -> None:
        """Same function names but different line numbers produce same fingerprint."""
        agg = StackTraceAggregator()
        threads_a = [
            PySpyThread(
                thread_name="MainThread",
                active=True,
                owns_gil=False,
                frames=[
                    PySpyFrame(name="func_a", filename="file.py", line=10),
                    PySpyFrame(name="func_b", filename="file.py", line=20),
                ],
            ),
        ]
        threads_b = [
            PySpyThread(
                thread_name="MainThread",
                active=True,
                owns_gil=False,
                frames=[
                    PySpyFrame(name="func_a", filename="file.py", line=99),
                    PySpyFrame(name="func_b", filename="file.py", line=88),
                ],
            ),
        ]
        assert agg._extract_fingerprint(threads_a) == agg._extract_fingerprint(threads_b)

    def test_fingerprint_ignores_filename(self) -> None:
        """Same function names but different filenames produce same fingerprint."""
        agg = StackTraceAggregator()
        threads_a = [
            PySpyThread(
                thread_name="MainThread",
                active=True,
                owns_gil=False,
                frames=[PySpyFrame(name="func_a", filename="file_v1.py", line=10)],
            ),
        ]
        threads_b = [
            PySpyThread(
                thread_name="MainThread",
                active=True,
                owns_gil=False,
                frames=[PySpyFrame(name="func_a", filename="file_v2.py", line=10)],
            ),
        ]
        assert agg._extract_fingerprint(threads_a) == agg._extract_fingerprint(threads_b)

    def test_fingerprint_differs_on_function_name(self) -> None:
        agg = StackTraceAggregator()
        threads_a = [
            PySpyThread(
                thread_name="MainThread",
                active=True,
                owns_gil=False,
                frames=[PySpyFrame(name="func_a", filename="file.py", line=10)],
            ),
        ]
        threads_b = [
            PySpyThread(
                thread_name="MainThread",
                active=True,
                owns_gil=False,
                frames=[PySpyFrame(name="func_DIFFERENT", filename="file.py", line=10)],
            ),
        ]
        assert agg._extract_fingerprint(threads_a) != agg._extract_fingerprint(threads_b)

    def test_fingerprint_uses_top_n_frames(self) -> None:
        """Fingerprint includes multiple inner frames but truncates beyond max_frames."""
        agg = StackTraceAggregator(max_frames=2)
        threads = [
            PySpyThread(
                thread_name="MainThread",
                active=True,
                owns_gil=False,
                frames=[
                    PySpyFrame(name="innermost_func", filename="inner.py", line=1),
                    PySpyFrame(name="middle_func", filename="mid.py", line=2),
                    PySpyFrame(name="outermost_func", filename="outer.py", line=3),
                ],
            ),
        ]
        fp = agg._extract_fingerprint(threads)
        assert "innermost_func" in fp
        assert "middle_func" in fp
        assert "outermost_func" not in fp

    def test_same_leaf_different_caller_produces_different_fingerprint(self) -> None:
        """Same innermost frame but different caller produces different fingerprint."""
        agg = StackTraceAggregator()
        threads_forward = [
            PySpyThread(
                thread_name="MainThread",
                active=True,
                owns_gil=False,
                frames=[
                    PySpyFrame(name="nccl_allreduce", filename="nccl.py", line=1),
                    PySpyFrame(name="forward", filename="model.py", line=10),
                ],
            ),
        ]
        threads_backward = [
            PySpyThread(
                thread_name="MainThread",
                active=True,
                owns_gil=False,
                frames=[
                    PySpyFrame(name="nccl_allreduce", filename="nccl.py", line=1),
                    PySpyFrame(name="backward", filename="model.py", line=20),
                ],
            ),
        ]
        assert agg._extract_fingerprint(threads_forward) != agg._extract_fingerprint(threads_backward)

    def test_fingerprint_skips_empty_frame_threads(self) -> None:
        agg = StackTraceAggregator()
        threads = [
            PySpyThread(
                thread_name="EmptyThread",
                active=False,
                owns_gil=False,
                frames=[],
            ),
        ]
        assert agg._extract_fingerprint(threads) == ""

    def test_real_sample_produces_nonempty_fingerprint(self) -> None:
        agg = StackTraceAggregator()
        fp = agg._extract_fingerprint(SAMPLE_PYSPY_THREADS_NORMAL)
        assert fp != ""


class TestEmptyFingerprintHandling:
    def test_node_with_empty_fingerprint_is_suspect(self) -> None:
        """Previously a node whose threads all had empty frames produced
        fingerprint "" which was treated as a valid group in majority
        voting. This could mask the real outlier or produce misleading
        aggregation. Now empty-fingerprint nodes are excluded from
        fingerprint grouping and always marked as suspects."""
        empty_threads = [
            PySpyThread(thread_name="T", active=False, owns_gil=False, frames=[]),
        ]
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_THREADS_NORMAL,
            "node-1": SAMPLE_PYSPY_THREADS_NORMAL,
            "node-empty": empty_threads,
        }
        result = agg.aggregate(traces=traces)
        assert "node-empty" in result.suspect_node_ids

    def test_all_nodes_empty_fingerprint_returns_all_suspect(self) -> None:
        empty_threads = [
            PySpyThread(thread_name="T", active=False, owns_gil=False, frames=[]),
        ]
        agg = StackTraceAggregator()
        traces = {
            "node-0": empty_threads,
            "node-1": empty_threads,
        }
        result = agg.aggregate(traces=traces)
        assert sorted(result.suspect_node_ids) == ["node-0", "node-1"]


class TestAggregationResult:
    def test_result_contains_fingerprint_groups(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_THREADS_STUCK,
            "node-1": SAMPLE_PYSPY_THREADS_STUCK,
            "node-2": SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK,
        }
        result = agg.aggregate(traces=traces)

        assert len(result.fingerprint_groups) == 2
        group_sizes = sorted(len(nodes) for nodes in result.fingerprint_groups.values())
        assert group_sizes == [1, 2]

    def test_result_contains_raw_traces(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_THREADS_NORMAL,
            "node-1": SAMPLE_PYSPY_THREADS_NORMAL,
        }
        result = agg.aggregate(traces=traces)

        assert result.raw_traces == traces

    def test_result_serializable_to_json(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_THREADS_STUCK,
            "node-1": SAMPLE_PYSPY_THREADS_STUCK,
            "node-2": SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK,
        }
        result = agg.aggregate(traces=traces)

        json_str = result.model_dump_json()
        assert "node-0" in json_str
        assert "node-2" in json_str
