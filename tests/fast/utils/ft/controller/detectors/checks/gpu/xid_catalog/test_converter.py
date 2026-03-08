"""Tests for the XID catalog converter logic (audit, codegen, column mapping)."""

from __future__ import annotations

import polars as pl
import pytest

from miles.utils.ft.controller.detectors.checks.gpu.xid_catalog.converter import (
    AUTO_RECOVERABLE_BUCKETS,
    NON_AUTO_RECOVERABLE_BUCKETS,
    _audit_buckets,
    _generate_info_py,
    _read_xids_sheet,
)


def _make_xid_df(
    rows: list[tuple[int, str, str | None]],
) -> pl.DataFrame:
    """Build a DataFrame matching the normalized schema (code, mnemonic, bucket)."""
    return pl.DataFrame(
        {"code": [r[0] for r in rows], "mnemonic": [r[1] for r in rows], "bucket": [r[2] for r in rows]},
        schema={"code": pl.Int32, "mnemonic": pl.Utf8, "bucket": pl.Utf8},
    )


# ---------------------------------------------------------------------------
# _audit_buckets
# ---------------------------------------------------------------------------


class TestAuditBuckets:
    def test_known_buckets_pass(self) -> None:
        df = _make_xid_df([
            (1, "GPU_LOST", "RESET_GPU"),
            (2, "GPU_RECOVERED", "IGNORE"),
            (3, "UNCLASSIFIED", None),
        ])
        _audit_buckets(df)

    def test_unknown_bucket_raises(self) -> None:
        df = _make_xid_df([
            (1, "GPU_LOST", "COMPLETELY_NEW_BUCKET"),
        ])
        with pytest.raises(ValueError, match="Unknown resolution bucket"):
            _audit_buckets(df)

    def test_all_non_auto_recoverable_buckets_accepted(self) -> None:
        rows = [(i, f"M{i}", bucket) for i, bucket in enumerate(NON_AUTO_RECOVERABLE_BUCKETS)]
        df = _make_xid_df(rows)
        _audit_buckets(df)

    def test_all_auto_recoverable_buckets_accepted(self) -> None:
        rows = [(i, f"M{i}", bucket) for i, bucket in enumerate(AUTO_RECOVERABLE_BUCKETS) if bucket is not None]
        df = _make_xid_df(rows)
        _audit_buckets(df)


# ---------------------------------------------------------------------------
# _generate_info_py
# ---------------------------------------------------------------------------


class TestGenerateInfoPy:
    def test_output_is_valid_python(self) -> None:
        df = _make_xid_df([
            (48, "DBE", "WORKFLOW_XID_48"),
            (79, "GPU_HAS_FALLEN_OFF_THE_BUS", "RESET_GPU"),
        ])
        source = _generate_info_py(df)
        compile(source, filename="<test>", mode="exec")

    def test_contains_frozenset_with_codes(self) -> None:
        df = _make_xid_df([
            (48, "DBE", "WORKFLOW_XID_48"),
            (79, "GPU_HAS_FALLEN_OFF_THE_BUS", "RESET_GPU"),
        ])
        source = _generate_info_py(df)
        assert "NON_AUTO_RECOVERABLE_XIDS" in source
        assert "48," in source
        assert "79," in source
        assert "frozenset" in source

    def test_empty_df_generates_empty_frozenset(self) -> None:
        df = _make_xid_df([])
        source = _generate_info_py(df)
        ns: dict = {}
        exec(source, ns)
        assert ns["NON_AUTO_RECOVERABLE_XIDS"] == frozenset()


# ---------------------------------------------------------------------------
# _read_xids_sheet (column mapping via in-memory xlsx)
# ---------------------------------------------------------------------------


class TestReadXidsSheet:
    def test_column_mapping_with_standard_names(self, tmp_path: pytest.TempPathFactory) -> None:
        """Round-trip: write a minimal xlsx, read it back through _read_xids_sheet."""
        xlsx_path = tmp_path / "test_xid.xlsx"  # type: ignore[operator]
        df = pl.DataFrame({
            "Code": [1, 2],
            "Mnemonic": ["FOO", "BAR"],
            "Immediate Action / Resolution Workflow": ["RESET_GPU", "IGNORE"],
        })
        df.write_excel(xlsx_path, worksheet="Xids")

        result = _read_xids_sheet(xlsx_path)
        assert "code" in result.columns
        assert "mnemonic" in result.columns
        assert "bucket" in result.columns
        assert result["code"].to_list() == [1, 2]

    def test_missing_column_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        xlsx_path = tmp_path / "bad.xlsx"  # type: ignore[operator]
        df = pl.DataFrame({"Code": [1], "Mnemonic": ["FOO"]})
        df.write_excel(xlsx_path, worksheet="Xids")

        with pytest.raises(ValueError, match="Could not map"):
            _read_xids_sheet(xlsx_path)


# ---------------------------------------------------------------------------
# Bucket completeness sanity check
# ---------------------------------------------------------------------------


class TestBucketSets:
    def test_no_overlap_between_auto_and_non_auto(self) -> None:
        overlap = NON_AUTO_RECOVERABLE_BUCKETS & {b for b in AUTO_RECOVERABLE_BUCKETS if b is not None}
        assert overlap == set(), f"Buckets in both sets: {overlap}"
