# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars>=1.0",
#     "fastexcel>=0.7",  # polars Excel engine, used by pl.read_excel()
#     "typer>=0.9",
# ]
# ///
"""Convert NVIDIA Xid-Catalog.xlsx into info.py with NON_AUTO_RECOVERABLE_XIDS frozenset.

Usage:
    uv run converter.py /path/to/Xid-Catalog.xlsx [--output-dir DIR]

The xlsx can be downloaded from:
    https://docs.nvidia.com/deploy/xid-errors/Xid-Catalog.xlsx
with doc at:
    https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html
"""
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

# Every resolution bucket from the xlsx must appear in exactly one of these two
# sets. This makes it easy to audit: if NVIDIA adds a new bucket in a future
# xlsx release, the validation in _audit_buckets will fail loudly, forcing a
# human to classify it.
NON_AUTO_RECOVERABLE_BUCKETS: frozenset[str] = frozenset(
    {
        "CONTACT_SUPPORT",
        "RESET_GPU",
        "RESTART_BM",
        "UPDATE_SWFW",
        "WORKFLOW_XID_48",
        "WORKFLOW_NVLINK_ERR",
        "WORKFLOW_NVLINK5_ERR",
    }
)

AUTO_RECOVERABLE_BUCKETS: frozenset[str | None] = frozenset(
    {
        "CHECK_MECHANICALS",
        "CHECK_UVM",
        "IGNORE",
        "RESTART_APP",
        "RESTART_VM",
        "WORKFLOW_XID_45",
        "XID_154",
        None,  # Not yet classified in the xlsx — review manually when regenerating
    }
)

_REQUIRED_COLUMNS: frozenset[str] = frozenset({"code", "mnemonic", "bucket"})

_DEFAULT_OUTPUT_DIR: Path = Path(__file__).resolve().parent


def main(
    xlsx_path: Annotated[Path, typer.Argument(help="Path to Xid-Catalog.xlsx")],
    output_dir: Annotated[Path, typer.Option(help="Output directory for info.py")] = _DEFAULT_OUTPUT_DIR,
) -> None:
    if not xlsx_path.exists():
        raise typer.BadParameter(f"File not found: {xlsx_path}")

    df = _read_xids_sheet(xlsx_path)
    _audit_buckets(df)

    non_auto_recoverable = df.filter(pl.col("bucket").is_in(NON_AUTO_RECOVERABLE_BUCKETS))
    unclassified = df.filter(pl.col("bucket").is_null())

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "info.py"
    source = _generate_info_py(non_auto_recoverable)
    compile(source, filename=str(output_path), mode="exec")
    output_path.write_text(source, encoding="utf-8")

    print(f"Found {len(non_auto_recoverable)} non-auto-recoverable XIDs from {xlsx_path.name}")
    for row in non_auto_recoverable.iter_rows(named=True):
        print(f"  XID {row['code']:3d}  {row['bucket']:<25s}  {row['mnemonic']}")

    if len(unclassified) > 0:
        typer.echo(
            f"\nWARNING: {len(unclassified)} XIDs have no resolution bucket (review manually):",
            err=True,
        )
        for row in unclassified.iter_rows(named=True):
            typer.echo(f"  XID {row['code']:3d}  {row['mnemonic']}", err=True)

    print(f"\nWritten to {output_path}")


def _read_xids_sheet(xlsx_path: Path) -> pl.DataFrame:
    try:
        df = pl.read_excel(xlsx_path, sheet_name="Xids")
    except Exception as exc:
        raise typer.BadParameter(f"Failed to read sheet 'Xids' from {xlsx_path}: {exc}") from exc

    col_map: dict[str, str] = {}
    for col in df.columns:
        normalized = col.strip().replace("\n", " ")
        if normalized == "Code":
            col_map[col] = "code"
        elif normalized == "Mnemonic":
            col_map[col] = "mnemonic"
        elif "Immediate Action" in normalized:
            col_map[col] = "bucket"

    missing = _REQUIRED_COLUMNS - set(col_map.values())
    if missing:
        raise ValueError(f"Could not map xlsx columns to: {missing}. " f"Available columns: {df.columns}")

    df = df.rename(col_map).select("code", "mnemonic", "bucket")

    return df.with_columns(
        pl.col("code").cast(pl.Int32),
        pl.col("bucket").str.strip_chars().replace("", None),
    ).sort("code")


def _audit_buckets(df: pl.DataFrame) -> None:
    all_known: frozenset[str] = NON_AUTO_RECOVERABLE_BUCKETS | frozenset(
        b for b in AUTO_RECOVERABLE_BUCKETS if b is not None
    )
    seen_buckets: set[str] = set(df["bucket"].drop_nulls().unique().to_list())
    unknown: set[str] = seen_buckets - all_known

    if unknown:
        raise ValueError(
            f"Unknown resolution bucket(s): {unknown}. "
            f"Add to NON_AUTO_RECOVERABLE_BUCKETS or AUTO_RECOVERABLE_BUCKETS in converter.py."
        )


def _generate_info_py(df: pl.DataFrame) -> str:
    lines: list[str] = [
        '"""NVIDIA XID codes that will NOT auto-recover — requires GPU reset, node reboot, or RMA.',
        "",
        "Auto-generated from Xid-Catalog.xlsx by converter.py.",
        "Source: https://docs.nvidia.com/deploy/xid-errors/Xid-Catalog.xlsx",
        '"""',
        "",
        "NON_AUTO_RECOVERABLE_XIDS: frozenset[int] = frozenset({",
    ]

    for row in df.iter_rows(named=True):
        lines.append(f"    {row['code']},  # {row['mnemonic']}, {row['bucket']}")

    lines.append("})")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    typer.run(main)
