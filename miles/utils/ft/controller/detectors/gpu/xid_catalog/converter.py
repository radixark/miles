# /// script
# requires-python = ">=3.10"
# dependencies = ["openpyxl>=3.1", "typer>=0.9"]
# ///
"""Convert NVIDIA Xid-Catalog.xlsx into info.py with FATAL_XIDS frozenset.

Usage:
    uv run converter.py /path/to/Xid-Catalog.xlsx [--output-dir DIR]

The xlsx can be downloaded from:
    https://docs.nvidia.com/deploy/xid-errors/Xid-Catalog.xlsx
"""
import logging
from pathlib import Path
from typing import Annotated

import openpyxl
import typer

logger = logging.getLogger(__name__)

FATAL_RESOLUTION_BUCKETS: frozenset[str] = frozenset({
    "RESET_GPU",
    "RESTART_BM",
    "WORKFLOW_XID_48",
    "WORKFLOW_NVLINK_ERR",
    "WORKFLOW_NVLINK5_ERR",
})


def _extract_fatal_xids(xlsx_path: Path) -> list[tuple[int, str, str]]:
    """Return sorted list of (code, mnemonic, resolution_bucket) for fatal XIDs."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["Xids"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    header = [str(h).strip() for h in rows[0]]
    code_idx = header.index("Code")
    mnemonic_idx = header.index("Mnemonic")
    bucket_idx = next(i for i, h in enumerate(header) if "Immediate Action" in h)

    results: list[tuple[int, str, str]] = []
    for row in rows[1:]:
        bucket = str(row[bucket_idx]).strip() if row[bucket_idx] else ""
        if bucket not in FATAL_RESOLUTION_BUCKETS:
            continue
        code = int(row[code_idx])
        mnemonic = str(row[mnemonic_idx]).strip()
        results.append((code, mnemonic, bucket))

    results.sort(key=lambda x: x[0])
    return results


def _generate_info_py(fatal_xids: list[tuple[int, str, str]]) -> str:
    lines: list[str] = [
        '"""NVIDIA XID codes that require GPU reset, node reboot, or hardware replacement.',
        "",
        "Auto-generated from Xid-Catalog.xlsx by converter.py.",
        "Source: https://docs.nvidia.com/deploy/xid-errors/Xid-Catalog.xlsx",
        '"""',
        "",
        "FATAL_XIDS: frozenset[int] = frozenset({",
    ]
    for code, mnemonic, bucket in fatal_xids:
        lines.append(f"    {code},  # {mnemonic}, {bucket}")
    lines.append("})")
    lines.append("")
    return "\n".join(lines)


def main(
    xlsx_path: Annotated[Path, typer.Argument(help="Path to Xid-Catalog.xlsx")],
    output_dir: Annotated[
        Path, typer.Option(help="Output directory for info.py")
    ] = Path(__file__).parent,
) -> None:
    if not xlsx_path.exists():
        raise typer.BadParameter(f"File not found: {xlsx_path}")

    fatal_xids = _extract_fatal_xids(xlsx_path)
    print(f"Found {len(fatal_xids)} fatal XIDs from {xlsx_path.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "info.py"
    output_path.write_text(_generate_info_py(fatal_xids), encoding="utf-8")
    print(f"Written to {output_path}")

    for code, mnemonic, bucket in fatal_xids:
        print(f"  XID {code:3d}  {bucket:<25s}  {mnemonic}")


if __name__ == "__main__":
    typer.run(main)
