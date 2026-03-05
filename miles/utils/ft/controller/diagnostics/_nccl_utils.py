from __future__ import annotations

import re

_AVG_BUS_BW_PATTERN = re.compile(r"#\s*Avg bus bandwidth\s*:\s*([\d.]+)")
_BUSBW_COLUMN_INDEX = 7


def parse_avg_bus_bandwidth(output: str) -> float | None:
    """Parse average bus bandwidth (GB/s) from nccl-tests text output.

    Primary path: look for the ``# Avg bus bandwidth`` summary line.
    Fallback: parse the last data row and extract the busbw column
    (column index 7, out-of-place, 0-indexed).
    """
    match = _AVG_BUS_BW_PATTERN.search(output)
    if match:
        return float(match.group(1))

    last_data_row: list[str] | None = None
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) > _BUSBW_COLUMN_INDEX:
            try:
                float(parts[0])
                last_data_row = parts
            except ValueError:
                continue

    if last_data_row is not None:
        try:
            return float(last_data_row[_BUSBW_COLUMN_INDEX])
        except (IndexError, ValueError):
            return None

    return None
