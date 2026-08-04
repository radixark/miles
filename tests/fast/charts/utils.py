import os
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CHARTS_DIR = REPO_ROOT / "charts"
SHARED_INFRA_SCHEMA_PATH = CHARTS_DIR / "shared-infra.schema.json"

NAMESPACE = "myns"


requires_helm = pytest.mark.skipif(
    shutil.which("helm") is None and not os.environ.get("CI"),
    reason="helm is required to render charts; CI installs it, so a CI run fails instead of skipping",
)


def library_chart_directories() -> list[Path]:
    return sorted(
        path.parent
        for path in SHARED_INFRA_SCHEMA_PATH.parent.glob("*/Chart.yaml")
        if "type: library" in path.read_text()
    )
