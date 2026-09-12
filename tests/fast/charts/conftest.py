import shutil
import subprocess

import pytest

from tests.fast.charts.utils import chart_directories


@pytest.fixture(scope="session", autouse=True)
def vendored_dependencies():
    if shutil.which("helm") is None:
        return
    for chart_dir in chart_directories():
        if (chart_dir / "Chart.lock").exists():
            subprocess.run(["helm", "dependency", "build", str(chart_dir)], capture_output=True, check=True)
