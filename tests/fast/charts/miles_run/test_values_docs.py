import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
CHART_VALUES = REPO_ROOT / "charts" / "miles-run" / "values.yaml"
PLATFORM_DOC = REPO_ROOT / "charts" / "miles-run" / "README.md"

_YAML_BLOCK = re.compile(r"```yaml\n(.*?)```", re.DOTALL)
_DEFAULTS_BLOCK_KEYS = {"infra", "run", "adhoc"}


def documented_chart_defaults(doc: Path) -> dict[str, Any]:
    blocks = [yaml.safe_load(match.group(1)) for match in _YAML_BLOCK.finditer(doc.read_text())]
    defaults = [block for block in blocks if isinstance(block, dict) and set(block) == _DEFAULTS_BLOCK_KEYS]
    assert len(defaults) == 1, f"{doc} should hold exactly one chart-defaults yaml block, found {len(defaults)}"
    return defaults[0]


class TestDocumentedChartDefaults:
    def test_the_documented_block_is_the_charts_own_default_values(self):
        """The page presents this block as what a bare install renders, so a drifted key misleads every reader."""
        assert documented_chart_defaults(PLATFORM_DOC) == yaml.safe_load(CHART_VALUES.read_text())
