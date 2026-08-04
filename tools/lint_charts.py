from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHARTS_DIR = REPO_ROOT / "charts"

VARIANTS: dict[str, list[list[str]]] = {}


def run(command: list[str]) -> subprocess.CompletedProcess:
    print("+ " + " ".join(command), file=sys.stderr)
    return subprocess.run(command, capture_output=True, text=True)


def all_charts() -> list[Path]:
    return sorted(chart_yaml.parent for chart_yaml in CHARTS_DIR.glob("*/Chart.yaml"))


def lint_chart(chart: Path) -> bool:
    if (chart / "Chart.lock").exists():
        built = run(["helm", "dependency", "build", str(chart)])
        if built.returncode != 0:
            print(built.stdout + built.stderr, file=sys.stderr)
            return False

    ok = True
    for extra in [[], *VARIANTS.get(chart.name, [])]:
        result = run(["helm", "lint", str(chart), *extra])
        if result.returncode != 0:
            print(result.stdout + result.stderr, file=sys.stderr)
            ok = False
    return ok


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description="helm lint every chart under charts/").parse_args(argv)

    if shutil.which("helm") is None:
        message = "helm is not installed"
        if os.environ.get("CI"):
            print(f"{message}; CI must provide it", file=sys.stderr)
            return 1
        print(f"{message}; skipping chart lint", file=sys.stderr)
        return 0

    return 0 if all([lint_chart(chart) for chart in all_charts()]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
