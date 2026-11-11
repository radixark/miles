"""The version of the published sglang-miles wheel built from one sglang commit.

Usage: sglang_wheel_version.py <sglang_commit>

build-wheels.yml publishes every sglang-miles commit as <tag>.post<n>+miles.g<commit10>, one wheel per
Python and arch, under the sglang-miles-* releases of radixark/miles-wheels. A release pins that version,
so every one of those wheels must be there.
"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

WHEELS_REPO = "radixark/miles-wheels"
ARCHES = ("x86_64", "aarch64")


def _pythons() -> list[str]:
    workflow = Path(__file__).parents[2] / "workflows" / "build-wheels.yml"
    return json.loads(re.search(r"SGLANG_PYTHONS: '(.*)'", workflow.read_text()).group(1))


def _releases() -> list[dict]:
    # shortcut: the 100 newest releases only, since a branch cut pins sglang-miles' head, whose wheels are in
    # this week's or last week's release; page through them if a cut ever pins an old commit
    request = urllib.request.Request(f"https://api.github.com/repos/{WHEELS_REPO}/releases?per_page=100")
    if token := os.environ.get("GH_TOKEN"):  # unauthenticated calls share the runner IP's 60 per hour
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def wheel_version(commit: str, releases: list[dict], pythons: list[str]) -> str:
    marker = f"+miles.g{commit[:10]}-"
    wheels = [
        asset["name"]
        for release in releases
        if release["tag_name"].startswith("sglang-miles-")
        for asset in release["assets"]
        if asset["name"].endswith(".whl") and marker in asset["name"]
    ]
    missing = [
        f"cp{python.replace('.', '')} {arch}"
        for python in pythons
        for arch in ARCHES
        if not any(f"-cp{python.replace('.', '')}-" in name and name.endswith(f"_{arch}.whl") for name in wheels)
    ]
    if missing:
        raise SystemExit(
            f"no sglang-miles wheel for {commit} on {', '.join(missing)} in {WHEELS_REPO}; "
            "build-wheels.yml publishes each within the hour, or shows why one failed"
        )
    return wheels[0].split("-")[1]


if __name__ == "__main__":
    print(wheel_version(sys.argv[1], _releases(), _pythons()))
