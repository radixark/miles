"""The version of the published sglang-miles wheel built from one sglang commit.

Usage: sglang_wheel_version.py <sglang_commit>

build-wheels.yml publishes every sglang-miles commit as <tag>.post<n>+miles.g<commit10> under the
sglang-miles-* releases of radixark/miles-wheels; that string is what a release pins.
"""

import json
import os
import re
import sys
import urllib.request

WHEELS_REPO = "radixark/miles-wheels"


def _api(path: str) -> list[dict]:
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get("GH_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    url, items = f"https://api.github.com/{path}", []
    while url:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            items += json.load(response)
            links = {
                rel: href for href, rel in re.findall(r'<([^>]+)>; rel="(\w+)"', response.headers.get("Link", ""))
            }
        url = links.get("next")
    return items


def wheel_version(commit: str, releases: list[dict] | None = None) -> str:
    if releases is None:
        releases = _api(f"repos/{WHEELS_REPO}/releases?per_page=100")
    marker = f"+miles.g{commit[:10]}-"
    for release in releases:
        if not release["tag_name"].startswith("sglang-miles-"):
            continue
        for asset in release["assets"]:
            name = asset["name"]
            if name.endswith(".whl") and marker in name:
                return name.split("-")[1]
    raise SystemExit(
        f"no sglang-miles wheel for {commit} in {WHEELS_REPO}; build-wheels.yml publishes one within the hour"
    )


if __name__ == "__main__":
    print(wheel_version(sys.argv[1]))
