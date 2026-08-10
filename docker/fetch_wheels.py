#!/usr/bin/env python3
# doc-dev: docs/ci/02-docker-build.md
"""Populate /tmp/wheels from a miles-wheels release, via a persistent download cache.

Runs inside the wheels-download layer of docker/Dockerfile with:
  - /tmp/release.json        the GitHub release API response for the selected tag
  - $CACHE_DIR               a per-arch subdir of the /wheels-cache cache mount
  - /tmp/wheels              destination; pre-existing files (a locally COPY'd
                             wheel from <repo>/wheels/) win over the download

A cached asset is reused only when its recorded release asset id matches: rolling
releases replace assets under the same filename, so existence alone would pin
stale wheels forever. The layer itself re-runs whenever the WHEELS_FP build-arg
changes; this cache only saves re-downloading the assets that did not change.
"""

import json
import os
import shutil
import subprocess

cache = os.environ["CACHE_DIR"]
dest = "/tmp/wheels"

for asset in json.load(open("/tmp/release.json")).get("assets", []):
    name = asset["name"]
    if not name.endswith((".whl", ".tar.gz")):
        continue
    if os.path.exists(os.path.join(dest, name)):
        print(f"{name}: using locally provided copy")
        continue
    cached = os.path.join(cache, name)
    marker = os.path.join(cache, name + ".id")
    asset_id = str(asset["id"])
    if os.path.exists(cached) and os.path.exists(marker) and open(marker).read().strip() == asset_id:
        print(f"{name}: cache hit (asset id {asset_id})")
    else:
        print(f"{name}: downloading (asset id {asset_id})")
        subprocess.run(
            ["curl", "-fSL", "--retry", "3", "-o", cached, asset["browser_download_url"]],
            check=True,
        )
        with open(marker, "w") as f:
            f.write(asset_id)
    shutil.copy(cached, os.path.join(dest, name))
