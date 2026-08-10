#!/usr/bin/env python3
# doc-dev: docs/ci/02-docker-build.md
"""Resolve the upstream inputs a Miles image bakes.

Single source of truth for CI (docker-build.yml's resolve-upstream job) and for
local builds (docker/build.py fills in any pinned build-arg it wasn't given).
Prints one KEY=VALUE line per input; any unresolvable value is a hard error —
the build must never fall back to an unpinned branch HEAD.
"""

import argparse
import hashlib
import json
import subprocess
import urllib.request

WHEELS_REPO = "yueming-yuan/miles-wheels"

# Dockerfile build-arg -> (repo, branch) for the source trees the image checks out.
SOURCE_PINS = {
    "SGLANG_COMMIT": ("sgl-project/sglang", "sglang-miles"),
    "MEGATRON_COMMIT": ("radixark/Megatron-LM", "miles-main"),
    "MILES_COMMIT": ("radixark/miles", "main"),
}

# CLI output key (consumed as resolve-upstream job outputs) per build-arg.
OUTPUT_KEYS = {
    "SGLANG_COMMIT": "sglang_sha",
    "MEGATRON_COMMIT": "megatron_sha",
    "MILES_COMMIT": "miles_sha",
}

# CLI output key -> wheels release tag, for every rolling release the automatic
# builds install (cu13 multi-arch + cu12-x86).
WHEELS_TAG_OUTPUTS = {
    "wheels_fp_cu13_x86": "cu130-x86_64",
    "wheels_fp_cu13_arm64": "cu130-aarch64",
    "wheels_fp_cu12_x86": "cu129-x86_64",
}


def git_branch_head(repo: str, branch: str) -> str:
    out = subprocess.run(
        ["git", "ls-remote", f"https://github.com/{repo}.git", f"refs/heads/{branch}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if not out:
        raise SystemExit(f"no such branch: {repo}@{branch}")
    return out[0]


def wheels_fingerprint(tag: str) -> str:
    """Fingerprint a rolling release's asset list (name/id/size/updated_at).

    Rolling tags keep their name while assets get replaced, so the asset list — not
    a commit SHA — is what identifies the release content. Must stay byte-compatible
    with the fingerprints stored by resolve-upstream's rebuild gate.
    """
    url = f"https://api.github.com/repos/{WHEELS_REPO}/releases/tags/{tag}"
    with urllib.request.urlopen(url) as resp:
        assets = json.load(resp).get("assets", [])
    if not assets:
        raise SystemExit(f"wheels release {tag} has no assets")
    fp = sorted([a["name"], a["id"], a["size"], a["updated_at"]] for a in assets)
    return hashlib.sha256(repr(fp).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--miles-sha",
        default="",
        help="Use this miles commit instead of resolving main HEAD (CI passes the pushed commit).",
    )
    args = parser.parse_args()

    for arg_name, (repo, branch) in SOURCE_PINS.items():
        key = OUTPUT_KEYS[arg_name]
        if arg_name == "MILES_COMMIT" and args.miles_sha:
            print(f"{key}={args.miles_sha}")
            continue
        print(f"{key}={git_branch_head(repo, branch)}")
    for key, tag in WHEELS_TAG_OUTPUTS.items():
        print(f"{key}={wheels_fingerprint(tag)}")


if __name__ == "__main__":
    main()
