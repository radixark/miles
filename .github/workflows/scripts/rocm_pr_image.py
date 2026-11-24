#!/usr/bin/env python3
# doc-dev: docs/developer/ci/02-docker-build.md
"""Look up a public ROCm PR tag anonymously for /rerun-test.

Only Docker Hub's HTTP 404 selects the released default. Authentication, rate
limit, network and service errors fail the lookup instead of testing an older
image. No Docker config or publishing credentials are read.
"""

import argparse
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def published_tag(number):
    if not re.fullmatch(r"[1-9][0-9]*", number):
        raise ValueError("Invalid PR number for ROCm image lookup")
    tag = f"pr-{number}-rocm10-mi35x"
    url = f"https://hub.docker.com/v2/namespaces/radixark/repositories/miles/tags/{tag}"
    try:
        with urlopen(Request(url, method="HEAD"), timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"Unexpected Docker Hub status: {response.status}")
    except HTTPError as error:
        if error.code == 404:
            return ""
        raise
    return tag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr_number")
    args = parser.parse_args()
    try:
        print(published_tag(args.pr_number))
    except (ValueError, HTTPError, URLError, TimeoutError, RuntimeError) as error:
        print(f"::error::ROCm PR image lookup failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
