#!/usr/bin/env bash
set -euo pipefail
profile_dir=$(mktemp -d)
trap 'rm -rf "$profile_dir"' EXIT
uv run python dev/scripts/profiling/verify.py "$profile_dir"
