#!/usr/bin/env bash
# A job container mounts its runner's whole work directory, which keeps a checkout of every
# repository the runner served; delete all but the starting job's own before it runs.
set -euo pipefail
shopt -s nullglob
# RUNNER_WORKSPACE is <work dir>/<repository name>, whatever layout the host gives the runner.
work_dir=$(dirname "$RUNNER_WORKSPACE")
for checkout in "$work_dir"/*/; do
  name=$(basename "$checkout")
  # Without the trailing slash, rm removes a symlink itself rather than its target's contents.
  [[ $name == _* || $name == "${GITHUB_REPOSITORY#*/}" ]] || rm -rf -- "${checkout%/}"
done
