#!/usr/bin/env bash
# A job container mounts its runner's whole work directory, which keeps a checkout of every
# repository the runner served; delete all but the starting job's own before it runs.
set -euo pipefail
shopt -s nullglob
work_dir="/data/miles_ci/runner_${HOSTNAME}"
for checkout in "$work_dir"/*/; do
  name=$(basename "$checkout")
  [[ $name == _* || $name == "${GITHUB_REPOSITORY#*/}" ]] || rm -rf -- "$checkout"
done
