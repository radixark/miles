#!/usr/bin/env bash
set -euo pipefail
/data/miles_ci/runner-hooks/remove-foreign-checkouts.sh
exec python3 /data/miles_ci/runner-hooks/b200_job_lock.py
