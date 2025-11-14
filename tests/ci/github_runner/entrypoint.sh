#!/bin/sh
set -eux

# ref: https://github.com/actions/runner/issues/367#issuecomment-2007558723
# ref: https://github.com/actions/runner
# args ref: https://github.com/actions/runner/blob/68ff57dbc4c836d50f46602a8a53301fb9513eb4/src/Runner.Listener/CommandSettings.cs#L53

DIR="/data/miles_ci/runner_$(hostname)"
FLAG="$DIR/miles_runner_config_executed.txt"

mkdir -p "$DIR"

if [ ! -f "$FLAG" ]; then
  /home/runner/config.sh --url ${GITHUB_RUNNER_URL} --token ${GITHUB_RUNNER_TOKEN} --unattended --work "$DIR"
  echo "configured" > "$FLAG"
else
  echo "config.sh already executed, skipping"
fi

exec /home/runner/run.sh
