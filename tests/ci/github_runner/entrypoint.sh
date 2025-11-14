#!/bin/sh
set -eux

# ref: https://github.com/actions/runner/issues/367#issuecomment-2007558723
# ref: https://github.com/actions/runner
# args ref: https://github.com/actions/runner/blob/68ff57dbc4c836d50f46602a8a53301fb9513eb4/src/Runner.Listener/CommandSettings.cs#L53

DIR_BASE="/data/miles_ci/runner_$(hostname)"

mkdir -p "$DIR_BASE"
mkdir -p "$DIR_BASE/github_actions_work"

cd $DIR_BASE

CONFIG_EXECUTED_FLAG_PATH="$DIR_BASE/miles_runner_config_executed.txt"

if [ ! -f "$CONFIG_EXECUTED_FLAG_PATH" ]; then
  /home/runner/config.sh --url ${GITHUB_RUNNER_URL} --token ${GITHUB_RUNNER_TOKEN} --unattended --work "$DIR_BASE/github_actions_work"
  echo "configured" > "$CONFIG_EXECUTED_FLAG_PATH"
else
  echo "config.sh already executed, skipping"
fi

exec /home/runner/run.sh
